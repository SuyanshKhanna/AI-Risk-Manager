"""
UPI Fraud Detector — Hackathon MVP (LightGBM + SHAP).

Single-file, CPU-only fraud detector with:
- LightGBM binary classifier on engineered features
- SHAP explainability for every prediction
- Rule-based decision engine (ALLOW / CHALLENGE / BLOCK)
- Honest metrics: precision, recall, false-positive cost on held-out split
"""

import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore", category=UserWarning)

# ──────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """Training configuration."""
    # Data
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    
    # LightGBM params (tuned for imbalanced fraud data)
    lgb_params: Dict = None
    
    # Calibration
    calibrate: bool = True
    calibration_method: str = "isotonic"  # "sigmoid" or "isotonic"
    calibration_cv: int = 3
    
    # Decision thresholds
    challenge_threshold: float = 0.3   # score >= this → CHALLENGE
    block_threshold: float = 0.7       # score >= this → BLOCK
    
    # False-positive cost model (per merchant tier)
    fp_cost_per_blocked_txn: float = 150.0  # INR
    
    def __post_init__(self):
        if self.lgb_params is None:
            self.lgb_params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "num_leaves": 63,
                "learning_rate": 0.05,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "min_child_samples": 20,
                "scale_pos_weight": 100,  # approx 1:100 fraud ratio
                "verbosity": -1,
                "random_state": self.random_state,
                "n_jobs": -1,
            }

# ──────────────────────────────────────────────────────────────────────
# DECISION ENGINE
# ──────────────────────────────────────────────────────────────────────

class DecisionEngine:
    """Rule layer + SHAP explainability on top of model scores."""
    
    def __init__(self, config: ModelConfig, model, shap_explainer):
        self.config = config
        self.model = model
        self.explainer = shap_explainer
    
    def decide(self, features: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Score transactions and return decisions with plain-language reasons.
        
        Returns list of dicts with: score, action, reasons, shap_values
        """
        # Get probabilities
        if hasattr(self.model, "predict_proba"):
            probas = self.model.predict_proba(features)[:, 1]
        else:
            probas = self.model.predict(features, num_iteration=self.model.best_iteration)
        
        decisions = []
        for i, score in enumerate(probas):
            action, reasons = self._apply_rules(score, features.iloc[i])
            
            # SHAP explanation for this prediction
            shap_vals = self._get_shap_reasons(features.iloc[i:i+1], i)
            
            decisions.append({
                "score": float(score),
                "action": action,
                "reasons": reasons,
                "shap_top3": shap_vals,
                "model_version": getattr(self.model, "model_version", "1.0"),
            })
        
        return decisions
    
    def _apply_rules(self, score: float, row: pd.Series) -> Tuple[str, List[str]]:
        """Apply rule layer on top of model score."""
        reasons = []
        
        if score >= self.config.block_threshold:
            action = "BLOCK"
            reasons.append(f"High fraud risk score: {score:.2f} (threshold: {self.config.block_threshold})")
            
            # Add specific rule triggers
            if row.get("settlement_verified", 1) == 0:
                reasons.append("Settlement not verified by payment gateway")
            if row.get("qr_mismatch", 0) == 1:
                reasons.append("QR code hash mismatch detected")
            if row.get("remote_access_app", 0) == 1:
                reasons.append("Remote access application detected during transaction")
            if row.get("velocity_5min", 0) > 10:
                reasons.append(f"Unusually high velocity: {row['velocity_5min']} txns in 5 min")
                
        elif score >= self.config.challenge_threshold:
            action = "CHALLENGE"
            reasons.append(f"Elevated fraud risk score: {score:.2f} (threshold: {self.config.challenge_threshold})")
            reasons.append("Additional verification recommended (OTP / app confirm)")
            
            if row.get("device_emulator_score", 0) > 0.5:
                reasons.append("Device appears to be emulator/rooted")
            if row.get("vpn_probability", 0) > 0.5:
                reasons.append("VPN/proxy detected")
            if row.get("is_night", 0) == 1:
                reasons.append("Transaction during unusual hours (22:00-06:00)")
                
        else:
            action = "ALLOW"
            reasons.append(f"Low fraud risk score: {score:.2f}")
        
        return action, reasons
    
    def _get_shap_reasons(self, row_df: pd.DataFrame, idx: int) -> List[Dict]:
        """Extract top 3 SHAP contributors for this prediction."""
        try:
            shap_values = self.explainer(row_df)
            vals = shap_values.values[0]  # shape: (n_features,)
            feat_names = row_df.columns.tolist()
            
            # Get top 3 by absolute contribution
            top_idx = np.argsort(np.abs(vals))[-3:][::-1]
            return [
                {"feature": feat_names[i], "contribution": float(vals[i]), "value": float(row_df.iloc[0, i])}
                for i in top_idx
            ]
        except Exception:
            return []

# ──────────────────────────────────────────────────────────────────────
# TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────────────

class UpiFraudDetector:
    """End-to-end fraud detector: train → calibrate → explain → decide."""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.model = None
        self.calibrated_model = None
        self.explainer = None
        self.decision_engine = None
        self.feature_columns = None
        self.metrics = {}
        self.model_version = f"upi-lgbm-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train LightGBM with calibration and SHAP explainer.
        Returns metrics dict with precision, recall, PR-AUC, FP cost.
        """
        print(f"Training on {len(X):,} samples, {X.shape[1]} features")
        print(f"Fraud rate: {y.mean()*100:.3f}%")
        
        self.feature_columns = X.columns.tolist()
        
        # Split: train / val / test
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(self.config.test_size + self.config.val_size),
            stratify=y, random_state=self.config.random_state
        )
        val_ratio = self.config.val_size / (self.config.test_size + self.config.val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1-val_ratio), stratify=y_temp,
            random_state=self.config.random_state
        )
        
        print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
        
        # Train LightGBM with early stopping
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        
        self.model = lgb.train(
            self.config.lgb_params,
            lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_train, lgb_val],
            valid_names=["train", "val"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=50),
            ],
        )
        
        # Calibrate probabilities
        if self.config.calibrate:
            print("Calibrating probabilities...")
            base_clf = lgb.LGBMClassifier(**self.config.lgb_params, n_estimators=self.model.best_iteration)
            base_clf.fit(X_train, y_train)
            
            # Check if we have enough samples for CV calibration
            n_val_fraud = y_val.sum()
            n_val_legit = len(y_val) - n_val_fraud
            min_class = min(n_val_fraud, n_val_legit)
            
            if min_class >= self.config.calibration_cv:
                self.calibrated_model = CalibratedClassifierCV(
                    base_clf, method=self.config.calibration_method, cv=self.config.calibration_cv
                )
                self.calibrated_model.fit(X_val, y_val)
            else:
                # Fallback: use base model predictions with isotonic regression on validation
                # For small validation sets, we just use the base model without calibration
                print(f"  Not enough samples for calibration (min class: {min_class}), skipping calibration")
                self.calibrated_model = base_clf
            scoring_model = self.calibrated_model
        else:
            scoring_model = self.model
        
        # Evaluate on held-out test set
        if hasattr(scoring_model, 'predict_proba'):
            test_probas = scoring_model.predict_proba(X_test)[:, 1]
        else:
            # Raw LightGBM Booster
            test_probas = scoring_model.predict(X_test, num_iteration=scoring_model.best_iteration)
        self.metrics = self._compute_metrics(y_test, test_probas)
        
        # SHAP explainer (TreeExplainer is fast for LightGBM)
        print("Building SHAP explainer...")
        self.explainer = shap.TreeExplainer(self.model)
        
        # Decision engine
        self.decision_engine = DecisionEngine(self.config, scoring_model, self.explainer)
        
        # Print results
        self._print_metrics()
        
        return self.metrics
    
    def _compute_metrics(self, y_true: np.ndarray, y_score: np.ndarray) -> Dict:
        """Compute honest metrics on test set."""
        # PR-AUC
        pr_auc = average_precision_score(y_true, y_score)
        
        # ROC-AUC
        roc_auc = roc_auc_score(y_true, y_score)
        
        # Precision/Recall at decision thresholds
        results = {"pr_auc": pr_auc, "roc_auc": roc_auc}
        
        for name, thresh in [("challenge", self.config.challenge_threshold), 
                              ("block", self.config.block_threshold)]:
            y_pred = (y_score >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            # False-positive cost
            fp_cost = fp * self.config.fp_cost_per_blocked_txn
            
            results[f"{name}_precision"] = precision
            results[f"{name}_recall"] = recall
            results[f"{name}_fpr"] = fpr
            results[f"{name}_fp_cost"] = fp_cost
            results[f"{name}_tp"] = int(tp)
            results[f"{name}_fp"] = int(fp)
            results[f"{name}_fn"] = int(fn)
            results[f"{name}_tn"] = int(tn)
        
        return results
    
    def _print_metrics(self):
        m = self.metrics
        print("\n=== Test Set Metrics (Held-Out) ===")
        print("PR-AUC:     {:.4f}".format(m['pr_auc']))
        print("ROC-AUC:    {:.4f}".format(m['roc_auc']))
        print("\nAt CHALLENGE threshold ({}):".format(self.config.challenge_threshold))
        print("  Precision: {:.4f} | Recall: {:.4f} | FPR: {:.4f}".format(
            m['challenge_precision'], m['challenge_recall'], m['challenge_fpr']))
        print("  FP Cost:   INR {:,.0f}".format(m['challenge_fp_cost']))
        print("\nAt BLOCK threshold ({}):".format(self.config.block_threshold))
        print("  Precision: {:.4f} | Recall: {:.4f} | FPR: {:.4f}".format(
            m['block_precision'], m['block_recall'], m['block_fpr']))
        print("  FP Cost:   INR {:,.0f}".format(m['block_fp_cost']))
        
        return results
    
    def _print_metrics(self):
        m = self.metrics
        print("\n=== Test Set Metrics (Held-Out) ===")
        print("PR-AUC:     {:.4f}".format(m['pr_auc']))
        print("ROC-AUC:    {:.4f}".format(m['roc_auc']))
        print("\nAt CHALLENGE threshold ({}):".format(self.config.challenge_threshold))
        print("  Precision: {:.4f} | Recall: {:.4f} | FPR: {:.4f}".format(
            m['challenge_precision'], m['challenge_recall'], m['challenge_fpr']))
        print("  FP Cost:   INR {:,.0f}".format(m['challenge_fp_cost']))
        print("\nAt BLOCK threshold ({}):".format(self.config.block_threshold))
        print("  Precision: {:.4f} | Recall: {:.4f} | FPR: {:.4f}".format(
            m['block_precision'], m['block_recall'], m['block_fpr']))
        print("  FP Cost:   INR {:,.0f}".format(m['block_fp_cost']))
    
    def predict(self, features: pd.DataFrame) -> List[Dict]:
        """Score new transactions and return decisions."""
        if self.decision_engine is None:
            raise RuntimeError("Model not trained. Call train() first.")
        
        # Ensure feature alignment
        X = features.reindex(columns=self.feature_columns, fill_value=0)
        return self.decision_engine.decide(X)
    
    def save(self, path: str):
        """Save model, explainer, config to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save LightGBM model
        self.model.save_model(str(path / "model.txt"))
        
        # Save calibrated model if exists
        if self.calibrated_model:
            with open(path / "calibrated_model.pkl", "wb") as f:
                pickle.dump(self.calibrated_model, f)
        
        # Save config and feature columns
        with open(path / "config.json", "w") as f:
            json.dump({
                "config": asdict(self.config),
                "feature_columns": self.feature_columns,
                "model_version": self.model_version,
                "metrics": self.metrics,
            }, f, indent=2, default=str)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> "UpiFraudDetector":
        """Load model from disk."""
        path = Path(path)
        
        with open(path / "config.json") as f:
            saved = json.load(f)
        
        config = ModelConfig(**saved["config"])
        detector = cls(config)
        detector.feature_columns = saved["feature_columns"]
        detector.model_version = saved["model_version"]
        detector.metrics = saved["metrics"]
        
        # Load model
        detector.model = lgb.Booster(model_file=str(path / "model.txt"))
        
        # Load calibrated model
        if (path / "calibrated_model.pkl").exists():
            with open(path / "calibrated_model.pkl", "rb") as f:
                detector.calibrated_model = pickle.load(f)
        
        # Rebuild explainer and decision engine
        detector.explainer = shap.TreeExplainer(detector.model)
        scoring_model = detector.calibrated_model or detector.model
        detector.decision_engine = DecisionEngine(config, scoring_model, detector.explainer)
        
        return detector


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def load_training_data(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Load training data from CSV/Parquet."""
    path = Path(path)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
    
    # Assume last column is label or explicit 'label' column
    if "label" in df.columns:
        y = df["label"]
        X = df.drop(columns=["label"])
    else:
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
    
    return X, y


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="UPI Fraud Detector - Train / Predict")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--data", required=True, help="Training data (CSV/Parquet)")
    train_parser.add_argument("--output", default="models/upi_fraud_lgbm", help="Output model directory")
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--val-size", type=float, default=0.1)
    train_parser.add_argument("--no-calibrate", action="store_true")
    
    # Predict command
    pred_parser = subparsers.add_parser("predict", help="Score transactions")
    pred_parser.add_argument("--model", required=True, help="Model directory")
    pred_parser.add_argument("--input", required=True, help="Input data (CSV/Parquet)")
    pred_parser.add_argument("--output", default="predictions.json", help="Output JSON")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate on test data")
    eval_parser.add_argument("--model", required=True, help="Model directory")
    eval_parser.add_argument("--data", required=True, help="Test data (CSV/Parquet)")
    
    args = parser.parse_args()
    
    if args.command == "train":
        X, y = load_training_data(args.data)
        config = ModelConfig(test_size=args.test_size, val_size=args.val_size, calibrate=not args.no_calibrate)
        detector = UpiFraudDetector(config)
        detector.train(X, y)
        detector.save(args.output)
        
    elif args.command == "predict":
        detector = UpiFraudDetector.load(args.model)
        df = pd.read_csv(args.input) if args.input.endswith(".csv") else pd.read_parquet(args.input)
        decisions = detector.predict(df)
        
        with open(args.output, "w") as f:
            json.dump(decisions, f, indent=2)
        print(f"Predictions saved to {args.output}")
        
    elif args.command == "evaluate":
        detector = UpiFraudDetector.load(args.model)
        X, y = load_training_data(args.data)
        X = X.reindex(columns=detector.feature_columns, fill_value=0)
        
        if detector.calibrated_model:
            probas = detector.calibrated_model.predict_proba(X)[:, 1]
        else:
            probas = detector.model.predict(X, num_iteration=detector.model.best_iteration)
        
        metrics = detector._compute_metrics(y.values, probas)
        detector._print_metrics()

if __name__ == "__main__":
    main()
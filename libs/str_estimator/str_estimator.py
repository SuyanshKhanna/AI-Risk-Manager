"""
Sequential Triply Robust (STR) Estimator for Causal Label Recovery.

Based on: Dhama (2026) - "Causal Label Recovery in Payment Networks" (arXiv:2605.29272)

This implements the STR estimator that corrects for:
1. Authorization gate (declined transactions generate no labels)
2. Issuer reporting gate (unreported fraud is invisible)
3. Delay gate (pending chargebacks missing at training time)
4. Corruption layer (first-party misuse, issuer misclassification)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.model_selection import cross_val_predict
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb

logger = logging.getLogger(__name__)


@dataclass
class STRConfig:
    """Configuration for STR estimator."""
    # Propensity models
    propensity_model: str = "lightgbm"  # "lightgbm", "xgboost", "logistic"
    propensity_params: Dict[str, Any] = None
    
    # Outcome regression models
    outcome_model: str = "lightgbm"
    outcome_params: Dict[str, Any] = None
    
    # Corruption correction
    corruption_model: str = "beta_binomial"  # "beta_binomial", "fixed_rate"
    noise_rate_prior: Tuple[float, float] = (1.0, 10.0)  # Beta prior for noise rate
    
    # Empirical Bayes shrinkage
    shrinkage_enabled: bool = True
    shrinkage_min_samples: int = 50
    
    # Variance estimation
    variance_estimator: str = "plugin"  # "plugin", "bootstrap"
    bootstrap_samples: int = 100
    
    # Training delay optimization
    optimize_training_delay: bool = True
    max_training_delay_days: int = 90
    
    # Cross-fitting
    n_folds: int = 5
    random_state: int = 42


class PropensityModel(ABC):
    """Abstract base class for propensity models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "PropensityModel":
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass


class LightGBMPropensity(PropensityModel):
    """LightGBM propensity model."""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42,
        }
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMPropensity":
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(self.params, train_data, num_boost_round=100)
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        return np.column_stack([1 - preds, preds])


class OutcomeModel(ABC):
    """Abstract base class for outcome regression models."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "OutcomeModel":
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        pass


class LightGBMOutcome(OutcomeModel):
    """LightGBM outcome regression model."""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 63,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42,
        }
        self.model = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LightGBMOutcome":
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(self.params, train_data, num_boost_round=200)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.model.predict(X) > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = self.model.predict(X)
        return np.column_stack([1 - preds, preds])


class STR Estimator:
    """
    Sequential Triply Robust Estimator for label recovery in payment networks.
    
    Corrects for four sequential impairments:
    1. Authorization: P(R=1|X) - probability transaction is authorized
    2. Reporting: P(O=1|R=1,X) - probability fraud is reported given authorized
    3. Delay: P(D=1|O=1,R=1,X) - probability chargeback is mature
    4. Corruption: P(Y* = Y|D=1,O=1,R=1,X) - label correctness
    
    The STR estimator combines:
    - Inverse propensity weighting (IPW)
    - Outcome regression (OR)
    - Doubly robust (DR) augmentation
    - Corruption correction via noise-rate-adjusted pseudo-labels
    """
    
    def __init__(self, config: STRConfig = None):
        self.config = config or STRConfig()
        self.propensity_models: Dict[str, PropensityModel] = {}
        self.outcome_models: Dict[str, OutcomeModel] = {}
        self.corruption_rates: Dict[str, float] = {}
        self.is_fitted = False
        
        # Default parameters
        if self.config.propensity_params is None:
            self.config.propensity_params = {}
        if self.config.outcome_params is None:
            self.config.outcome_params = {}
    
    def _get_propensity_model(self, name: str) -> PropensityModel:
        """Get or create propensity model."""
        if name not in self.propensity_models:
            if self.config.propensity_model == "lightgbm":
                self.propensity_models[name] = LightGBMPropensity(self.config.propensity_params)
            else:
                raise ValueError(f"Unknown propensity model: {self.config.propensity_model}")
        return self.propensity_models[name]
    
    def _get_outcome_model(self, name: str) -> OutcomeModel:
        """Get or create outcome model."""
        if name not in self.outcome_models:
            if self.config.outcome_model == "lightgbm":
                self.outcome_models[name] = LightGBMOutcome(self.config.outcome_params)
            else:
                raise ValueError(f"Unknown outcome model: {self.config.outcome_model}")
        return self.outcome_models[name]
    
    def fit(
        self,
        X: np.ndarray,
        Y_obs: np.ndarray,
        R: np.ndarray,  # Authorization indicator
        O: np.ndarray,  # Reporting indicator
        D: np.ndarray,  # Delay/maturity indicator
        groups: np.ndarray = None,  # Group identifiers (e.g., issuer_id)
    ) -> "STREstimator":
        """
        Fit the STR estimator.
        
        Parameters:
        - X: Features (n_samples, n_features)
        - Y_obs: Observed labels (0/1, only valid where R=O=D=1)
        - R: Authorization indicator (1 if authorized, 0 if declined)
        - O: Reporting indicator (1 if fraud reported, 0 otherwise)
        - D: Delay/maturity indicator (1 if chargeback mature, 0 pending)
        - groups: Optional group identifiers for empirical Bayes shrinkage
        """
        n_samples = len(X)
        
        # Validate inputs
        assert len(Y_obs) == n_samples
        assert len(R) == n_samples
        assert len(O) == n_samples
        assert len(D) == n_samples
        
        # Only use samples where R=O=D=1 for label-dependent modeling
        labeled_mask = (R == 1) & (O == 1) & (D == 1)
        n_labeled = labeled_mask.sum()
        
        logger.info(f"Total samples: {n_samples}, Labeled (R=O=D=1): {n_labeled}")
        
        if n_labeled == 0:
            raise ValueError("No labeled samples available (R=O=D=1)")
        
        X_labeled = X[labeled_mask]
        Y_labeled = Y_obs[labeled_mask]
        groups_labeled = groups[labeled_mask] if groups is not None else None
        
        # Stage 1: Fit propensity models for each gate
        logger.info("Fitting propensity models...")
        
        # Gate 1: Authorization propensity P(R=1|X)
        self.propensity_models["auth"] = self._get_propensity_model("auth")
        self.propensity_models["auth"].fit(X, R)
        pi_auth = self.propensity_models["auth"].predict_proba(X)[:, 1]
        pi_auth = np.clip(pi_auth, 0.01, 0.99)  # Prevent division by zero
        
        # Gate 2: Reporting propensity P(O=1|R=1,X)
        auth_mask = R == 1
        if auth_mask.sum() > 0:
            self.propensity_models["report"] = self._get_propensity_model("report")
            self.propensity_models["report"].fit(X[auth_mask], O[auth_mask])
            pi_report = np.ones(n_samples)
            pi_report[auth_mask] = self.propensity_models["report"].predict_proba(X[auth_mask])[:, 1]
            pi_report = np.clip(pi_report, 0.01, 0.99)
        else:
            pi_report = np.ones(n_samples)
        
        # Gate 3: Delay propensity P(D=1|O=1,R=1,X)
        report_mask = (R == 1) & (O == 1)
        if report_mask.sum() > 0:
            self.propensity_models["delay"] = self._get_propensity_model("delay")
            self.propensity_models["delay"].fit(X[report_mask], D[report_mask])
            pi_delay = np.ones(n_samples)
            pi_delay[report_mask] = self.propensity_models["delay"].predict_proba(X[report_mask])[:, 1]
            pi_delay = np.clip(pi_delay, 0.01, 0.99)
        else:
            pi_delay = np.ones(n_samples)
        
        # Combined propensity
        pi_combined = pi_auth * pi_report * pi_delay
        
        # Stage 2: Fit outcome regression on labeled data
        logger.info("Fitting outcome regression...")
        self.outcome_models["main"] = self._get_outcome_model("main")
        self.outcome_models["main"].fit(X_labeled, Y_labeled)
        
        # Get outcome predictions for all samples
        mu_hat = self.outcome_models["main"].predict_proba(X)[:, 1]
        
        # Stage 3: Corruption correction
        logger.info("Estimating corruption rates...")
        self._estimate_corruption_rates(
            X_labeled, Y_labeled, groups_labeled, pi_combined[labeled_mask]
        )
        
        # Stage 4: Compute STR pseudo-labels
        logger.info("Computing STR pseudo-labels...")
        pseudo_labels = self._compute_pseudo_labels(
            X, Y_obs, R, O, D, mu_hat, pi_combined, groups
        )
        
        # Stage 5: Empirical Bayes shrinkage for group-level weights
        if self.config.shrinkage_enabled and groups is not None:
            logger.info("Applying empirical Bayes shrinkage...")
            pseudo_labels = self._apply_shrinkage(pseudo_labels, groups, pi_combined)
        
        # Store for later use
        self.pi_auth = pi_auth
        self.pi_report = pi_report
        self.pi_delay = pi_delay
        self.pi_combined = pi_combined
        self.mu_hat = mu_hat
        self.pseudo_labels = pseudo_labels
        self.is_fitted = True
        
        logger.info("STR estimator fitted successfully")
        return self
    
    def _estimate_corruption_rates(
        self,
        X_labeled: np.ndarray,
        Y_labeled: np.ndarray,
        groups_labeled: np.ndarray,
        pi_labeled: np.ndarray,
    ):
        """Estimate label corruption rates."""
        if self.config.corruption_model == "beta_binomial":
            # Use empirical Bayes with Beta prior
            alpha, beta = self.config.noise_rate_prior
            
            if groups_labeled is not None:
                # Group-wise corruption estimation
                unique_groups = np.unique(groups_labeled)
                for group in unique_groups:
                    mask = groups_labeled == group
                    n_group = mask.sum()
                    
                    if n_group >= self.config.shrinkage_min_samples:
                        # Observed positive rate
                        pos_rate = Y_labeled[mask].mean()
                        # Posterior mean of noise rate
                        self.corruption_rates[group] = (alpha + pos_rate * n_group) / (alpha + beta + n_group)
                    else:
                        # Shrink toward global
                        global_pos_rate = Y_labeled.mean()
                        n_global = len(Y_labeled)
                        self.corruption_rates[group] = (alpha + global_pos_rate * n_global) / (alpha + beta + n_global)
            else:
                # Global corruption rate
                n_global = len(Y_labeled)
                global_pos_rate = Y_labeled.mean()
                self.corruption_rates["global"] = (alpha + global_pos_rate * n_global) / (alpha + beta + n_global)
        
        elif self.config.corruption_model == "fixed_rate":
            # Use fixed noise rate (e.g., from domain knowledge)
            self.corruption_rates["global"] = 0.1  # 10% assumed noise
        
        logger.info(f"Estimated corruption rates: {self.corruption_rates}")
    
    def _compute_pseudo_labels(
        self,
        X: np.ndarray,
        Y_obs: np.ndarray,
        R: np.ndarray,
        O: np.ndarray,
        D: np.ndarray,
        mu_hat: np.ndarray,
        pi_combined: np.ndarray,
        groups: np.ndarray,
    ) -> np.ndarray:
        """Compute STR pseudo-labels with corruption correction."""
        n = len(X)
        pseudo = np.zeros(n)
        
        # For labeled samples (R=O=D=1)
        labeled_mask = (R == 1) & (O == 1) & (D == 1)
        
        if labeled_mask.sum() > 0:
            # Corruption correction for labeled samples
            if groups is not None:
                noise_rates = np.array([
                    self.corruption_rates.get(g, self.corruption_rates.get("global", 0.1))
                    for g in groups[labeled_mask]
                ])
            else:
                noise_rates = np.full(labeled_mask.sum(), self.corruption_rates.get("global", 0.1))
            
            # Noise-rate-adjusted pseudo-labels
            # Y* = (Y_obs - noise_rate) / (1 - 2*noise_rate) for symmetric noise
            # For asymmetric: more complex adjustment
            Y_labeled = Y_obs[labeled_mask]
            pseudo[labeled_mask] = (Y_labeled - noise_rates) / (1 - 2 * noise_rates)
            pseudo[labeled_mask] = np.clip(pseudo[labeled_mask], 0, 1)
        
        # For unlabeled samples, use outcome regression prediction
        pseudo[~labeled_mask] = mu_hat[~labeled_mask]
        
        return pseudo
    
    def _apply_shrinkage(
        self,
        pseudo_labels: np.ndarray,
        groups: np.ndarray,
        pi_combined: np.ndarray,
    ) -> np.ndarray:
        """Apply empirical Bayes shrinkage to inverse propensity weights."""
        unique_groups = np.unique(groups)
        shrunk_labels = pseudo_labels.copy()
        
        for group in unique_groups:
            mask = groups == group
            n_group = mask.sum()
            
            if n_group >= self.config.shrinkage_min_samples:
                # Compute group-level average pseudo-label
                group_mean = pseudo_labels[mask].mean()
                global_mean = pseudo_labels.mean()
                
                # Shrinkage factor (James-Stein type)
                # More shrinkage for smaller groups
                shrinkage_factor = n_group / (n_group + self.config.shrinkage_min_samples)
                
                # Shrink toward global mean
                shrunk_labels[mask] = (
                    shrinkage_factor * pseudo_labels[mask] +
                    (1 - shrinkage_factor) * global_mean
                )
        
        return shrunk_labels
    
    def predict_pseudo_labels(self, X: np.ndarray) -> np.ndarray:
        """Get pseudo-labels for new data (uses outcome regression)."""
        if not self.is_fitted:
            raise ValueError("Estimator not fitted")
        
        return self.outcome_models["main"].predict_proba(X)[:, 1]
    
    def get_training_weights(self) -> np.ndarray:
        """Get inverse propensity weights for training."""
        if not self.is_fitted:
            raise ValueError("Estimator not fitted")
        
        # Stabilized weights
        weights = 1.0 / self.pi_combined
        weights = weights / weights.mean()  # Normalize
        return weights
    
    def estimate_variance(self, X: np.ndarray, pseudo_labels: np.ndarray) -> np.ndarray:
        """Estimate variance of pseudo-labels using plugin estimator."""
        if not self.is_fitted:
            raise ValueError("Estimator not fitted")
        
        if self.config.variance_estimator == "plugin":
            # Plugin variance estimator
            # Var = E[IF^2] where IF is influence function
            mu = self.mu_hat
            pi = self.pi_combined
            R = np.ones(len(X))  # Assume all authorized for new data
            
            # Simplified variance for labeled data
            labeled_var = pseudo_labels * (1 - pseudo_labels) / pi
            unlabeled_var = mu * (1 - mu)
            
            variance = np.where(
                np.isclose(pseudo_labels, mu),
                unlabeled_var,
                labeled_var
            )
            return variance
        
        elif self.config.variance_estimator == "bootstrap":
            # Bootstrap variance (computationally expensive)
            n = len(X)
            variances = []
            for _ in range(self.config.bootstrap_samples):
                idx = np.random.choice(n, n, replace=True)
                # Recompute pseudo-labels on bootstrap sample (simplified)
                boot_pseudo = pseudo_labels[idx]
                variances.append(boot_pseudo.var())
            return np.array(variances).mean(axis=0)
        
        return np.ones(len(X)) * 0.25  # Default conservative variance
    
    def optimal_training_delay(
        self,
        X: np.ndarray,
        Y_obs: np.ndarray,
        R: np.ndarray,
        O: np.ndarray,
        D: np.ndarray,
        delay_days: np.ndarray,  # Days since transaction
    ) -> int:
        """
        Estimate optimal training delay that minimizes MSE.
        
        Balances label quality (improves with delay) vs model staleness (worsens with delay).
        """
        if not self.config.optimize_training_delay:
            return 30  # Default 30 days
        
        max_delay = self.config.max_training_delay_days
        best_delay = 30
        best_mse = float('inf')
        
        for d in range(7, max_delay + 1, 7):  # Weekly grid
            # Maturity indicator for this delay
            D_d = (delay_days >= d).astype(int)
            
            # Fit STR with this delay
            try:
                str_d = STREstimator(self.config)
                str_d.fit(X, Y_obs, R, O, D_d)
                
                # Estimate MSE on validation (using pseudo-labels as proxy)
                pseudo = str_d.pseudo_labels
                weights = str_d.get_training_weights()
                
                # Weighted MSE
                mse = np.average((pseudo - str_d.mu_hat) ** 2, weights=weights)
                
                if mse < best_mse:
                    best_mse = mse
                    best_delay = d
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate delay {d}: {e}")
                continue
        
        logger.info(f"Optimal training delay: {best_delay} days (MSE: {best_mse:.4f})")
        return best_delay


def create_str_estimator(config: STRConfig = None) -> STREstimator:
    """Factory function to create STR estimator."""
    return STREstimator(config)


class STRClassifier(BaseEstimator, ClassifierMixin):
    """
    Scikit-learn compatible classifier using STR pseudo-labels.
    """
    
    def __init__(
        self,
        base_estimator=None,
        str_config: STRConfig = None,
        calibration: str = "isotonic",
    ):
        self.base_estimator = base_estimator or lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=63,
            random_state=42,
            verbose=-1,
        )
        self.str_config = str_config or STRConfig()
        self.calibration = calibration
        self.str_estimator = None
        self.calibrated_classifier = None
    
    def fit(self, X, y, R=None, O=None, D=None, groups=None):
        """Fit STR estimator then train classifier on pseudo-labels."""
        # Default gates if not provided
        if R is None:
            R = np.ones(len(X))
        if O is None:
            O = np.ones(len(X))
        if D is None:
            D = np.ones(len(X))
        if groups is None:
            groups = np.zeros(len(X), dtype=int)
        
        # Fit STR estimator
        self.str_estimator = STREstimator(self.str_config)
        self.str_estimator.fit(X, y, R, O, D, groups)
        
        # Get pseudo-labels and weights
        pseudo_labels = self.str_estimator.pseudo_labels
        weights = self.str_estimator.get_training_weights()
        
        # Train base estimator on pseudo-labels with IPW weights
        self.base_estimator.fit(X, pseudo_labels, sample_weight=weights)
        
        # Calibrate
        if self.calibration:
            self.calibrated_classifier = CalibratedClassifierCV(
                self.base_estimator,
                method=self.calibration,
                cv=3,
            )
            self.calibrated_classifier.fit(X, pseudo_labels, sample_weight=weights)
        
        self.classes_ = np.array([0, 1])
        return self
    
    def predict(self, X):
        """Predict class labels."""
        if self.calibrated_classifier:
            return self.calibrated_classifier.predict(X)
        return self.base_estimator.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if self.calibrated_classifier:
            return self.calibrated_classifier.predict_proba(X)
        return self.base_estimator.predict_proba(X)
    
    def get_str_diagnostics(self) -> Dict[str, Any]:
        """Get STR diagnostic information."""
        if self.str_estimator is None:
            return {}
        
        return {
            "corruption_rates": self.str_estimator.corruption_rates,
            "propensity_means": {
                "auth": self.str_estimator.pi_auth.mean(),
                "report": self.str_estimator.pi_report.mean(),
                "delay": self.str_estimator.pi_delay.mean(),
            },
            "pseudo_label_stats": {
                "mean": self.str_estimator.pseudo_labels.mean(),
                "std": self.str_estimator.pseudo_labels.std(),
                "min": self.str_estimator.pseudo_labels.min(),
                "max": self.str_estimator.pseudo_labels.max(),
            },
        }
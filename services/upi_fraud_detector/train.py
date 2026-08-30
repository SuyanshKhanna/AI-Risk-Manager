"""Training script for UPI Fraud Detector."""

import asyncio
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
import structlog

from service import UpiFraudDetector

logger = structlog.get_logger(__name__)


def generate_training_data(n_samples: int = 10000) -> tuple:
    """Generate synthetic training data for UPI fraud detection."""
    np.random.seed(42)
    
    # Generate features
    n_features = 100
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Generate labels with some fraud patterns
    # Fraud more likely with certain feature combinations
    fraud_signal = (
        2.0 * (X[:, 0] > 1.5) +  # High velocity
        1.5 * (X[:, 1] < -1.0) +  # Settlement issues
        1.0 * (X[:, 2] > 2.0) +   # Device anomaly
        0.5 * np.random.randn(n_samples)
    )
    
    # Convert to probabilities
    fraud_prob = 1 / (1 + np.exp(-fraud_signal))
    fraud_prob = np.clip(fraud_prob, 0.001, 0.5)  # Cap at 50%
    
    # Sample labels
    y = np.random.binomial(1, fraud_prob)
    
    # Gates
    R = np.ones(n_samples, dtype=int)  # All authorized
    O = np.where(y == 1, np.random.binomial(1, 0.7, n_samples), 0)  # 70% reporting rate
    D = np.where(O == 1, np.random.binomial(1, 0.8, n_samples), 0)  # 80% maturity
    
    # Groups (issuer IDs)
    groups = np.random.randint(1, 20, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(n_features)])
    df["label"] = y
    df["authorized"] = R
    df["reported"] = O
    df["mature"] = D
    df["issuer_id"] = groups
    
    # Split
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size+val_size].copy()
    test_df = df.iloc[train_size+val_size:].copy()
    
    return train_df, val_df, test_df


async def main():
    """Main training function."""
    # Load config
    config = OmegaConf.load("configs/upi_local.yaml")
    
    # Generate training data
    logger.info("Generating training data...")
    train_df, val_df, test_df = generate_training_data(50000)
    
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info(f"Train fraud rate: {train_df['label'].mean():.4f}")
    logger.info(f"Val fraud rate: {val_df['label'].mean():.4f}")
    
    # Initialize detector
    detector = UpiFraudDetector(config)
    await detector.initialize()
    
    # Train
    logger.info("Starting training...")
    await detector.train(train_df, val_df)
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
    
    # Get predictions (simplified)
    test_X = detector._prepare_features(test_df)
    test_scores = []
    for i in range(len(test_X)):
        features = {"sequence": np.zeros((30, 100))}  # Placeholder
        score = await detector._predict(features)
        test_scores.append(score)
    
    test_scores = np.array(test_scores)
    test_y = test_df["label"].values
    
    auc = roc_auc_score(test_y, test_scores)
    pr_auc = average_precision_score(test_y, test_scores)
    
    logger.info(f"Test ROC-AUC: {auc:.4f}")
    logger.info(f"Test PR-AUC: {pr_auc:.4f}")
    
    # Print precision-recall at different thresholds
    precision, recall, thresholds = precision_recall_curve(test_y, test_scores)
    for target_recall in [0.5, 0.7, 0.8, 0.9]:
        idx = np.argmin(np.abs(recall - target_recall))
        if idx < len(thresholds):
            logger.info(f"Recall={recall[idx]:.3f}, Precision={precision[idx]:.3f}, Threshold={thresholds[idx]:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
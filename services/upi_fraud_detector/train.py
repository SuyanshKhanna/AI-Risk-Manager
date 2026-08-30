#!/usr/bin/env python3
"""
Train UPI Fraud Detector (LightGBM) on synthetic data.

Usage:
    python -m upi_fraud_detector.train --data data/upi_fraud/upi_fraud_training.parquet --output models/upi_fraud_lgbm
"""

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pathlib import Path
import pandas as pd
from detector import UpiFraudDetector, ModelConfig, load_training_data


def main():
    # Paths
    data_path = Path("data/upi_fraud/upi_fraud_training.parquet")
    model_path = Path("models/upi_fraud_lgbm")
    
    if not data_path.exists():
        print(f"Training data not found at {data_path}")
        print("Run: python scripts/data_gen/generate_upi_fraud_data.py --add-rolling --format parquet")
        return
    
    print(f"Loading training data from {data_path}")
    X, y = load_training_data(str(data_path))
    print(f"Loaded {len(X):,} samples, {X.shape[1]} features, fraud rate: {y.mean()*100:.3f}%")
    
    # Configure
    config = ModelConfig(
        test_size=0.2,
        val_size=0.1,
        calibrate=True,
        challenge_threshold=0.3,
        block_threshold=0.7,
        fp_cost_per_blocked_txn=150.0,
    )
    
    # Train
    detector = UpiFraudDetector(config)
    metrics = detector.train(X, y)
    
    # Save
    detector.save(str(model_path))
    print(f"\nModel saved to {model_path}")
    print(f"Model version: {detector.model_version}")


if __name__ == "__main__":
    main()
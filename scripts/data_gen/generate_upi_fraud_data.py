#!/usr/bin/env python3
"""
Synthetic UPI Fraud Data Generator for Hackathon MVP.

Generates realistic UPI transaction data with ground-truth fraud labels.
Covers 4 key UPI fraud patterns:
1. Fake payment screenshots (velocity + settlement mismatch)
2. QR code tampering (device anomaly + new merchant)
3. Refund fraud loops (high refund velocity + same UPI handle)
4. Screen-sharing scams (remote access + unusual timing)

Output: CSV/Parquet with engineered features + ground-truth label column.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple
import hashlib
import json

# Reproducibility
np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────

FRAUD_PATTERNS = {
    "fake_screenshot": {
        "description": "Buyer sends fake success screenshot; no real settlement",
        "weight": 0.35,
        "feature_signature": {
            "settlement_verified": 0,        # always 0 for this pattern
            "velocity_5min": (5, 20),        # high burst
            "amount_zscore": (2, 5),         # unusually large
            "new_upi_handle": 1,             # throwaway handle
            "device_emulator_score": (0.7, 1.0),
            "hour": (22, 6),                 # late night
        }
    },
    "qr_tampering": {
        "description": "QR code swapped/overlaid; funds go to attacker",
        "weight": 0.25,
        "feature_signature": {
            "settlement_verified": 1,        # settlement happens but to wrong VPA
            "qr_mismatch": 1,                # QR hash != registered
            "new_merchant": 1,               # first time at this merchant
            "device_root_score": (0.5, 0.9),
            "vpn_probability": (0.6, 1.0),
            "amount_zscore": (0.5, 2),
        }
    },
    "refund_loop": {
        "description": "Repeated refund requests on same UPI handle/device",
        "weight": 0.25,
        "feature_signature": {
            "settlement_verified": 1,
            "refund_count_1hr": (3, 10),
            "refund_rate": (0.5, 1.0),
            "same_device_refunds": (3, 8),
            "velocity_1hr": (10, 30),
            "device_fraud_reports_30d": (1, 5),
        }
    },
    "screen_sharing": {
        "description": "Victim tricked into screen share; attacker initiates txn",
        "weight": 0.15,
        "feature_signature": {
            "settlement_verified": 1,
            "remote_access_app": 1,          # TeamViewer/AnyDesk detected
            "session_duration_sec": (300, 1800),  # long session
            "unusual_hour": 1,               # outside normal hours
            "new_device": 1,
            "amount_zscore": (1, 3),
        }
    }
}

LEGITIMATE_BASE = {
    "settlement_verified": 1,
    "qr_mismatch": 0,
    "remote_access_app": 0,
    "new_upi_handle": 0,
    "new_merchant": 0,
    "new_device": 0,
    "unusual_hour": 0,
}

# ──────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

def sample_range(rng: Tuple[float, float]) -> float:
    """Uniform sample from range."""
    return np.random.uniform(rng[0], rng[1])

def sample_int_range(rng: Tuple[int, int]) -> int:
    """Uniform integer sample from range."""
    return np.random.randint(rng[0], rng[1] + 1)

def hash_id(prefix: str, idx: int) -> str:
    """Deterministic short hash ID."""
    return f"{prefix}_{hashlib.md5(f'{prefix}{idx}'.encode()).hexdigest()[:8]}"

def pick_hour(pattern_hour_range: Tuple[int, int]) -> int:
    """Pick hour, handling wrap-around (e.g., 22-6)."""
    start, end = pattern_hour_range
    if start <= end:
        return np.random.randint(start, end + 1)
    # wrap around midnight
    return np.random.choice(list(range(start, 24)) + list(range(0, end + 1)))

# ──────────────────────────────────────────────────────────────────────
# CORE GENERATOR
# ──────────────────────────────────────────────────────────────────────

def generate_legitimate_transaction(txn_id: int, base_time: datetime) -> Dict:
    """Generate a legitimate transaction with realistic feature values."""
    # Base time with some jitter
    txn_time = base_time + timedelta(minutes=np.random.exponential(10))
    
    # Legitimate users have consistent patterns
    merchant_id = hash_id("M", np.random.randint(1, 200))
    customer_id = hash_id("C", np.random.randint(1, 5000))
    device_fp = hash_id("dev", np.random.randint(1, 3000))
    upi_handle = hash_id("upi", np.random.randint(1, 4000))
    
    # Amount follows log-normal (typical UPI distribution)
    amount = int(np.random.lognormal(8.5, 0.7))
    
    # Rolling stats (simulated)
    merchant_avg = np.random.lognormal(8.5, 0.5)
    merchant_std = merchant_avg * 0.4
    amount_zscore = max(-3, min(3, (amount - merchant_avg) / max(merchant_std, 1)))
    
    # Velocity features
    velocity_5min = np.random.poisson(0.5)
    velocity_1hr = np.random.poisson(3)
    refund_count_1hr = np.random.poisson(0.02)
    refund_rate = 0.0 if velocity_1hr == 0 else refund_count_1hr / velocity_1hr
    
    # Device features
    device_emulator_score = np.random.beta(1, 50)
    device_root_score = np.random.beta(1, 30)
    device_fraud_reports_30d = np.random.poisson(0.01)
    vpn_probability = np.random.beta(1, 20)
    
    # Time features
    hour = txn_time.hour
    is_night = 1 if hour >= 22 or hour <= 6 else 0
    
    return {
        "txn_id": f"TXN{txn_id:08d}",
        "timestamp": txn_time.isoformat(),
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "device_fingerprint": device_fp,
        "upi_handle": upi_handle,
        "amount_paise": amount,
        "amount_zscore": round(amount_zscore, 3),
        "merchant_avg_amount": round(merchant_avg, 2),
        "velocity_5min": velocity_5min,
        "velocity_1hr": velocity_1hr,
        "refund_count_1hr": refund_count_1hr,
        "refund_rate": round(refund_rate, 3),
        "same_device_refunds": 0,
        "settlement_verified": 1,
        "qr_mismatch": 0,
        "remote_access_app": 0,
        "device_emulator_score": round(device_emulator_score, 4),
        "device_root_score": round(device_root_score, 4),
        "device_fraud_reports_30d": device_fraud_reports_30d,
        "vpn_probability": round(vpn_probability, 4),
        "hour": hour,
        "is_night": is_night,
        "new_upi_handle": 0,
        "new_merchant": 0,
        "new_device": 0,
        "unusual_hour": 0,
        "session_duration_sec": 0,
        "is_fraud": 0,
        "fraud_type": "legitimate",
    }

def generate_fraud_transaction(txn_id: int, base_time: datetime, pattern_name: str) -> Dict:
    """Generate a fraudulent transaction following a specific pattern."""
    pattern = FRAUD_PATTERNS[pattern_name]
    sig = pattern["feature_signature"]
    
    txn_time = base_time + timedelta(minutes=np.random.exponential(5))
    
    # Fraud often uses newer/throwaway identities
    merchant_id = hash_id("M", np.random.randint(1, 200))
    customer_id = hash_id("C", np.random.randint(1, 5000))
    device_fp = hash_id("dev", np.random.randint(1, 3000))
    upi_handle = hash_id("upi", np.random.randint(1, 4000))
    
    # Amount - often larger for fraud
    base_amount = int(np.random.lognormal(8.5, 0.7))
    if "amount_zscore" in sig:
        z = sample_range(sig["amount_zscore"])
        amount = int(base_amount * (1 + z * 0.5))
    else:
        amount = base_amount
    
    merchant_avg = np.random.lognormal(8.5, 0.5)
    amount_zscore = max(-3, min(3, (amount - merchant_avg) / max(merchant_avg * 0.4, 1)))
    
    # Build feature dict starting from legitimate base
    features = {
        "txn_id": f"TXN{txn_id:08d}",
        "timestamp": txn_time.isoformat(),
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "device_fingerprint": device_fp,
        "upi_handle": upi_handle,
        "amount_paise": amount,
        "amount_zscore": round(amount_zscore, 3),
        "merchant_avg_amount": round(merchant_avg, 2),
        "velocity_5min": 0,
        "velocity_1hr": 0,
        "refund_count_1hr": 0,
        "refund_rate": 0.0,
        "same_device_refunds": 0,
        "settlement_verified": 1,
        "qr_mismatch": 0,
        "remote_access_app": 0,
        "device_emulator_score": 0.0,
        "device_root_score": 0.0,
        "device_fraud_reports_30d": 0,
        "vpn_probability": 0.0,
        "hour": txn_time.hour,
        "is_night": 0,
        "new_upi_handle": 0,
        "new_merchant": 0,
        "new_device": 0,
        "unusual_hour": 0,
        "session_duration_sec": 0,
        "is_fraud": 1,
        "fraud_type": pattern_name,
    }
    
    # Override with pattern-specific values
    for key, val in sig.items():
        if isinstance(val, tuple):
            if key in ["hour"]:
                features[key] = pick_hour(val)
                features["is_night"] = 1 if features[key] >= 22 or features[key] <= 6 else 0
                features["unusual_hour"] = features["is_night"]
            else:
                features[key] = round(sample_range(val), 4)
        else:
            features[key] = val
    
    # Derived features
    if "velocity_5min" in sig:
        features["velocity_5min"] = sample_int_range(sig["velocity_5min"])
    if "velocity_1hr" in sig:
        features["velocity_1hr"] = sample_int_range(sig["velocity_1hr"])
    if "refund_count_1hr" in sig:
        features["refund_count_1hr"] = sample_int_range(sig["refund_count_1hr"])
        features["refund_rate"] = round(features["refund_count_1hr"] / max(features["velocity_1hr"], 1), 3)
    if "same_device_refunds" in sig:
        features["same_device_refunds"] = sample_int_range(sig["same_device_refunds"])
    if "session_duration_sec" in sig:
        features["session_duration_sec"] = sample_int_range(sig["session_duration_sec"])
    
    return features

def generate_dataset(n_legitimate: int = 50000, n_fraud: int = 500, fraud_ratio: float = None) -> pd.DataFrame:
    """
    Generate mixed dataset with realistic fraud ratio.
    
    Args:
        n_legitimate: Number of legitimate transactions
        n_fraud: Number of fraud transactions (if fraud_ratio is None)
        fraud_ratio: If set, overrides n_fraud to achieve this ratio
    """
    if fraud_ratio is not None:
        n_fraud = int(n_legitimate * fraud_ratio / (1 - fraud_ratio))
    
    print(f"Generating {n_legitimate:,} legitimate + {n_fraud:,} fraud transactions")
    print(f"Fraud ratio: {n_fraud/(n_legitimate+n_fraud)*100:.3f}%")
    
    base_time = datetime.now(timezone.utc) - timedelta(days=90)
    records = []
    
    # Generate legitimate transactions
    for i in range(n_legitimate):
        records.append(generate_legitimate_transaction(i, base_time))
    
    # Generate fraud transactions distributed across patterns
    fraud_weights = {k: v["weight"] for k, v in FRAUD_PATTERNS.items()}
    pattern_names = list(fraud_weights.keys())
    pattern_probs = list(fraud_weights.values())
    
    for i in range(n_fraud):
        pattern = np.random.choice(pattern_names, p=pattern_probs)
        records.append(generate_fraud_transaction(n_legitimate + i, base_time, pattern))
    
    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    return df

# ──────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (for model training)
# ──────────────────────────────────────────────────────────────────────

def add_rolling_features(df: pd.DataFrame, windows: List[int] = [5, 60, 1440]) -> pd.DataFrame:
    """Add rolling window features per merchant/device/UPI handle.
    
    Note: For large datasets, this is a placeholder. In production, use a more efficient
    streaming approach with a proper feature store (Feast).
    """
    # Placeholder - in production, use Feast or a streaming feature store
    # For hackathon MVP, we skip expensive rolling computations
    for entity_col in ["merchant_id", "device_fingerprint", "upi_handle"]:
        for window_min in windows:
            window_str = f"{window_min}min"
            df[f"{entity_col}_txn_count_{window_str}"] = 0
            df[f"{entity_col}_amount_sum_{window_min}min"] = 0
    return df

def add_deviation_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deviation-from-baseline features."""
    df = df.copy()
    
    # Merchant-level rolling stats
    merchant_stats = df.groupby("merchant_id")["amount_paise"].agg(["mean", "std"]).reset_index()
    merchant_stats.columns = ["merchant_id", "merchant_mean_amt", "merchant_std_amt"]
    merchant_stats["merchant_std_amt"] = merchant_stats["merchant_std_amt"].replace(0, 1)
    df = df.merge(merchant_stats, on="merchant_id", how="left")
    
    # Z-score vs merchant
    df["amount_zscore_merchant"] = (df["amount_paise"] - df["merchant_mean_amt"]) / df["merchant_std_amt"]
    df["amount_zscore_merchant"] = df["amount_zscore_merchant"].clip(-5, 5)
    
    # Customer-level stats
    cust_stats = df.groupby("customer_id")["amount_paise"].agg(["mean", "std"]).reset_index()
    cust_stats.columns = ["customer_id", "cust_mean_amt", "cust_std_amt"]
    cust_stats["cust_std_amt"] = cust_stats["cust_std_amt"].replace(0, 1)
    df = df.merge(cust_stats, on="customer_id", how="left")
    df["amount_zscore_cust"] = (df["amount_paise"] - df["cust_mean_amt"]) / df["cust_std_amt"]
    df["amount_zscore_cust"] = df["amount_zscore_cust"].clip(-5, 5)
    
    return df

def prepare_training_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Select and prepare final feature columns for training."""
    # Core engineered features
    feature_cols = [
        "amount_paise",
        "amount_zscore",
        "amount_zscore_merchant",
        "amount_zscore_cust",
        "velocity_5min",
        "velocity_1hr",
        "refund_count_1hr",
        "refund_rate",
        "same_device_refunds",
        "settlement_verified",
        "qr_mismatch",
        "remote_access_app",
        "device_emulator_score",
        "device_root_score",
        "device_fraud_reports_30d",
        "vpn_probability",
        "hour",
        "is_night",
        "new_upi_handle",
        "new_merchant",
        "new_device",
        "unusual_hour",
        "session_duration_sec",
        # Rolling counts
        "merchant_id_txn_count_5min",
        "merchant_id_txn_count_1hr",
        "device_fingerprint_txn_count_5min",
        "device_fingerprint_txn_count_1hr",
        "upi_handle_txn_count_5min",
        "upi_handle_txn_count_1hr",
    ]
    
    # Filter to existing columns
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols].fillna(0)
    y = df["is_fraud"].astype(int)
    
    return X, y, feature_cols

# ──────────────────────────────────────────────────────────────────────
# MAIN / CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic UPI fraud dataset")
    parser.add_argument("--n-legitimate", type=int, default=50000, help="Number of legitimate transactions")
    parser.add_argument("--n-fraud", type=int, default=500, help="Number of fraud transactions")
    parser.add_argument("--fraud-ratio", type=float, default=None, help="Target fraud ratio (overrides n-fraud)")
    parser.add_argument("--output-dir", type=str, default="data/upi_fraud", help="Output directory")
    parser.add_argument("--format", choices=["csv", "parquet", "both"], default="both")
    parser.add_argument("--add-rolling", action="store_true", help="Add rolling window features")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate base dataset
    df = generate_dataset(
        n_legitimate=args.n_legitimate,
        n_fraud=args.n_fraud,
        fraud_ratio=args.fraud_ratio
    )
    
    # Add rolling features if requested
    if args.add_rolling:
        print("Adding rolling window features...")
        df = add_rolling_features(df)
        df = add_deviation_features(df)
    
    # Save raw dataset
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"upi_fraud_{timestamp}"
    
    if args.format in ["csv", "both"]:
        csv_path = output_path / f"{base_name}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
    
    if args.format in ["parquet", "both"]:
        pq_path = output_path / f"{base_name}.parquet"
        df.to_parquet(pq_path, index=False)
        print(f"Saved Parquet: {pq_path}")
    
    # Save feature matrix for training
    X, y, feature_cols = prepare_training_features(df)
    train_df = pd.concat([X, y.rename("label")], axis=1)
    
    if args.format in ["csv", "both"]:
        train_csv = output_path / f"{base_name}_training.csv"
        train_df.to_csv(train_csv, index=False)
        print(f"Saved training CSV: {train_csv}")
    
    if args.format in ["parquet", "both"]:
        train_pq = output_path / f"{base_name}_training.parquet"
        train_df.to_parquet(train_pq, index=False)
        print(f"Saved training Parquet: {train_pq}")
    
    # Save feature column list
    with open(output_path / f"{base_name}_features.json", "w") as f:
        json.dump({"features": feature_cols, "target": "label"}, f, indent=2)
    
    # Print summary stats
    print("\n=== Dataset Summary ===")
    print(f"Total transactions: {len(df):,}")
    print(f"Fraud rate: {df['is_fraud'].mean()*100:.3f}%")
    print(f"Fraud types:\n{df[df['is_fraud']==1]['fraud_type'].value_counts()}")
    print(f"\nFeature columns ({len(feature_cols)}): {feature_cols}")
    
    # Class balance check
    print(f"\nClass balance - Legitimate: {(df['is_fraud']==0).sum():,}, Fraud: {(df['is_fraud']==1).sum():,}")

if __name__ == "__main__":
    main()
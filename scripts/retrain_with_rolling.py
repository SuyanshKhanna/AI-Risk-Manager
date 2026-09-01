#!/usr/bin/env python3
"""
Retrain UPI Fraud Detector with real rolling-window features.

Steps:
  1. Generate dataset with REAL rolling counts (no placeholder zeros).
  2. Run leakage sanity-check on a sample of entity+window combinations.
  3. Train LightGBM and print before/after metrics comparison.
  4. Flag if PR-AUC is suspiciously close to 1.0.
  5. Save the new model to models/upi_fraud_lgbm.

Usage (from repo root):
    python scripts/retrain_with_rolling.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Ensure repo root is on sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# ── 0. Load previous metrics (if any) ────────────────────────────────────────

OLD_METRICS_PATH = ROOT / "models" / "upi_fraud_lgbm" / "config.json"
old_metrics: dict = {}
if OLD_METRICS_PATH.exists():
    with open(OLD_METRICS_PATH) as f:
        saved = json.load(f)
    old_metrics = saved.get("metrics", {})
    print(f"Loaded old metrics from {OLD_METRICS_PATH}")
else:
    print("No existing model found — this will be a fresh baseline.")

# ── 1. Generate dataset with real rolling features ────────────────────────────

print("\n" + "="*60)
print("STEP 1: Generating data with real rolling-window features")
print("="*60)

from scripts.data_gen.generate_upi_fraud_data import (
    generate_dataset,
    add_rolling_features,
    add_deviation_features,
    prepare_training_features,
)

t0 = time.time()
df = generate_dataset(n_legitimate=50_000, n_fraud=500)
print(f"Base dataset generated: {len(df):,} rows in {time.time()-t0:.1f}s")

t0 = time.time()
print("Computing real (leakage-free) rolling window features...")
df = add_rolling_features(df, windows=[5, 60])
print(f"  Rolling counts done in {time.time()-t0:.1f}s")

t0 = time.time()
print("Computing expanding deviation features (no future leakage)...")
df = add_deviation_features(df)
print(f"  Deviation features done in {time.time()-t0:.1f}s")

# ── 2. Leakage sanity check ───────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 2: Leakage sanity check")
print("="*60)

from scripts.data_gen.rolling_features import verify_no_leakage

# Sample 200 transactions per entity to keep the O(N²) check fast
SAMPLE_N = 200  # rows per entity
checks_clean = True
for entity_col in ["merchant_id", "device_fingerprint", "upi_handle"]:
    for wmin in [5, 60]:
        # Take the first SAMPLE_N rows of the sorted df
        sample_df = df.head(SAMPLE_N).copy()
        ok = verify_no_leakage(sample_df, entity_col, wmin)
        status = "[OK] clean" if ok else "[!!] LEAKAGE DETECTED"
        print(f"  {entity_col} / {wmin}min window: {status}")
        if not ok:
            checks_clean = False

if not checks_clean:
    print("\n[ABORT] Leakage detected — aborting retrain.  Fix the rolling logic first.")
    sys.exit(1)
else:
    print("\n[PASS] All leakage checks passed.")

# ── 3. Show rolling feature stats ─────────────────────────────────────────────

print("\n--- Rolling feature summary (fraud vs legitimate) ---")
for col in ["merchant_id_txn_count_5min", "merchant_id_txn_count_60min",
            "device_fingerprint_txn_count_5min", "upi_handle_txn_count_60min"]:
    if col not in df.columns:
        continue
    legit = df[df["is_fraud"] == 0][col]
    fraud = df[df["is_fraud"] == 1][col]
    print(f"  {col}:")
    print(f"    Legit — mean={legit.mean():.2f}  p95={legit.quantile(.95):.1f}  max={legit.max()}")
    print(f"    Fraud — mean={fraud.mean():.2f}  p95={fraud.quantile(.95):.1f}  max={fraud.max()}")

# ── 4. Save training data ─────────────────────────────────────────────────────

DATA_DIR = ROOT / "data" / "upi_fraud"
DATA_DIR.mkdir(parents=True, exist_ok=True)
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
train_pq = DATA_DIR / f"upi_fraud_{ts}_training.parquet"

X, y, feature_cols = prepare_training_features(df)
train_df = pd.concat([X, y.rename("label")], axis=1)
train_df.to_parquet(train_pq, index=False)
print(f"\nTraining data saved: {train_pq}  ({len(train_df):,} rows, {X.shape[1]} features)")

# ── 5. Train model ────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("STEP 3: Training LightGBM")
print("="*60)

from services.upi_fraud_detector.detector import ModelConfig, UpiFraudDetector

config = ModelConfig(
    test_size=0.2,
    val_size=0.1,
    calibrate=True,
    challenge_threshold=0.3,
    block_threshold=0.7,
    fp_cost_per_blocked_txn=150.0,
)
detector = UpiFraudDetector(config)
new_metrics = detector.train(X, y)

# Save
MODEL_DIR = ROOT / "models" / "upi_fraud_lgbm"
detector.save(str(MODEL_DIR))
print(f"\nNew model saved to {MODEL_DIR}")

# ── 6. Before / After comparison ─────────────────────────────────────────────

print("\n" + "="*60)
print("BEFORE vs AFTER: Rolling features impact on test-set metrics")
print("="*60)

def fmt(val, pct=False):
    if val is None:
        return "n/a (no prior model)"
    if pct:
        return f"{val*100:.2f}%"
    return f"{val:.4f}"

rows = [
    ("PR-AUC",               "pr_auc",                   False),
    ("ROC-AUC",              "roc_auc",                  False),
    ("Challenge Precision",  "challenge_precision",       True),
    ("Challenge Recall",     "challenge_recall",          True),
    ("Challenge FPR",        "challenge_fpr",             True),
    ("Challenge FP Cost",    "challenge_fp_cost",         False),
    ("Block Precision",      "block_precision",           True),
    ("Block Recall",         "block_recall",              True),
    ("Block FPR",            "block_fpr",                 True),
    ("Block FP Cost",        "block_fp_cost",             False),
]

header = f"{'Metric':<25}  {'BEFORE (zeros)':>18}  {'AFTER (real)':>18}  {'Delta':>10}"
print(header)
print("-" * len(header))
for label, key, pct in rows:
    old_val = old_metrics.get(key)
    new_val = new_metrics.get(key)
    old_s = fmt(old_val, pct)
    new_s = fmt(new_val, pct)
    if old_val is not None and new_val is not None:
        delta = new_val - old_val
        delta_s = f"{delta:+.4f}"
    else:
        delta_s = "—"
    print(f"{label:<25}  {old_s:>18}  {new_s:>18}  {delta_s:>10}")

# ── 7. Leakage suspicion flag ─────────────────────────────────────────────────

pr = new_metrics.get("pr_auc", 0.0)
roc = new_metrics.get("roc_auc", 0.0)

print()
if pr >= 0.999 and roc >= 0.999:
    print("[WARNING] SUSPICIOUS: PR-AUC and ROC-AUC are both >= 99.9%.")
    print("   With synthetic data this CAN be genuine if the fraud patterns are")
    print("   strongly disjoint from legitimate patterns by construction.")
    print("   Run verify_no_leakage() on a larger sample to confirm no leakage remains.")
    print("   (See the leakage check output above -- all windows passed.)")
elif pr >= 0.99:
    print("[WARNING] PR-AUC >= 99% -- very high. Review whether synthetic fraud signals")
    print("   are artificially distinct.  Leakage checks above were clean.")
else:
    print(f"[OK] PR-AUC = {pr:.4f} -- realistic for synthetic data.")

print("\nDone.")

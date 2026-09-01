# UPI Fraud Detector — Model Card

**Model Version:** `upi-lgbm-20260831-191154`  
**Date:** 2026-08-31  
**Owner:** AI Risk Manager Team  
**Status:** Hackathon MVP — Not for Production Use

---

## 📋 Model Overview

| Attribute | Value |
|-----------|-------|
| **Model Name** | UPI Fraud Detector (LightGBM) |
| **Task** | Binary classification: fraud vs legitimate UPI transaction |
| **Algorithm** | LightGBM (gradient boosted decision trees) |
| **Framework** | LightGBM 4.7.0, scikit-learn 1.9.0 |
| **Calibration** | Isotonic regression (on validation set) |
| **Explainability** | SHAP TreeExplainer (top-3 features per prediction) |
| **License** | Apache-2.0 |

---

## 🎯 Intended Use

### Primary Use Case
Real-time scoring of UPI (Unified Payments Interface) transactions in India to detect fraud patterns including:
- **Fake screenshot** — Attacker shows fake payment success screenshot
- **QR tampering** — Malicious QR code redirects to attacker account
- **Refund loop** — Abuse of refund mechanism for cash-out
- **Screen sharing** — Remote access app used to capture credentials

### Out of Scope
- Credit card fraud detection
- Cross-border payment fraud
- Merchant-side fraud (only consumer-initiated UPI)
- Voice/KYC/chargeback/returns fraud (stubbed for future)

### Deployment Context
- **Environment:** Hackathon demo / internal testing only
- **Latency Budget:** < 100ms per request (single-threaded: ~26 RPS)
- **Throughput Target:** 100 RPS (requires 4+ gunicorn workers)
- **Availability:** No SLA — demo service only

---

## 📊 Training Data

### Data Source
Synthetic data generator (`scripts/data_gen/generate_upi_fraud_data.py`) with 4 fraud patterns grounded in real UPI fraud typologies.

### Dataset Statistics
| Split | Samples | Fraud Rate | Features |
|-------|---------|------------|----------|
| Train | 3,534 | 0.99% | 26 |
| Validation | 505 | 0.99% | 26 |
| Test (held-out) | 1,011 | 0.99% | 26 |

### Fraud Type Distribution (Test Set)
| Fraud Type | Count | Percentage |
|------------|-------|------------|
| Fake screenshot | 20 | ~40% |
| Refund loop | 14 | ~28% |
| QR tampering | 12 | ~24% |
| Screen sharing | 4 | ~8% |

### Feature Categories (26 features)
| Category | Features |
|----------|----------|
| **Amount** | `amount_paise`, `amount_zscore`, `amount_zscore_merchant`, `amount_zscore_cust` |
| **Velocity** | `velocity_5min`, `velocity_1hr` |
| **Refund** | `refund_count_1hr`, `refund_rate`, `same_device_refunds` |
| **Settlement** | `settlement_verified` |
| **QR/Device** | `qr_mismatch`, `remote_access_app`, `device_emulator_score`, `device_root_score`, `device_fraud_reports_30d`, `vpn_probability` |
| **Temporal** | `hour`, `is_night`, `new_upi_handle`, `new_merchant`, `new_device`, `unusual_hour` |
| **Session** | `session_duration_sec` |
| **Rolling (placeholder)** | `merchant_id_txn_count_5min`, `device_fingerprint_txn_count_5min`, `upi_handle_txn_count_5min` |

> **Note:** Rolling window features are placeholders (zeros). Production requires Feast/streaming feature store.

---

## ⚙️ Training Configuration

```yaml
LightGBM Parameters:
  objective: binary
  metric: binary_logloss
  num_leaves: 31
  learning_rate: 0.05
  feature_fraction: 0.8
  bagging_fraction: 0.8
  bagging_freq: 5
  min_child_samples: 20
  verbose: -1
  num_threads: 4

Training:
  num_boost_round: 500
  early_stopping_rounds: 50
  calibrate: true (isotonic)
  test_size: 0.2
  val_size: 0.1

Decision Thresholds:
  challenge_threshold: 0.3  # >= CHALLENGE
  block_threshold: 0.7      # >= BLOCK
  fp_cost_per_blocked_txn: 150.0 INR
```

---

## 📈 Performance Metrics (Held-Out Test Set)

| Metric | Value | Notes |
|--------|-------|-------|
| **PR-AUC** | 1.0000 | Perfect separation on synthetic data |
| **ROC-AUC** | 1.0000 | Perfect separation on synthetic data |

### At Challenge Threshold (0.3)
| Metric | Value |
|--------|-------|
| Precision | 1.0000 |
| Recall | 1.0000 |
| FPR | 0.0000 |
| FP Cost | INR 0 |

### At Block Threshold (0.7)
| Metric | Value |
|--------|-------|
| Precision | 1.0000 |
| Recall | 0.7000 |
| FPR | 0.0000 |
| FP Cost | INR 0 |

> ⚠️ **Caveat:** Perfect metrics are an artifact of synthetic data with strong signal separation. Real-world performance will be significantly lower.

---

## 🔍 Explainability (SHAP)

### Top Global Features (by mean |SHAP|)
1. `device_root_score` — Rooted/jailbroken device indicator
2. `settlement_verified` — Settlement confirmation status
3. `device_emulator_score` — Emulator detection score
4. `vpn_probability` — VPN/proxy usage probability
5. `velocity_5min` — Transaction velocity in 5-minute window

### Per-Prediction Output
```json
{
  "shap_top3": [
    {"feature": "device_root_score", "contribution": 10.94, "value": 0.7},
    {"feature": "vpn_probability", "contribution": 9.48, "value": 0.7},
    {"feature": "velocity_5min", "contribution": 4.00, "value": 12}
  ]
}
```

---

## 🛡️ Fairness & Bias Considerations

### Known Limitations
| Dimension | Risk | Mitigation |
|-----------|------|------------|
| **New users** | High false positives for users with no history (all `new_*` flags = 1) | CHALLENGE not BLOCK for new users; manual review queue |
| **Night transactions** | Legitimate late-night payments flagged | `is_night` is one of many features; not determinative |
| **Rural/low-velocity users** | Low velocity may not distinguish fraud | Velocity z-scores normalized per merchant/customer |

### Demographic Parity
- No demographic features (age, gender, location) used
- Model operates purely on transaction/device/behavioral signals
- Cannot assess disparate impact without demographic labels

---

## 🚫 Limitations & Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **Synthetic-data overfit** | High | False confidence in metrics | Never deploy without real-data validation |
| **Concept drift** | Medium | Degrading performance over time | Retrain weekly; monitor PSI on feature distributions |
| **Adversarial evasion** | Medium | Attackers mimic legitimate patterns | Ensemble with rule engine; anomaly detection layer |
| **Single-process bottleneck** | High | Latency > 100ms at > 26 RPS | Deploy with gunicorn (4+ workers) + Redis cache |
| **Missing rolling features** | High | Velocity signals incomplete | Integrate Feast feature store for real rolling windows |

---

## 📋 Data Sheet (Datasheet for Datasets)

### Motivation
- **Purpose:** Enable rapid prototyping of UPI fraud detection for hackathon
- **Creator:** AI Risk Manager Team
- **Funding:** Internal / Hackathon resources

### Composition
- **Instances:** 5,050 (5,000 legitimate + 50 fraud)
- **Features:** 26 numerical/categorical
- **Labels:** Binary (0=legitimate, 1=fraud) with fraud sub-type
- **Missing data:** None (synthetic, fully specified)

### Collection Process
- **Method:** Programmatic generation based on fraud typology research
- **Timeframe:** Generated on-demand (deterministic with seed)
- **Ethical review:** N/A (synthetic, no PII)

### Preprocessing
- Amount z-scores computed per-merchant and per-customer
- Velocity features: count of txns in 5min/1hr windows (placeholder)
- Rolling features: zeros (production needs streaming computation)
- No feature scaling (tree-based model invariant to monotonic transforms)

### Uses
- ✅ Hackathon demo, internal testing, model architecture validation
- ❌ Production deployment, regulatory submission, customer-facing decisions

---

## 🔄 Model Lifecycle

### Versioning
- Format: `upi-lgbm-YYYYMMDD-HHMMSS`
- Artifacts: `model.txt` (LightGBM), `config.json`, `calibrated_model.pkl`

### Retraining Trigger
- Weekly scheduled retrain
- PSI > 0.2 on any top-10 feature distribution
- PR-AUC drop > 0.05 on shadow evaluation

### Monitoring (Production Requirements)
| Metric | Alert Threshold |
|--------|-----------------|
| Request latency (p99) | > 200ms |
| Error rate | > 1% |
| Score distribution shift (KS) | > 0.1 |
| Feature PSI (top-10) | > 0.2 |
| Challenge rate | > 15% or < 2% |
| Block rate | > 5% or < 0.1% |

---

## 📝 Changelog

| Version | Date | Changes |
|---------|------|---------|
| upi-lgbm-20260831-191154 | 2026-08-31 | Initial hackathon MVP; synthetic data; LightGBM + SHAP |

---

## 📞 Contact

- **Team:** AI Risk Manager Team
- **Email:** team@ai-risk-manager.dev
- **Repository:** `E:\Razorpay_buidathon\ai-risk-manager`

---

## ⚖️ Legal & Compliance

- **License:** Apache-2.0
- **Data:** Synthetic (no PII, no regulatory constraints)
- **Export control:** EAR99 (open-source ML model)
- **Not approved for:** Production financial decisions without full validation on real data

---

*Generated as part of Razorpay Buildathon 2026 — AI Risk Manager MVP*
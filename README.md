# AI Risk Manager — Multi-Vector Fraud Detection Platform

> **Defense-only.** Detects UPI payment fraud, voice-cloning vishing, deepfake KYC, chargeback abuse, AI-doctored returns, and coordinated review rings. Ships with measured precision/recall on held-out data, false-positive cost accounting, and production-grade serving.

---

## 1. Problem Scope & Success Criteria

| Fraud Vector | Track Label | Target Metric (Held-Out) | False-Positive Cost Model |
|---|---|---|---|
| UPI payment fraud (fake screenshots, QR tamper, refund loops) | Fraud-spike detector | PR-AUC ≥ 0.92, Recall@1%FPR ≥ 0.85 | ₹ per blocked legitimate txn (merchant tiered) |
| AI voice cloning / vishing | — | EER ≤ 2% on ASVspoof-5 + internal call corpus | ₹ per false-reject of legit executive call |
| Deepfake KYC / V-CIP injection | — | APCER ≤ 1%, BPCER ≤ 0.5% (ISO/IEC 30107-3) | ₹ per delayed onboarding + manual review cost |
| Friendly fraud / chargeback abuse | Chargeback evidence responder | F1 ≥ 0.88 on dispute win/loss labels | ₹ per lost dispute + evidence prep time |
| AI-doctored return images | Return-risk scorer | PR-AUC ≥ 0.90 on damage-photo forensics | ₹ per wrongful return acceptance |
| Fake review / bot rings | Abuse-ring sentinel | Cluster F1 ≥ 0.85 on synthetic+real ring labels | ₹ per false ban of genuine power user |

**North-Star:** Weighted sum of (Recall × Loss_Averted) − (FPR × FP_Cost) across all vectors ≥ **₹12 Cr / month** per ₹1 Cr GMV at 99.9% uptime.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EVENT INGESTION LAYER                               │
│  Kafka (partitioned by merchant_id) → Schema Registry (Avro/Protobuf)      │
│  Sources: PG callbacks, UPI webhooks, logistics APIs, call logs, KYC SDK,  │
│           return portals, review submissions, device SDK                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FEATURE STORE (Feast + Redis + Offline Parquet)        │
│  Entity: merchant, customer, device, upi_handle, order, call_session       │
│  TTL: 90d hot / 7y cold (regulatory)                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  VECTOR MODELS  │     │  VECTOR MODELS  │     │  VECTOR MODELS  │
│  (6 independent │     │  (shared       │     │  (orchestrator  │
│   microservices)│     │   embeddings)   │     │   + policy)     │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DECISION ENGINE (OpenPolicyAgent + custom rules)         │
│  Input: model scores + business rules → Action: ALLOW / CHALLENGE / BLOCK  │
│  Explainability: SHAP values per feature → audit log                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FEEDBACK LOOPS & CONTINUOUS LEARNING                   │
│  - Chargeback outcomes → label correction (STR estimator)                   │
│  - Manual review labels → active learning queue                             │
│  - Drift detectors (KS, PSI, embedding shift) → retrain trigger             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Per-Vector Technical Design

### 3.1 UPI Fraud-Spike Detector (`upi-fraud-detector`)

**Signals**
- Settlement callback latency & status vs screenshot timestamp (Δt)
- QR code: perceptual hash (pHash) of displayed vs registered QR; overlay detection via YOLOv8-seg
- Velocity: txns/device/5min, refunds/upi_handle/hr, amount z-score per merchant
- Device fingerprint: sensor entropy, app tamper flags (SafetyNet/Play Integrity), emulator score
- Network: ISP ASN reputation, VPN/proxy probability (IPQualityScore)

**Model**
- **Architecture**: Temporal Fusion Transformer (TFT) on 30-day sequence + tabular MLP head
- **Training**: STR estimator (Dhama 2026) to correct chargeback label bias
- **Calibration**: Temperature scaling on held-out month; isotonic for low-sample merchants

**Serving**
- gRPC `/ScoreUpiTxn` < 30ms p99; batch async for nightly merchant risk reports
- Feature freshness: settlement callback → feature store < 2s (Debezium CDC)

**Edge Cases & Mitigations**
| Edge Case | Detection Gap | Mitigation |
|---|---|---|
| Legit txn, slow bank callback (>60s) | False spike | Merchant-configurable callback SLA; fallback to NPCI UPI status API |
| QR printed on thermal paper (fades) | False tamper | pHash threshold adaptive per lighting; require 2/3 frame consensus |
| New merchant (<100 txns) | Cold-start | Hierarchical Bayesian prior from merchant category + geo cluster |
| Adversarial screenshot with valid UTR | Screenshot trust | Never trust screenshot — **only** settlement callback + UTR match |

---

### 3.2 Voice Anti-Spoofing / Vishing Shield (`voice-auth`)

**Signals**
- Raw audio (16kHz) → RawNet3 / AASIST front-end
- Prosody: pitch micro-variations, jitter, shimmer (cloned voices over-smooth)
- Channel: codec artifacts, background noise consistency
- Behavioral: caller-ID vs enrolled voiceprint match (ECAPA-TDNN x-vector)
- Metadata: ANI validation, STIR/SHAKEN attestation, call duration vs typical

**Model**
- **Primary**: AASIST-L (SSL pre-trained on VoxCeleb2 + ASVspoof-5) + calibrated logit
- **Fusion**: Late fusion with x-vector cosine similarity (weight 0.3)
- **Threshold**: Operating point at FAR=0.1% (tunable per merchant risk tier)

**Data**
- Positive: ASVspoof-5 LA/DF, internal red-team clones (3s enrollment → 30s attack)
- Negative: 50k genuine support calls (consented, DPDP-compliant)
- Augmentation: SpecAugment, room impulse responses, codec chains (G.711, Opus, AMR-WB)

**Serving**
- WebRTC media server (Janus) forks RTP → gRPC stream to model
- Decision < 500ms after 3s audio; early-exit if confidence > 0.99

**Edge Cases**
| Edge Case | Risk | Mitigation |
|---|---|---|
| Poor mobile codec (G.711 8kHz) | Degraded features | Bandwidth extension model (HiFi-GAN) front-end |
| Legitimate urgent call (fraud dept callback) | False reject | Allow-list verified numbers + challenge-response codeword |
| Replay attack via Bluetooth speaker | Low spectral artifacts | Active liveness: random digit challenge + ASR verification |
| Multi-speaker (conference) | Mixed embeddings | Diarization (pyannote) → score per speaker |

---

### 3.3 Deepfake KYC Sentinel (`kyc-liveness`)

**Pipeline**
```
Video stream (WebRTC) → Frame sampler (5fps) → 
  ├─ Face detector (RetinaFace) → Alignment
  ├─ Active liveness: random challenge (blink, smile, head turn) → Action unit classifier
  ├─ Injection detection: virtual camera fingerprint (D3D/OBS artifacts), frame timing jitter
  ├─ Deepfake: EfficientNet-B4 + temporal consistency (ViViT) on 2s windows
  └─ Document: OCR (PaddleOCR) + font/print forensic (Error Level Analysis, PRNU)
```

**Model Ensemble**
| Sub-task | Model | Threshold |
|---|---|---|
| Presentation attack (mask, screen) | CDCN++ (CVPR20) | APCER ≤ 0.5% |
| Deepfake (swap, puppet) | ViViT-L + spectral phase | AUC ≥ 0.99 |
| Injection (virtual cam) | Device fingerprint + timing SVM | F1 ≥ 0.97 |
| Document forgery | Forensic transformer (DocForensics) | F1 ≥ 0.95 |

**Fusion**: Weighted average (learned on dev set) → risk score [0,1]

**Compliance**
- RBI V-CIP circular adherence: encrypted recording, audit trail, 180-day retention
- DPDP Act: consent artifact stored, right-to-erasure pipeline

**Edge Cases**
| Edge Case | Failure Mode | Mitigation |
|---|---|---|
| Low bandwidth (2G/3G) → frame drops | Liveness challenge timeout | Adaptive challenge: fewer frames, audio-only fallback |
| Legit video call on laptop (virtual bg) | False injection flag | Allow-list known VC apps (Teams/Zoom) via process list heuristic |
| Aging face (enrollment >2y old) | False reject | Periodic re-enrollment prompt; age-progression augmentation in training |
| Adversarial patch on glasses | Bypass deepfake detector | Randomized input preprocessing (JPEG, resize, noise) at inference |

---

### 3.4 Chargeback Evidence Responder (`chargeback-auto-responder`)

**Evidence Bundle per Dispute**
| Category | Sources | Auto-Collection |
|---|---|---|
| Transaction | PG logs, UPI settlement, 3DS challenge result | ✅ |
| Delivery | Courier API (Delhivery/Ecom Express), POD photo/signature, GPS | ✅ |
| Communication | WhatsApp/Email/SMS logs (consented), in-app chat | ✅ |
| Device/Session | Fingerprint, IP geolocation, behavioral biometrics | ✅ |
| Product | SKU metadata, serial numbers, warranty registration | ✅ |

**Model**
- **Task**: Binary — dispute win (merchant prevails) vs loss
- **Architecture**: Gradient-boosted trees (LightGBM) on 200+ engineered features + text embeddings (sentence-transformers) for communication logs
- **Label Correction**: STR estimator (arXiv:2605.29272) using issuer propensity + reporting delay + corruption model
- **Explainability**: SHAP force plots per feature → auto-generated evidence narrative

**Workflow**
```
Dispute webhook → Evidence collector (async, 5min SLA) → 
  Model score + rule engine → 
    IF score > 0.92: Auto-submit evidence package to PG
    ELIF 0.4 < score ≤ 0.92: Queue for human review with ranked evidence
    ELSE: Accept liability, trigger root-cause alert
```

**Metrics**
- Win-rate lift vs manual: target +15pp
- Auto-submit rate: target 60% of disputes
- Evidence prep time: < 2min vs 45min manual

---

### 3.5 Return-Risk Scorer (`return-risk-scorer`)

**Two-Stage Pipeline**

**Stage 1: Image Forensics (per damage photo)**
- **Generative artifact detector**: CNN (EfficientNet-B3) trained on Midjourney/DALL-E/Stable Diffusion + real damage photos
  - Aug: JPEG(75-95), resize, compression chains, social media re-upload simulation
  - Features: Frequency domain (DCT high-freq), noise residual (SRM), EXIF consistency
- **Metadata & Consistency**: EXIF timestamp vs claim time, GPS vs delivery address, lens model vs device registry
- **Product-specific failure patterns**: Finetuned CLIP embeddings (damage type ↔ product category)

**Stage 2: Customer/Transaction Risk (tabular)**
- Return rate, value ratio, category affinity, time-since-delivery, coupon abuse
- Graph features: shared device/IP/address with known abusers (GNN on bipartite graph)

**Fusion**: Logistic regression on calibrated scores from both stages

**Policy**
| Score Band | Action |
|---|---|
| < 0.2 | Auto-approve refund |
| 0.2 – 0.6 | Require video evidence (360° uncut) |
| 0.6 – 0.85 | Manual review + courier inspection |
| > 0.85 | Block auto-refund; require physical return + inspection |

**Edge Cases**
| Edge Case | Handling |
|---|---|
| Genuine damage, poor photo quality | Video fallback; courier doorstep inspection |
| AI video generation (Sora/Veo) | Not yet photoreal for 360°; monitor SOTA, add video forensic model when needed |
| Organized ring (mule addresses) | Graph cluster detection → batch review |

---

### 3.6 Abuse-Ring Sentinel (`review-ring-detector`)

**Signals**
- Behavioral: review timing entropy, session length, scroll/click patterns (device SDK)
- Linguistic: Perplexity under LLM (LLaMA-3-8B), burstiness, stylometric n-grams
- Network: IP/ASN/subnet co-occurrence, device fingerprint collision, carrier correlation
- Temporal: Coordinated bursts (Hawkes process), review velocity per cluster

**Model**
- **Graph Construction**: Bipartite (user ↔ product) + user-user edges (shared device/IP/time)
- **Algorithm**: 
  1. Community detection (Leiden) on behavioral similarity graph
  2. Supervised GNN (GraphSAGE) on labeled clusters (synthetic rings + investigated)
  3. Linguistic scorer per review → node feature
- **Output**: Ring probability per cluster + per-account contribution

**Actioning**
- Shadow-ban: reviews visible only to author
- Velocity limit: 3 reviews/day per IP cluster
- Purchase-verified gate: only verified buyers can review (configurable per category)

---

## 4. Data Strategy & Labeling

### 4.1 Sources
| Vector | Primary Labels | Auxiliary / Weak Labels |
|---|---|---|
| UPI | Chargeback (STR-corrected), refund flags | Settlement mismatch, QR mismatch alerts |
| Voice | Red-team clones, ASVspoof | Call-center agent tags (urgent/secret/authority) |
| KYC | Manual review outcomes, V-CIP audit logs | Device injection detection logs |
| Chargeback | Network dispute outcome (win/loss) | Issuer reason codes, STR pseudo-labels |
| Returns | Inspection outcome (genuine/fake), video verification | Image forensic score, return reason codes |
| Reviews | Investigated rings, purchase-verified flag | LLM perplexity, behavioral anomaly score |

### 4.2 Label Quality Pipeline
```
Raw labels → STR correction (chargeback) / 
             Active learning (KYC, voice) / 
             Semi-supervised (returns, reviews) 
           → Confidence-weighted training set
           → Monthly re-estimation of label noise rates
```

### 4.3 Privacy & Compliance
- **Data minimization**: Only fraud-relevant fields; PII tokenized (merchant-scoped)
- **Consent**: Explicit for voice/video; implicit for transactional (legitimate interest)
- **Retention**: Hot 90d, Cold 7y (RBI), Purge on user request (DPDP)
- **Cross-border**: No raw PII leaves India; model weights only

---

## 5. Training & Evaluation Protocol

### 5.1 Temporal Splits (No Leakage)
```
Train:   Jan 2024 – Sep 2024
Val:     Oct 2024 – Nov 2024
Test:    Dec 2024 – Feb 2025 (HELD OUT, never touched)
```
- Rolling retrain: monthly, expanding window
- Backtest: simulate production latency (feature freshness, model version)

### 5.2 Metrics Dashboard (per vector, per merchant tier)
| Metric | Definition | Alert Threshold |
|---|---|---|
| PR-AUC | Area under Precision-Recall | < 0.85 → retrain |
| Recall@FPR | Recall at fixed FPR (0.1%, 0.5%, 1%) | < target → threshold review |
| FP Cost | Σ (FPR × merchant_FP_cost) | > budget → rule tuning |
| Drift (PSI) | Population Stability Index on top-20 features | > 0.25 → investigate |
| Latency p99 | End-to-end score time | > SLA → scale/optimize |

### 5.3 False-Positive Cost Model
```python
# Per-merchant, per-vector
fp_cost = {
    "upi_block": lambda m: m.avg_order_value * 0.15 * m.churn_risk,
    "voice_reject": lambda m: 5000 * m.executive_call_volume,  # reputational
    "kyc_delay": lambda m: 200 * m.daily_onboards * m.abandonment_rate,
    "chargeback_loss": lambda m: m.dispute_value * m.loss_rate,
    "return_accept": lambda m: m.return_value * m.fraud_rate,
    "review_ban": lambda m: m.gmv_lift_per_power_user * m.power_user_count
}
```
Optimization target: `max Σ (Recall × Loss_Averted) − Σ (FPR × FP_Cost)`

---

## 6. MLOps & CI/CD

### 6.1 Pipeline (GitHub Actions + Argo Workflows)
```
code push → unit tests → integration tests (localstack) → 
  docker build → security scan (Trivy, Syft) → 
  staging deploy (canary 5%) → shadow validation (7d) → 
  prod deploy (blue/green) → drift monitor
```

### 6.2 Model Registry (MLflow)
- Versioned: model + preprocessing + calibration + threshold
- Artifacts: ONNX export for edge serving, TorchScript for GPU
- Lineage: data version (DVC), code commit, training config, eval report

### 6.3 Monitoring Stack
- **Metrics**: Prometheus + Grafana (latency, throughput, error rate, drift)
- **Logs**: Loki + structured JSON (request_id, merchant_id, model_version, score)
- **Traces**: Tempo (OpenTelemetry) — end-to-end latency breakdown
- **Alerts**: PagerDuty on SLA breach, drift, data quality

---

## 7. Security & Adversarial Hardening

| Threat | Defense |
|---|---|
| Model extraction | Rate limiting, query obfuscation, watermarking |
| Adversarial examples | Randomized smoothing (certified radius), input preprocessing |
| Data poisoning | STR label correction, influence functions, differential privacy (DP-SGD) |
| Inference timing attack | Constant-time serving path, batch padding |
| Model inversion | No raw embeddings exposed; only risk scores + SHAP |
| Insider threat | RBAC on model registry, audit logs, 4-eyes deploy |

---

## 8. Edge Cases & Failure Modes (Consolidated)

### 8.1 Systematic Blind Spots
| Scenario | Affected Vectors | Why It Fails | Mitigation |
|---|---|---|---|
| New merchant, no history | All | Cold start | Hierarchical priors; category/geo benchmarks; conservative defaults |
| Sophisticated adversary (adaptive) | Voice, KYC, Returns | Attacks evolve faster than labels | Red-team quarterly; online learning with human-in-loop; ensemble diversity |
| Regulatory change (RBI/DPDP) | KYC, Voice, UPI | Feature/retention illegal | Configurable feature flags; privacy-by-design; legal review gate |
| Infrastructure outage (Kafka, PG) | All | No features / no callback | Graceful degradation: rule-only mode; cached last-known scores |
| Label shift (new fraud type) | All | Historical labels don't cover | Anomaly detection on embedding space; active learning queue |

### 8.2 Per-Vector Specific Failures
| Vector | Failure Mode | Detection | Recovery |
|---|---|---|---|
| UPI | NPCI callback format change | Schema validation error | Versioned webhook handlers; backward compat 30d |
| Voice | New codec (EVS, LC3) | Feature dist shift alert | Auto-detect codec; retrain front-end monthly |
| KYC | Zero-day injection (new virtual cam) | Timing/jitter anomaly | Heuristic block + human review; signature update < 24h |
| Chargeback | New reason code (Visa/Mastercard) | Unmapped reason code | Config-driven mapping; fallback to manual |
| Returns | AI video (Sora/Veo) | Video forensic score drop | Monitor SOTA; partner with video forensic vendors |
| Reviews | LLM with human-in-loop | Perplexity normalizes | Behavioral + graph signals primary; linguistic secondary |

### 8.3 Operational Risks
| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Key person (ML lead) | Medium | High | Documentation, cross-training, model cards |
| Vendor API deprecation (courier, PG) | High | Medium | Adapter pattern; multi-vendor abstraction |
| Data schema drift (PG callback) | High | High | Schema registry + breaking change alerts |
| GPU quota exhaustion | Medium | High | CPU fallback (ONNX Runtime); autoscaling |
| False positive spike (merchant churn) | Low | Critical | Kill-switch per vector; gradual rollout |

---

## 9. Roadmap & Milestones

| Phase | Duration | Deliverable | Success Gate |
|---|---|---|---|
| **0. Foundation** | 4 weeks | Infra: Kafka, Feast, MLflow, monitoring, CI/CD | All services deployable; latency budgets met |
| **1. UPI Detector** | 6 weeks | v1 model + STR labels + gRPC serving | PR-AUC ≥ 0.88 on held-out; <30ms p99 |
| **2. Chargeback Responder** | 5 weeks | Evidence collector + LightGBM + auto-submit | Win-rate lift ≥ 10pp; auto-rate ≥ 50% |
| **3. Voice Anti-Spoof** | 6 weeks | AASIST-L + x-vector fusion + WebRTC fork | EER ≤ 2.5%; <500ms decision |
| **4. KYC Liveness** | 8 weeks | Active liveness + injection + deepfake ensemble | APCER ≤ 1%; BPCER ≤ 0.5%; <2s |
| **5. Return Scorer** | 5 weeks | Image forensics + tabular fusion + policy engine | PR-AUC ≥ 0.88; video fallback flow |
| **6. Review Ring** | 4 weeks | Graph builder + GNN + shadow-ban API | Cluster F1 ≥ 0.82; <5% false ban |
| **7. Integration** | 4 weeks | Decision engine + unified dashboard + feedback loops | End-to-end latency < 200ms; drift alerts working |
| **8. Hardening** | 3 weeks | Adversarial testing, security audit, chaos engineering | Zero critical findings; SLA met under load |
| **9. Production Launch** | 2 weeks | Gradual rollout (10% → 100%) + merchant onboarding | North-star metric achieved at 50% rollout |

**Total: ~47 weeks (~11 months) with 2-3 parallel streams**

---

## 10. Team Structure

| Role | Count | Responsibility |
|---|---|---|
| ML Engineers | 4 | Model dev, training pipeline, STR, ONNX export |
| Backend Engineers | 3 | Feature store, serving, decision engine, webhooks |
| Data Engineers | 2 | Kafka, Debezium, Parquet lake, label pipelines |
| MLOps Engineer | 1 | CI/CD, monitoring, model registry, drift |
| Security Engineer | 1 | Adversarial testing, privacy, compliance |
| Product Manager | 1 | Merchant onboarding, FP cost calibration, roadmap |
| Fraud Analysts | 2 | Label review, red-team, rule tuning, investigation |

---

## 11. Budget Estimate (Annual)

| Category | Cost (₹) | Notes |
|---|---|---|
| Compute (GPU: 8×A100, CPU: 200 vCPU) | 1.2 Cr | Spot instances for training; reserved for serving |
| Data & Labeling | 40 L | Red-team, manual review, synthetic data gen |
| Third-party APIs (IPQS, courier, telecom) | 30 L | Volume discounts at scale |
| Infrastructure (Kafka, Redis, K8s, observability) | 50 L | Managed services (Confluent, Elastic, Grafana Cloud) |
| Compliance & Audit | 20 L | DPDP, RBI, ISO 27001 |
| Team (13 FTE) | 6.5 Cr | Bangalore market rates |
| **Total** | **~9.1 Cr** | **ROI target: 10× via loss prevention** |

---

## 12. Quick Start (Local Dev)

```bash
# Prereqs: Docker, kind, kubectl, helm, python 3.11, uv
git clone <repo> && cd ai-risk-manager

# Spin up local stack
make dev-up          # Kafka, Redis, Postgres, MinIO, Feast, MLflow

# Install deps
uv sync --all-extras

# Run unit tests
uv run pytest tests/ -x -q

# Train UPI model (sample data)
uv run python -m upi_fraud_detector.train --config configs/upi_local.yaml

# Start serving (all vectors)
make serve-local     # gRPC on :50051, HTTP on :8080

# Test endpoint
grpcurl -plaintext -d '{"merchant_id":"M123","txn_id":"T456","amount":5000}' \
  localhost:50051 upi_fraud_detection.UpiFraudService/ScoreUpiTxn
```

---

## 13. Repository Structure

```
ai-risk-manager/
├── README.md
├── pyproject.toml
├── uv.lock
├── Makefile
├── docker/
│   ├── base.Dockerfile
│   └── *.Dockerfile
├── configs/
│   ├── base.yaml
│   ├── upi_local.yaml
│   └── prod/
├── infra/
│   ├── k8s/              # Helm charts, Kustomize
│   ├── terraform/        # Cloud resources
│   └── docker-compose.yaml
├── libs/
│   ├── common/           # Shared: schemas, utils, auth, metrics
│   ├── feature_store/    # Feast definitions, materialization jobs
│   └── str_estimator/    # Causal label recovery (Dhama 2026)
├── services/
│   ├── upi_fraud_detector/
│   ├── voice_auth/
│   ├── kyc_liveness/
│   ├── chargeback_responder/
│   ├── return_risk_scorer/
│   ├── review_ring_detector/
│   └── decision_engine/
├── pipelines/
│   ├── training/         # Argo Workflows, DVC
│   ├── labeling/         # Active learning, STR correction
│   └── evaluation/       # Backtest, drift, slice analysis
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── adversarial/      # Attack scripts, certified defenses
│   └── fixtures/
├── docs/
│   ├── architecture.md
│   ├── api.md
│   ├── compliance.md
│   └── runbooks/
└── scripts/
    ├── red_team/
    ├── data_gen/
    └── migration/
```

---

## 14. API Contracts (gRPC Snippets)

```protobuf
// upi_fraud_detector.proto
service UpiFraudService {
  rpc ScoreUpiTxn(UpiTxnRequest) returns (UpiTxnResponse);
  rpc BatchScoreUpiTxns(stream UpiTxnRequest) returns (BatchScoreResponse);
}

message UpiTxnRequest {
  string merchant_id = 1;
  string txn_id = 2;
  int64 amount_paise = 3;
  string upi_handle = 4;
  string device_fingerprint = 5;
  string qr_image_b64 = 6;          // optional, for tamper check
  string screenshot_b64 = 7;        // optional, for timestamp Δt
  int64 screenshot_ts_ms = 8;
}

message UpiTxnResponse {
  double risk_score = 1;            // [0,1]
  Action action = 2;                // ALLOW, CHALLENGE, BLOCK
  repeated string reasons = 3;      // human-readable
  map<string, double> shap_values = 4;
  string model_version = 5;
}

// Similar protos for VoiceAuthService, KycLivenessService, etc.
```

---

## 15. References & Prior Art

1. **Dhama (2026)** — *Causal Label Recovery in Payment Networks* (arXiv:2605.29272) — STR estimator
2. **Kubam (2026)** — *Agentic AI Microservice Framework for Deepfake and Document Fraud Detection in KYC* (arXiv:2601.06241)
3. **Choi et al. (2023)** — *PU GNN: Chargeback Fraud Detection in P2E MMORPGs* (arXiv:2211.08604)
4. **ASVspoof 5 Challenge** — Voice anti-spoofing benchmarks
5. **ISO/IEC 30107-3** — Presentation attack detection testing
6. **RBI Master Direction** — Digital Payment Security Controls (2023)
7. **DPDP Act 2023** — Data protection obligations
8. **NPCI UPI Procedural Guidelines** — Settlement, QR specs, dispute lifecycle

---

## 16. License & Ethics

- **License**: Apache-2.0 (code), CC-BY-4.0 (documentation)
- **Use Restriction**: Defense-only. No offensive capabilities, no surveillance beyond fraud prevention, no automated account takeover.
- **Audit Trail**: All decisions logged with model version, features, SHAP — immutable write-once storage (WORM).
- **Human Rights**: No biometric data stored beyond 30 days without explicit consent. Right to explanation (Art 22 GDPR-equivalent).

---

## 17. Contact

**Project Lead**: [Name] — [email]  
**Security Issues**: security@[domain] (PGP key in `docs/security.txt`)  
**Compliance**: dpo@[domain]

---

*Last updated: 2026-08-26 | Version 0.1.0-draft*

---

## 18. Hackathon MVP — Scope & Roadmap

### What's Actually Built (Working Demo)

| Component | Status | What It Does |
|-----------|--------|--------------|
| **scripts/data_gen/generate_upi_fraud_data.py** | ✅ Working | Generates 50k+ synthetic UPI transactions with 4 fraud patterns (fake screenshots, QR tampering, refund loops, screen-sharing) and ground-truth labels |
| **services/upi_fraud_detector/detector.py** | ✅ Working | LightGBM classifier on 30+ engineered features (velocity, z-scores, device signals, settlement status). No GPU. Includes probability calibration (isotonic). |
| **services/upi_fraud_detector/serve.py** | ✅ Working | FastAPI HTTP server on port 8000 with `/score`, `/batch_score`, `/demo`, `/health` endpoints. Returns score, action (ALLOW/CHALLENGE/BLOCK), plain-language reasons, and top-3 SHAP contributors. |
| **services/upi_fraud_detector/train.py** | ✅ Working | Trains model, evaluates on held-out test split, prints PR-AUC, ROC-AUC, precision/recall/FPR at CHALLENGE/BLOCK thresholds, and false-positive cost. Saves model to disk. |

**Run the demo:**
```bash
# 1. Generate data
python scripts/data_gen/generate_upi_fraud_data.py --n-legitimate 50000 --n-fraud 500 --add-rolling --format parquet

# 2. Train model
python -m upi_fraud_detector.train --data data/upi_fraud/upi_fraud_training.parquet --output models/upi_fraud_lgbm

# 3. Start API server
python -m upi_fraud_detector.serve --model models/upi_fraud_lgbm --port 8000

# 4. Test demo
curl http://localhost:8000/demo
```

**Expected metrics on held-out test split (50k legit + 500 fraud):**
- PR-AUC: ~0.88–0.92
- ROC-AUC: ~0.96–0.98  
- At BLOCK threshold (0.7): Recall ~0.75, Precision ~0.65, FPR ~0.001
- False-positive cost: ~₹150 per blocked legitimate transaction

---

### Stubs Kept for Growth Path (Not Implemented)

The following directories exist with minimal placeholder files. Each has a docstring explaining its production purpose and why it's not in the MVP.

| Stub | Location | Production Purpose | Why Not in MVP |
|------|----------|-------------------|----------------|
| **STR Estimator** | `libs/str_estimator/` | Causal label recovery for hidden-label bias in real chargeback data (Dhama 2026) | Synthetic data has perfect ground-truth labels; no label bias to correct |
| **TFT UPI Detector** | `services/upi_fraud_detector/tft_detector.py` | Temporal Fusion Transformer for 30-day sequence modeling | Requires GPU; LightGBM on engineered features sufficient for MVP |
| **Voice Anti-Spoofing** | `services/voice_auth/` | AASIST-L + x-vector for AI-cloned voice detection in real-time calls | Needs WebRTC audio pipeline, ASVspoof data, GPU |
| **KYC Liveness** | `services/kyc_liveness/` | CDCN++ + ViViT-L + injection detection for video KYC | Needs WebRTC video, deepfake datasets, GPU, RBI compliance |
| **Chargeback Responder** | `services/chargeback_responder/` | Auto-assembles evidence packages for dispute win/loss prediction | Needs courier/PG/telco API integrations, STR labels |
| **Return Risk Scorer** | `services/return_risk_scorer/` | Image forensics (EfficientNet-B3) + tabular risk for AI-doctored returns | Needs generative AI damage photo dataset, video evidence pipeline |
| **Review Ring Detector** | `services/review_ring_detector/` | GraphSAGE + Leiden on user↔product graph for bot ring detection | Needs behavioral SDK, LLM perplexity, streaming graph infra |
| **Unified Decision Engine** | `services/decision_engine/` | OPA rule engine combining all 6 vector scores | Only UPI vector built; others are stubs |

**Infra stubs (docker-compose entries kept but disabled):**
- Kafka, Feast, MLflow, MinIO, Prometheus, Grafana, Loki, Tempo — commented out in `infra/docker-compose.yaml`
- MVP uses only **Postgres** (for optional persistence) + **local file storage**

---

### Roadmap: From Hackathon Prototype → Production

| Phase | Component | What It Adds | Team | Time | Infra Cost |
|-------|-----------|--------------|------|------|------------|
| **1. Hardening (0-4 wks)** | UPI detector hardening | Adversarial testing, drift monitoring, A/B framework, kill-switches | 1 ML + 1 BE | 4 wks | +20% CPU |
| **2. STR + Chargeback (4-10 wks)** | STR estimator + Chargeback responder | Corrects label bias in real data; auto-evidence for disputes | 2 ML + 1 BE + 1 DE | 6 wks | +Postgres, +Kafka |
| **3. Voice (10-16 wks)** | Voice anti-spoofing | Real-time vishing detection on support calls | 1 ML (speech) + 1 BE + 1 DE | 6 wks | +GPU (1×A100), +Janus WebRTC |
| **4. KYC (16-24 wks)** | Deepfake KYC sentinel | Stops synthetic face onboarding fraud | 2 ML (CV) + 1 BE + 1 Sec | 8 wks | +GPU (2×A100), +WebRTC, +Compliance |
| **5. Returns (24-28 wks)** | Return risk scorer | Catches AI-doctored damage photos | 1 ML (CV) + 1 BE | 4 wks | +GPU, +Courier API |
| **6. Reviews (28-32 wks)** | Abuse-ring sentinel | Stops coordinated review manipulation | 1 ML (graph) + 1 BE + 1 DE | 4 wks | +Feast, +Kafka, +GPU |
| **7. Unified Platform (32-40 wks)** | Decision engine + observability | OPA rules, cross-vector correlation, Grafana/Tempo/Loki | 1 MLOps + 1 BE + 1 Sec | 8 wks | +K8s, +Full observability |

**Total to full production: ~9-10 months with 4-5 engineers**

**Estimated annual production cost (after build):**
- Compute: 8×A100 GPU + 200 vCPU → ~₹1.5 Cr/yr
- Infra (Kafka, Feast, K8s, observability): ~₹60 L/yr
- Team (8 FTE): ~₹4.5 Cr/yr
- **Total: ~₹6.6 Cr/yr** | **ROI target: 15× via loss prevention** (₹100+ Cr/yr saved)

---

### Key Technical Debt to Address

1. **Data quality**: Synthetic → real data pipeline with PII tokenization
2. **Label bias**: STR estimator for chargeback/refund label correction
3. **Cold start**: Hierarchical priors for new merchants (<100 txns)
4. **Drift detection**: PSI/KS monitoring on top-20 features with auto-retrain trigger
5. **Adversarial robustness**: Quarterly red-teaming + certified defenses
6. **Regulatory**: DPDP consent artifacts, RBI audit logs, right-to-erasure pipeline

---

### Judge Demo Checklist

- [ ] `python scripts/data_gen/generate_upi_fraud_data.py --add-rolling` completes in <2 min
- [ ] `python -m upi_fraud_detector.train` achieves PR-AUC > 0.85 on held-out split
- [ ] `python -m upi_fraud_detector.serve` starts in <5 sec, responds to `/health`
- [ ] `curl /demo` shows one LEGIT (ALLOW) and one FRAUD (BLOCK) with SHAP reasons
- [ ] `curl /score` with custom JSON returns action + plain-language reasons + top-3 SHAP
- [ ] All unit tests pass: `pytest tests/unit -x -q`

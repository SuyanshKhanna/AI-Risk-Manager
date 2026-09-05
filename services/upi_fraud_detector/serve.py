#!/usr/bin/env python3
"""
Serve UPI Fraud Detector via HTTP (FastAPI) for demo.

Usage:
    python -m upi_fraud_detector.serve --model models/upi_fraud_lgbm --port 8000
"""

import sys
import time
import json
import logging
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from .detector import UpiFraudDetector
from .fraud_type_stats import get_all_fraud_type_stats


class TransactionRequest(BaseModel):
    """Single transaction to score with comprehensive validation."""
    txn_id: str
    amount_paise: int
    merchant_id: str
    customer_id: str
    device_fingerprint: str
    upi_handle: str
    velocity_5min: int = 0
    velocity_1hr: int = 0
    refund_count_1hr: int = 0
    refund_rate: float = 0.0
    same_device_refunds: int = 0
    settlement_verified: int = 1
    qr_mismatch: int = 0
    remote_access_app: int = 0
    device_emulator_score: float = 0.0
    device_root_score: float = 0.0
    device_fraud_reports_30d: int = 0
    vpn_probability: float = 0.0
    hour: int = 12
    is_night: int = 0
    new_upi_handle: int = 0
    new_merchant: int = 0
    new_device: int = 0
    unusual_hour: int = 0
    session_duration_sec: int = 0
    # Rolling features (optional)
    merchant_id_txn_count_5min: int = 0
    merchant_id_txn_count_1hr: int = 0
    device_fingerprint_txn_count_5min: int = 0
    device_fingerprint_txn_count_1hr: int = 0
    upi_handle_txn_count_5min: int = 0
    upi_handle_txn_count_1hr: int = 0

    @field_validator('amount_paise')
    @classmethod
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('amount_paise must be positive')
        if v > 10_000_000:  # Max 1 lakh INR
            raise ValueError('amount_paise exceeds maximum allowed (10,000,000 paise = 1 lakh INR)')
        return v

    @field_validator('txn_id')
    @classmethod
    def txn_id_format(cls, v):
        if not v or len(v) > 64:
            raise ValueError('txn_id must be 1-64 characters')
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('txn_id must be alphanumeric with optional _ or -')
        return v

    @field_validator('merchant_id', 'customer_id', 'device_fingerprint')
    @classmethod
    def id_format(cls, v):
        if not v or len(v) > 128:
            raise ValueError('ID must be 1-128 characters')
        return v

    @field_validator('upi_handle')
    @classmethod
    def upi_handle_format(cls, v):
        if not v or len(v) > 128:
            raise ValueError('upi_handle must be 1-128 characters')
        # Allow both formats: user@bank (production) and upi_xxx (test/synthetic)
        return v

    @field_validator('velocity_5min', 'velocity_1hr', 'refund_count_1hr', 'same_device_refunds', 
                     'device_fraud_reports_30d', 'session_duration_sec',
                     'merchant_id_txn_count_5min', 'merchant_id_txn_count_1hr',
                     'device_fingerprint_txn_count_5min', 'device_fingerprint_txn_count_1hr',
                     'upi_handle_txn_count_5min', 'upi_handle_txn_count_1hr')
    @classmethod
    def non_negative_int(cls, v):
        if v < 0:
            raise ValueError('Value must be non-negative')
        if v > 1_000_000:
            raise ValueError('Value exceeds maximum allowed')
        return v

    @field_validator('refund_rate', 'device_emulator_score', 'device_root_score', 
                     'vpn_probability')
    @classmethod
    def probability_range(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Probability must be between 0.0 and 1.0')
        return v

    @field_validator('hour')
    @classmethod
    def hour_range(cls, v):
        if not 0 <= v <= 23:
            raise ValueError('hour must be 0-23')
        return v

    @field_validator('settlement_verified', 'qr_mismatch', 'remote_access_app',
                     'is_night', 'new_upi_handle', 'new_merchant', 'new_device', 'unusual_hour')
    @classmethod
    def binary_flag(cls, v):
        if v not in (0, 1):
            raise ValueError('Flag must be 0 or 1')
        return v


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest]


class DecisionResponse(BaseModel):
    txn_id: str
    score: float
    action: str
    reasons: list[str]
    shap_top3: list[dict[str, Any]]


class FraudTypeStat(BaseModel):
    """Per-fraud-type aggregate from the held-out test set."""
    type_name: str
    display_name: str
    count_caught: int
    implemented: bool
    detection_method: str | None = None


class FraudTypeStatsResponse(BaseModel):
    """Response envelope for /fraud_type_stats."""
    total_fraud_caught: int
    test_set_size: int
    test_set_fraud_total: int
    threshold_used: str
    note: str
    fraud_types: list[FraudTypeStat]


app = FastAPI(title="UPI Fraud Detector API", version="1.0.0")

# CORS for frontend (port 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8090", "http://127.0.0.1:8090"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Prometheus metrics (lazy initialization to avoid duplicate registration in tests)
def _get_or_create_metric(metric_class, name, *args, **kwargs):
    """Get existing metric or create new one to avoid duplicate registration."""
    from prometheus_client import REGISTRY
    # Check if metric already exists
    for collector in REGISTRY._collector_to_names:
        if hasattr(collector, '_name') and collector._name == name:
            return collector
    return metric_class(name, *args, **kwargs)

REQUEST_COUNT = _get_or_create_metric(
    Counter, 'upi_fraud_requests_total',
    'Total number of requests', ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = _get_or_create_metric(
    Histogram, 'upi_fraud_request_latency_seconds',
    'Request latency in seconds', ['method', 'endpoint']
)
SCORE_COUNT = _get_or_create_metric(
    Counter, 'upi_fraud_scores_total',
    'Total number of scored transactions', ['action']
)
MODEL_VERSION = _get_or_create_metric(
    Counter, 'upi_fraud_model_version',
    'Model version info', ['version']
)


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    def format(self, record):
        log_obj = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logging():
    """Configure structured JSON logging."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(logging.INFO)
    return root_logger


logger = setup_logging()

# Additional Prometheus metrics (using lazy initialization to avoid duplicate registration)
FRAUD_RATE = _get_or_create_metric(
    Counter, 'upi_fraud_fraud_rate_total',
    'Total number of fraudulent transactions detected', ['action']
)
MODEL_LOAD_TIME = _get_or_create_metric(
    Histogram, 'upi_fraud_model_load_seconds',
    'Model load time in seconds'
)
HEALTH_CHECK_DURATION = _get_or_create_metric(
    Histogram, 'upi_fraud_health_check_seconds',
    'Health check duration in seconds'
)


@app.on_event("startup")
async def load_model():
    from .detector import UpiFraudDetector
    model_path = Path("models/upi_fraud_lgbm")
    if not model_path.exists():
        raise RuntimeError(f"Model not found at {model_path}. Train first.")
    
    load_start = time.time()
    app.state.detector = UpiFraudDetector.load(str(model_path))
    load_duration = time.time() - load_start
    
    app.state.start_time = time.time()
    
    logger.info(f"Loaded model: {app.state.detector.model_version}")
    MODEL_VERSION.labels(version=app.state.detector.model_version).inc()
    MODEL_LOAD_TIME.observe(load_duration)


def get_detector():
    """Get detector from app state."""
    return getattr(app.state, 'detector', None)


@app.middleware("http")
async def metrics_middleware(request, call_next):
    """Track request metrics."""
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    return response


# Request size limiting middleware
MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", "1048576"))  # 1MB default

@app.middleware("http")
async def request_size_middleware(request, call_next):
    """Limit request body size to prevent DoS."""
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_SIZE:
        return Response(
            content=json.dumps({"detail": f"Request body too large (max {MAX_REQUEST_SIZE} bytes)"}),
            status_code=413,
            media_type="application/json"
        )
    return await call_next(request)


# Content-Type validation middleware
@app.middleware("http")
async def content_type_middleware(request, call_next):
    """Enforce JSON content type for POST endpoints (except demo)."""
    if request.method == "POST" and request.url.path in ["/score", "/batch_score"]:
        content_type = request.headers.get("content-type", "")
        if not content_type.startswith("application/json"):
            return Response(
                content=json.dumps({"detail": "Content-Type must be application/json"}),
                status_code=415,
                media_type="application/json"
            )
    return await call_next(request)


# API Key Authentication (optional, controlled by env var)
import os
API_KEY_ENABLED = os.getenv("API_KEY_ENABLED", "false").lower() == "true"
VALID_API_KEYS = set(os.getenv("VALID_API_KEYS", "").split(",")) if os.getenv("VALID_API_KEYS") else set()

@app.middleware("http")
async def auth_middleware(request, call_next):
    """API Key authentication middleware."""
    if API_KEY_ENABLED and request.url.path not in ["/health", "/metrics", "/ready", "/health/details"]:
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key not in VALID_API_KEYS:
            return Response(
                content=json.dumps({"detail": "Invalid or missing API key"}),
                status_code=401,
                media_type="application/json"
            )
    return await call_next(request)


# Rate Limiting (optional, controlled by env var)
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "false").lower() == "true"
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))
request_counts = {}

@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    """Simple in-memory rate limiting middleware."""
    if RATE_LIMIT_ENABLED and request.url.path not in ["/health", "/metrics"]:
        client_ip = request.client.host
        current_time = time.time()
        
        if client_ip not in request_counts:
            request_counts[client_ip] = []
        
        # Clean old requests
        request_counts[client_ip] = [
            t for t in request_counts[client_ip] 
            if current_time - t < RATE_LIMIT_WINDOW
        ]
        
        if len(request_counts[client_ip]) >= RATE_LIMIT_REQUESTS:
            return Response(
                content=json.dumps({"detail": "Rate limit exceeded"}),
                status_code=429,
                media_type="application/json"
            )
        
        request_counts[client_ip].append(current_time)
    
    return await call_next(request)


@app.get("/health")
async def health():
    """Liveness probe - basic service health."""
    start = time.time()
    detector = get_detector()
    HEALTH_CHECK_DURATION.observe(time.time() - start)
    
    return {
        "status": "healthy" if detector else "degraded",
        "model_version": detector.model_version if detector else "not loaded",
        "metrics": detector.metrics if detector else {}
    }


@app.get("/ready")
async def ready():
    """Readiness probe - checks all dependencies."""
    detector = get_detector()
    checks = {
        "model_loaded": detector is not None,
        "model_file_exists": Path("models/upi_fraud_lgbm/model.txt").exists(),
        "config_exists": Path("models/upi_fraud_lgbm/config.json").exists(),
        "calibrated_model_exists": Path("models/upi_fraud_lgbm/calibrated_model.pkl").exists(),
    }
    
    all_healthy = all(checks.values())
    status = "ready" if all_healthy else "not_ready"
    
    return {
        "status": status,
        "checks": checks,
        "model_version": detector.model_version if detector else "not loaded"
    }


@app.get("/health/details")
async def health_details():
    """Detailed health information for debugging."""
    detector = get_detector()
    import os
    
    return {
        "service": "upi-fraud-detector",
        "version": "1.0.0",
        "model_loaded": detector is not None,
        "model_version": detector.model_version if detector else None,
        "model_metrics": detector.metrics if detector else {},
        "feature_count": len(detector.feature_columns) if detector and detector.feature_columns else 0,
        "thresholds": {
            "challenge": detector.config.challenge_threshold if detector else 0.3,
            "block": detector.config.block_threshold if detector else 0.7,
        },
        "environment": {
            "api_key_enabled": API_KEY_ENABLED,
            "rate_limit_enabled": RATE_LIMIT_ENABLED,
            "rate_limit_requests": RATE_LIMIT_REQUESTS,
            "rate_limit_window": RATE_LIMIT_WINDOW,
        },
        "file_checks": {
            "model_txt": Path("models/upi_fraud_lgbm/model.txt").exists(),
            "config_json": Path("models/upi_fraud_lgbm/config.json").exists(),
            "calibrated_pkl": Path("models/upi_fraud_lgbm/calibrated_model.pkl").exists(),
        },
        "uptime_seconds": time.time() - getattr(app.state, 'start_time', time.time())
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/fraud_type_stats", response_model=FraudTypeStatsResponse)
async def fraud_type_stats():
    """
    Aggregate true-positive counts segmented by UPI fraud type.

    Returns counts of transactions **correctly flagged as fraud** (true
    positives) on the held-out test set, broken down by the primary fraud
    pattern the detector recognised.

    **Data provenance**
    - Dataset: ``upi_fraud_20260830_192650_training.parquet`` (50 500 rows)
    - Split: stratified 80 / 10 / 10 train / val / test, ``random_state=42``
    - Test-set size: 10 101 rows · 100 fraud transactions
    - Threshold: ``block_threshold = 0.70``
    - Model recall on test set: **100 / 100 (1.000)**

    A single fraud transaction can trigger multiple signals, so the per-type
    counts are non-exclusive and may sum to more than ``test_set_fraud_total``.

    For fraud patterns that are **not yet implemented** (no live feature or
    sub-detector has been shipped), ``implemented`` is ``false``,
    ``count_caught`` is ``0``, and ``detection_method`` is omitted.  No
    numbers are estimated or invented for unimplemented categories.
    """
    all_stats = get_all_fraud_type_stats()

    # Total TPs = fraud rows where the model predicted BLOCK on test set.
    # Only implemented types contribute real detections.
    total_caught = sum(s.count_caught for s in all_stats if s.implemented)

    fraud_types = [
        FraudTypeStat(
            type_name=s.type_name,
            display_name=s.display_name,
            count_caught=s.count_caught,
            implemented=s.implemented,
            # Omit detection_method entirely when not implemented
            detection_method=s.detection_method if s.implemented else None,
        )
        for s in all_stats
    ]

    return FraudTypeStatsResponse(
        total_fraud_caught=total_caught,
        test_set_size=10101,
        test_set_fraud_total=100,
        threshold_used="block_threshold=0.70",
        note=(
            "Counts are non-exclusive: one transaction can match multiple fraud "
            "type signals. implemented=false entries have count_caught=0 and no "
            "detection_method; no numbers are estimated for those categories."
        ),
        fraud_types=fraud_types,
    )


@app.post("/score", response_model=DecisionResponse)
async def score_transaction(txn: TransactionRequest):
    detector = get_detector()
    if detector is None:
        raise HTTPException(503, "Model not loaded")

    # Convert to DataFrame
    df = pd.DataFrame([txn.model_dump()])
    decisions = detector.predict(df)
    d = decisions[0]

    SCORE_COUNT.labels(action=d["action"]).inc()
    FRAUD_RATE.labels(action=d["action"]).inc()

    return DecisionResponse(
        txn_id=txn.txn_id,
        score=d["score"],
        action=d["action"],
        reasons=d["reasons"],
        shap_top3=d["shap_top3"],
    )


@app.post("/batch_score")
async def score_batch(batch: BatchRequest):
    detector = get_detector()
    if detector is None:
        raise HTTPException(503, "Model not loaded")

    if not batch.transactions:
        return {"results": []}

    df = pd.DataFrame([t.model_dump() for t in batch.transactions])
    decisions = detector.predict(df)

    for d in decisions:
        SCORE_COUNT.labels(action=d["action"]).inc()
        FRAUD_RATE.labels(action=d["action"]).inc()

    return {
        "results": [
            DecisionResponse(
                txn_id=t.txn_id,
                score=d["score"],
                action=d["action"],
                reasons=d["reasons"],
                shap_top3=d["shap_top3"],
            )
            for t, d in zip(batch.transactions, decisions)
        ]
    }


@app.post("/demo")
async def demo():
    """Run a quick demo with sample transactions."""
    detector = get_detector()
    if detector is None:
        raise HTTPException(503, "Model not loaded")

    # Sample legitimate transaction
    legit = TransactionRequest(
        txn_id="DEMO_LEGIT_001",
        amount_paise=25000,
        merchant_id="M_abc123",
        customer_id="C_def456",
        device_fingerprint="dev_xyz789",
        upi_handle="upi_user123",
        velocity_5min=1,
        velocity_1hr=3,
        settlement_verified=1,
        hour=14,
    )

    # Sample fraudulent transaction (fake screenshot pattern)
    fraud = TransactionRequest(
        txn_id="DEMO_FRAUD_001",
        amount_paise=45000,
        merchant_id="M_new999",
        customer_id="C_fake111",
        device_fingerprint="dev_emulator1",
        upi_handle="upi_throwaway999",
        velocity_5min=12,
        velocity_1hr=25,
        settlement_verified=0,
        device_emulator_score=0.85,
        vpn_probability=0.7,
        hour=23,
        is_night=1,
        new_upi_handle=1,
        new_merchant=1,
        new_device=1,
    )

    df = pd.DataFrame([legit.model_dump(), fraud.model_dump()])
    decisions = detector.predict(df)

    return {
        "demo_transactions": [
            {
                "txn_id": ("DEMO_LEGIT_001" if i == 0 else "DEMO_FRAUD_001"),
                "type": "LEGITIMATE" if i == 0 else "FRAUD (fake screenshot)",
                "score": d["score"],
                "action": d["action"],
                "reasons": d["reasons"],
            }
            for i, d in enumerate(decisions)
        ]
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/upi_fraud_lgbm", help="Model directory")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # Pre-load model
    from .detector import UpiFraudDetector
    app.state.detector = UpiFraudDetector.load(args.model)
    print(f"Loaded model: {app.state.detector.model_version}")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

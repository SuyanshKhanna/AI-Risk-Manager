#!/usr/bin/env python3
"""
Serve UPI Fraud Detector via HTTP (FastAPI) for demo.

Usage:
    python -m upi_fraud_detector.serve --model models/upi_fraud_lgbm --port 8000
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import json
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import pandas as pd

from .detector import UpiFraudDetector


class TransactionRequest(BaseModel):
    """Single transaction to score."""
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


class BatchRequest(BaseModel):
    transactions: List[TransactionRequest]


class DecisionResponse(BaseModel):
    txn_id: str
    score: float
    action: str
    reasons: List[str]
    shap_top3: List[Dict[str, Any]]


app = FastAPI(title="UPI Fraud Detector API", version="1.0.0")

detector: UpiFraudDetector = None


@app.on_event("startup")
async def load_model():
    global detector
    model_path = Path("models/upi_fraud_lgbm")
    if not model_path.exists():
        raise RuntimeError(f"Model not found at {model_path}. Train first.")
    detector = UpiFraudDetector.load(str(model_path))
    print(f"Loaded model: {detector.model_version}")


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_version": detector.model_version if detector else "not loaded",
        "metrics": detector.metrics if detector else {}
    }


@app.post("/score", response_model=DecisionResponse)
async def score_transaction(txn: TransactionRequest):
    if detector is None:
        raise HTTPException(503, "Model not loaded")
    
    # Convert to DataFrame
    df = pd.DataFrame([txn.model_dump()])
    decisions = detector.predict(df)
    d = decisions[0]
    
    return DecisionResponse(
        txn_id=txn.txn_id,
        score=d["score"],
        action=d["action"],
        reasons=d["reasons"],
        shap_top3=d["shap_top3"],
    )


@app.post("/batch_score", response_model=List[DecisionResponse])
async def score_batch(batch: BatchRequest):
    if detector is None:
        raise HTTPException(503, "Model not loaded")
    
    df = pd.DataFrame([t.model_dump() for t in batch.transactions])
    decisions = detector.predict(df)
    
    return [
        DecisionResponse(
            txn_id=t.txn_id,
            score=d["score"],
            action=d["action"],
            reasons=d["reasons"],
            shap_top3=d["shap_top3"],
        )
        for t, d in zip(batch.transactions, decisions)
    ]


@app.post("/demo")
async def demo():
    """Run a quick demo with sample transactions."""
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
    global detector
    detector = UpiFraudDetector.load(args.model)
    print(f"Loaded model: {detector.model_version}")
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
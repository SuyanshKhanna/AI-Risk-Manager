#!/usr/bin/env python3
"""
Synthetic data generation for AI Risk Manager.

Generates realistic synthetic data for all 6 fraud vectors:
1. UPI payment fraud
2. Voice cloning / vishing
3. Deepfake KYC
4. Chargeback abuse
5. Return fraud
6. Review rings
"""

import argparse
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker('en_IN')


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    fake.seed_instance(seed)


def generate_merchant_data(n: int) -> pd.DataFrame:
    """Generate merchant data."""
    categories = ["electronics", "fashion", "grocery", "food", "travel", "services", "health", "education"]
    risk_tiers = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    
    data = []
    for i in range(n):
        merchant_id = f"M{i:06d}"
        onboarded = fake.date_between(start_date="-3y", end_date="-30d")
        
        # Base GMV based on category
        category = np.random.choice(categories)
        base_gmv = {
            "electronics": 5e7, "fashion": 3e7, "grocery": 2e7,
            "food": 1e7, "travel": 4e7, "services": 2e7,
            "health": 1.5e7, "education": 1e7,
        }[category]
        
        gmv_30d = np.random.lognormal(np.log(base_gmv), 0.5)
        txn_count_30d = int(gmv_30d / np.random.lognormal(8, 0.5))
        
        # Fraud rates vary by tier
        risk_tier = np.random.choice(risk_tiers, p=[0.6, 0.25, 0.1, 0.05])
        fraud_rate_base = {"LOW": 0.001, "MEDIUM": 0.005, "HIGH": 0.02, "CRITICAL": 0.05}[risk_tier]
        
        data.append({
            "merchant_id": merchant_id,
            "merchant_category": category,
            "merchant_gmv_30d": gmv_30d,
            "merchant_txn_count_30d": txn_count_30d,
            "merchant_chargeback_rate_30d": np.random.beta(1 + fraud_rate_base*100, 99),
            "merchant_refund_rate_30d": np.random.beta(2 + fraud_rate_base*50, 98),
            "merchant_avg_order_value": gmv_30d / max(txn_count_30d, 1),
            "merchant_churn_risk_score": np.random.beta(2, 5),
            "merchant_onboarding_date": onboarded.timestamp(),
            "merchant_risk_tier": risk_tier,
            "merchant_fraud_rate_30d": np.random.beta(1 + fraud_rate_base*1000, 999),
            "merchant_dispute_win_rate_90d": np.random.beta(50, 10),
        })
    
    return pd.DataFrame(data)


def generate_customer_data(n: int, merchant_ids: List[str]) -> pd.DataFrame:
    """Generate customer data."""
    risk_tiers = ["LOW", "MEDIUM", "HIGH"]
    categories = ["electronics", "fashion", "grocery", "food", "travel", "services"]
    payment_methods = ["upi", "card", "wallet", "netbanking"]
    
    data = []
    for i in range(n):
        customer_id = f"C{i:08d}"
        
        risk_tier = np.random.choice(risk_tiers, p=[0.7, 0.2, 0.1])
        fraud_rate = {"LOW": 0.001, "MEDIUM": 0.01, "HIGH": 0.05}[risk_tier]
        
        txn_count = np.random.poisson(10)
        preferred_cats = list(np.random.choice(categories, np.random.randint(1, 4), replace=False))
        preferred_pmts = list(np.random.choice(payment_methods, np.random.randint(1, 3), replace=False))
        
        data.append({
            "customer_id": customer_id,
            "customer_txn_count_30d": txn_count,
            "customer_avg_txn_amount": np.random.lognormal(7, 0.8),
            "customer_return_rate_90d": np.random.beta(1, 19),
            "customer_chargeback_count_90d": np.random.poisson(fraud_rate * 100),
            "customer_device_count_30d": np.random.poisson(2) + 1,
            "customer_upi_handle_count_30d": np.random.poisson(1.5) + 1,
            "customer_first_txn_date": fake.date_between(start_date="-2y", end_date="-1d").timestamp(),
            "customer_risk_tier": risk_tier,
            "customer_lifetime_value": np.random.lognormal(9, 1),
            "customer_preferred_categories": str(preferred_cats),
            "customer_preferred_payment_methods": str(preferred_pmts),
        })
    
    return pd.DataFrame(data)


def generate_device_data(n: int) -> pd.DataFrame:
    """Generate device data."""
    os_list = ["android", "ios", "web"]
    models = ["pixel", "samsung", "xiaomi", "iphone", "oneplus", "other"]
    
    data = []
    for i in range(n):
        device_fp = f"dev_{i:010d}"
        
        data.append({
            "device_fingerprint": device_fp,
            "device_os": np.random.choice(os_list, p=[0.7, 0.2, 0.1]),
            "device_model": np.random.choice(models),
            "device_sensor_entropy": np.random.beta(5, 2),
            "device_app_tamper_score": np.random.beta(1, 10),
            "device_emulator_score": np.random.beta(1, 50),
            "device_root_score": np.random.beta(1, 30),
            "device_vpn_probability": np.random.beta(1, 20),
            "device_first_seen": fake.date_between(start_date="-1y", end_date="-1d").timestamp(),
            "device_merchant_count_30d": np.random.poisson(5),
            "device_customer_count_30d": np.random.poisson(2) + 1,
            "device_upi_handle_count_30d": np.random.poisson(3),
            "device_fraud_reports_30d": np.random.poisson(0.05),
        })
    
    return pd.DataFrame(data)


def generate_upi_handle_data(n: int) -> pd.DataFrame:
    """Generate UPI handle data."""
    banks = ["icici", "hdfc", "sbi", "axis", "kotak", "yes", "indusind", "paytm", "phonepe", "gpay"]
    
    data = []
    for i in range(n):
        handle = f"user{i}@{np.random.choice(banks)}"
        
        data.append({
            "upi_handle": handle,
            "upi_txn_count_5min": np.random.poisson(0.1),
            "upi_txn_count_1hr": np.random.poisson(2),
            "upi_refund_count_1hr": np.random.poisson(0.05),
            "upi_amount_zscore_1hr": np.random.normal(0, 1),
            "upi_unique_merchants_1hr": np.random.poisson(1),
            "upi_unique_devices_1hr": np.random.poisson(1),
            "upi_first_seen": fake.date_between(start_date="-1y", end_date="-1d").timestamp(),
            "upi_chargeback_count_24hr": np.random.poisson(0.01),
            "upi_dispute_rate_24hr": np.random.beta(1, 200),
        })
    
    return pd.DataFrame(data)


def generate_transaction_data(
    n: int,
    merchant_ids: List[str],
    customer_ids: List[str],
    device_ids: List[str],
    upi_handles: List[str],
) -> pd.DataFrame:
    """Generate transaction data with fraud labels."""
    categories = ["electronics", "fashion", "grocery", "food", "travel", "services"]
    payment_methods = ["upi", "card", "wallet", "netbanking"]
    delivery_statuses = ["delivered", "shipped", "processing", "returned", "cancelled"]
    return_statuses = ["none", "requested", "approved", "rejected", "received"]
    chargeback_statuses = ["none", "disputed", "won", "lost"]
    
    data = []
    for i in range(n):
        merchant_id = np.random.choice(merchant_ids)
        customer_id = np.random.choice(customer_ids)
        device_fp = np.random.choice(device_ids)
        upi_handle = np.random.choice(upi_handles)
        
        # Transaction amount
        amount = int(np.random.lognormal(8, 0.7))
        
        # Timestamp within last 90 days
        txn_time = fake.date_time_between(start_date="-90d", end_date="now", tzinfo=timezone.utc)
        
        # Fraud label (rare)
        is_fraud = np.random.random() < 0.005
        
        # Chargeback if fraud (with some probability)
        if is_fraud:
            cb_status = np.random.choice(chargeback_statuses[1:], p=[0.6, 0.2, 0.2])
        else:
            cb_status = "none"
        
        data.append({
            "order_id": f"ORD{i:010d}",
            "order_amount_paise": amount,
            "order_category": np.random.choice(categories),
            "order_payment_method": np.random.choice(payment_methods, p=[0.5, 0.2, 0.2, 0.1]),
            "order_shipping_address_hash": hashlib.sha256(f"addr{i}".encode()).hexdigest()[:16],
            "order_billing_address_hash": hashlib.sha256(f"bill{i}".encode()).hexdigest()[:16],
            "order_device_fingerprint": device_fp,
            "order_customer_id": customer_id,
            "order_merchant_id": merchant_id,
            "order_timestamp": txn_time.timestamp(),
            "order_delivery_status": np.random.choice(delivery_statuses, p=[0.7, 0.1, 0.1, 0.05, 0.05]),
            "order_return_status": np.random.choice(return_statuses, p=[0.85, 0.05, 0.05, 0.03, 0.02]),
            "order_chargeback_status": cb_status,
            "is_fraud": is_fraud,
        })
    
    return pd.DataFrame(data)


def generate_voice_call_data(n: int) -> pd.DataFrame:
    """Generate voice call data for vishing detection."""
    codecs = ["G.711", "Opus", "AMR-WB", "EVS"]
    networks = ["wifi", "4g", "5g", "3g"]
    stir_shaken = ["A", "B", "C", "none"]
    urgency_kw = ["urgent", "immediate", "emergency", "asap", "critical"]
    secrecy_kw = ["confidential", "secret", "private", "don't tell"]
    authority_kw = ["police", "government", "bank", "ceo", "director", "manager"]
    
    data = []
    for i in range(n):
        call_sid = f"CALL{i:010d}"
        is_vishing = np.random.random() < 0.002
        
        # Vishing calls have characteristic patterns
        if is_vishing:
            urgency_count = np.random.randint(2, 5)
            secrecy_count = np.random.randint(1, 3)
            authority_count = np.random.randint(1, 3)
        else:
            urgency_count = np.random.randint(0, 1)
            secrecy_count = np.random.randint(0, 1)
            authority_count = np.random.randint(0, 1)
        
        data.append({
            "call_session_id": call_sid,
            "call_duration_seconds": int(np.random.exponential(180)),
            "call_caller_id": f"+91{np.random.randint(7000000000, 9999999999)}",
            "call_callee_id": f"+91{np.random.randint(7000000000, 9999999999)}",
            "call_direction": np.random.choice(["inbound", "outbound"]),
            "call_codec": np.random.choice(codecs),
            "call_network_type": np.random.choice(networks),
            "call_stir_shaken_attestation": np.random.choice(stir_shaken),
            "call_ani_validated": np.random.choice(["true", "false"], p=[0.8, 0.2]),
            "call_urgency_keywords": str(list(np.random.choice(urgency_kw, urgency_count))),
            "call_secrecy_keywords": str(list(np.random.choice(secrecy_kw, secrecy_count))),
            "call_authority_keywords": str(list(np.random.choice(authority_kw, authority_count))),
            "call_timestamp": fake.date_time_between(start_date="-30d", end_date="now", tzinfo=timezone.utc).timestamp(),
            "is_vishing": is_vishing,
        })
    
    return pd.DataFrame(data)


def generate_kyc_session_data(n: int) -> pd.DataFrame:
    """Generate KYC session data for liveness detection."""
    challenge_types = ["blink", "smile", "head_turn", "nod", "speak"]
    devices = ["mobile", "laptop", "tablet"]
    networks = ["wifi", "4g", "5g"]
    
    data = []
    for i in range(n):
        session_id = f"KYC{i:010d}"
        is_deepfake = np.random.random() < 0.003
        is_injection = np.random.random() < 0.002
        
        # Deepfake sessions have lower quality
        if is_deepfake:
            face_quality = np.random.beta(2, 5)
            liveness_score = np.random.beta(1, 5)
        elif is_injection:
            face_quality = np.random.beta(3, 3)
            liveness_score = np.random.beta(1, 3)
        else:
            face_quality = np.random.beta(8, 2)
            liveness_score = np.random.beta(9, 1)
        
        data.append({
            "session_id": session_id,
            "customer_id": f"C{np.random.randint(1, 10000):08d}",
            "device_type": np.random.choice(devices),
            "network_type": np.random.choice(networks),
            "challenge_type": np.random.choice(challenge_types),
            "face_quality_score": face_quality,
            "liveness_score": liveness_score,
            "deepfake_detected": is_deepfake,
            "injection_detected": is_injection,
            "document_forged": np.random.random() < 0.001,
            "session_timestamp": fake.date_time_between(start_date="-30d", end_date="now", tzinfo=timezone.utc).timestamp(),
        })
    
    return pd.DataFrame(data)


def generate_return_data(
    n: int,
    order_ids: List[str],
    customer_ids: List[str],
) -> pd.DataFrame:
    """Generate return request data with fraud labels."""
    reasons = ["damaged", "wrong_item", "not_as_described", "size_issue", "changed_mind", "defective"]
    return_statuses = ["requested", "approved", "rejected", "received", "inspected"]
    
    data = []
    for i in range(n):
        order_id = np.random.choice(order_ids)
        customer_id = np.random.choice(customer_ids)
        
        is_fraud = np.random.random() < 0.02  # 2% return fraud
        
        if is_fraud:
            # AI-generated damage photos
            image_score = np.random.beta(2, 5)  # Low forensic score = suspicious
            video_provided = np.random.random() < 0.3  # Less likely to provide video
        else:
            image_score = np.random.beta(8, 2)  # High forensic score = genuine
            video_provided = np.random.random() < 0.7
        
        data.append({
            "return_id": f"RET{i:010d}",
            "order_id": order_id,
            "customer_id": customer_id,
            "return_reason": np.random.choice(reasons),
            "damage_image_forensic_score": image_score,
            "video_provided": video_provided,
            "return_status": np.random.choice(return_statuses),
            "inspection_result": "fake" if is_fraud else "genuine",
            "return_timestamp": fake.date_time_between(start_date="-90d", end_date="now", tzinfo=timezone.utc).timestamp(),
            "is_return_fraud": is_fraud,
        })
    
    return pd.DataFrame(data)


def generate_review_data(
    n: int,
    product_ids: List[str],
    customer_ids: List[str],
) -> pd.DataFrame:
    """Generate review data with ring detection labels."""
    data = []
    
    # Create some coordinated rings
    n_rings = 5
    ring_size = 20
    ring_accounts = []
    for r in range(n_rings):
        ring_customers = [f"RING{r}_C{c:04d}" for c in range(ring_size)]
        ring_accounts.extend(ring_customers)
    
    for i in range(n):
        product_id = np.random.choice(product_ids)
        
        # Some reviews from ring accounts
        if np.random.random() < 0.05 and ring_accounts:
            customer_id = np.random.choice(ring_accounts)
            is_ring = True
        else:
            customer_id = np.random.choice(customer_ids)
            is_ring = False
        
        review_text = fake.paragraph(nb_sentences=3)
        
        # Ring reviews have similar patterns
        if is_ring:
            rating = np.random.choice([4, 5], p=[0.3, 0.7])
            perplexity = np.random.uniform(10, 30)  # Low perplexity = AI-generated
            burst_score = np.random.uniform(0.7, 1.0)
        else:
            rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.05, 0.1, 0.3, 0.5])
            perplexity = np.random.uniform(30, 100)
            burst_score = np.random.uniform(0, 0.3)
        
        data.append({
            "review_id": f"REV{i:010d}",
            "product_id": product_id,
            "customer_id": customer_id,
            "review_text": review_text,
            "rating": rating,
            "llm_perplexity": perplexity,
            "burst_score": burst_score,
            "is_purchase_verified": np.random.random() < 0.6,
            "review_timestamp": fake.date_time_between(start_date="-30d", end_date="now", tzinfo=timezone.utc).timestamp(),
            "is_ring_review": is_ring,
        })
    
    return pd.DataFrame(data)


def generate_all_data(
    output_dir: str,
    n_merchants: int = 100,
    n_customers: int = 1000,
    n_devices: int = 500,
    n_upi_handles: int = 2000,
    n_transactions: int = 50000,
    n_calls: int = 5000,
    n_kyc: int = 5000,
    n_returns: int = 2000,
    n_reviews: int = 10000,
    seed: int = 42,
):
    """Generate all synthetic datasets."""
    set_seeds(seed)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating synthetic data in {output_path}")
    
    # Generate core entities
    print("Generating merchants...")
    merchants = generate_merchant_data(n_merchants)
    merchants.to_parquet(output_path / "merchants.parquet", index=False)
    
    print("Generating customers...")
    customers = generate_customer_data(n_customers, merchants["merchant_id"].tolist())
    customers.to_parquet(output_path / "customers.parquet", index=False)
    
    print("Generating devices...")
    devices = generate_device_data(n_devices)
    devices.to_parquet(output_path / "devices.parquet", index=False)
    
    print("Generating UPI handles...")
    upi_handles = generate_upi_handle_data(n_upi_handles)
    upi_handles.to_parquet(output_path / "upi_handles.parquet", index=False)
    
    print("Generating transactions...")
    transactions = generate_transaction_data(
        n_transactions,
        merchants["merchant_id"].tolist(),
        customers["customer_id"].tolist(),
        devices["device_fingerprint"].tolist(),
        upi_handles["upi_handle"].tolist(),
    )
    transactions.to_parquet(output_path / "transactions.parquet", index=False)
    
    print("Generating voice calls...")
    calls = generate_voice_call_data(n_calls)
    calls.to_parquet(output_path / "voice_calls.parquet", index=False)
    
    print("Generating KYC sessions...")
    kyc = generate_kyc_session_data(n_kyc)
    kyc.to_parquet(output_path / "kyc_sessions.parquet", index=False)
    
    print("Generating returns...")
    returns = generate_return_data(
        n_returns,
        transactions["order_id"].tolist(),
        customers["customer_id"].tolist(),
    )
    returns.to_parquet(output_path / "returns.parquet", index=False)
    
    print("Generating reviews...")
    reviews = generate_review_data(
        n_reviews,
        [f"PROD{i:06d}" for i in range(1000)],
        customers["customer_id"].tolist(),
    )
    reviews.to_parquet(output_path / "reviews.parquet", index=False)
    
    # Generate feature store format
    print("Generating feature store files...")
    feature_dir = output_path / "feast" / "offline"
    feature_dir.mkdir(parents=True, exist_ok=True)
    
    # Add event timestamps
    for df, name in [
        (merchants, "merchant"),
        (customers, "customer"),
        (devices, "device"),
        (upi_handles, "upi_handle"),
    ]:
        df["event_timestamp"] = datetime.now(timezone.utc)
        df["created_timestamp"] = datetime.now(timezone.utc)
        df.to_parquet(feature_dir / f"{name}_features.parquet", index=False)
    
    # Transaction features for orders
    order_features = transactions.copy()
    order_features["event_timestamp"] = datetime.now(timezone.utc)
    order_features["created_timestamp"] = datetime.now(timezone.utc)
    order_features.to_parquet(feature_dir / "order_features.parquet", index=False)
    
    # Call session features
    call_features = calls.copy()
    call_features["event_timestamp"] = datetime.now(timezone.utc)
    call_features["created_timestamp"] = datetime.now(timezone.utc)
    call_features.to_parquet(feature_dir / "call_session_features.parquet", index=False)
    
    print("Done!")
    
    # Print summary
    print(f"\nData Summary:")
    print(f"  Merchants: {len(merchants)}")
    print(f"  Customers: {len(customers)}")
    print(f"  Devices: {len(devices)}")
    print(f"  UPI Handles: {len(upi_handles)}")
    print(f"  Transactions: {len(transactions)} (fraud: {transactions['is_fraud'].sum()})")
    print(f"  Voice Calls: {len(calls)} (vishing: {calls['is_vishing'].sum()})")
    print(f"  KYC Sessions: {len(kyc)} (deepfake: {kyc['deepfake_detected'].sum()}, injection: {kyc['injection_detected'].sum()})")
    print(f"  Returns: {len(returns)} (fraud: {returns['is_return_fraud'].sum()})")
    print(f"  Reviews: {len(reviews)} (ring: {reviews['is_ring_review'].sum()})")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic fraud data")
    parser.add_argument("--output-dir", default="data/synthetic", help="Output directory")
    parser.add_argument("--n-merchants", type=int, default=100)
    parser.add_argument("--n-customers", type=int, default=1000)
    parser.add_argument("--n-devices", type=int, default=500)
    parser.add_argument("--n-upi-handles", type=int, default=2000)
    parser.add_argument("--n-transactions", type=int, default=50000)
    parser.add_argument("--n-calls", type=int, default=5000)
    parser.add_argument("--n-kyc", type=int, default=5000)
    parser.add_argument("--n-returns", type=int, default=2000)
    parser.add_argument("--n-reviews", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    generate_all_data(
        output_dir=args.output_dir,
        n_merchants=args.n_merchants,
        n_customers=args.n_customers,
        n_devices=args.n_devices,
        n_upi_handles=args.n_upi_handles,
        n_transactions=args.n_transactions,
        n_calls=args.n_calls,
        n_kyc=args.n_kyc,
        n_returns=args.n_returns,
        n_reviews=args.n_reviews,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
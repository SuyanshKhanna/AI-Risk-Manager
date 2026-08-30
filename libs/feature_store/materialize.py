"""Feature store materialization jobs."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import logging
from feast import FeatureStore
from feast.repo_config import RepoConfig

from .definitions import get_all_entities, get_all_feature_views, get_all_sources

logger = logging.getLogger(__name__)


class FeatureMaterializer:
    """Handles feature materialization for online and offline stores."""
    
    def __init__(self, repo_path: str = "libs/feature_store"):
        self.repo_path = repo_path
        self.store = FeatureStore(repo_path=repo_path)
    
    def materialize_incremental(self, end_date: datetime = None) -> Dict[str, Any]:
        """Materialize features incrementally up to end_date."""
        end_date = end_date or datetime.utcnow()
        start_date = end_date - timedelta(days=1)
        
        logger.info(f"Materializing features from {start_date} to {end_date}")
        
        try:
            self.store.materialize_incremental(end_date=end_date)
            return {"status": "success", "end_date": end_date.isoformat()}
        except Exception as e:
            logger.error(f"Materialization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def materialize(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Materialize features for a date range."""
        logger.info(f"Materializing features from {start_date} to {end_date}")
        
        try:
            self.store.materialize(start_date=start_date, end_date=end_date)
            return {"status": "success", "start_date": start_date.isoformat(), "end_date": end_date.isoformat()}
        except Exception as e:
            logger.error(f"Materialization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_online_features(
        self,
        entity_rows: List[Dict[str, Any]],
        features: List[str],
    ) -> Dict[str, Any]:
        """Get online features for entity rows."""
        try:
            feature_vector = self.store.get_online_features(
                entity_rows=entity_rows,
                features=features,
            )
            return feature_vector.to_dict()
        except Exception as e:
            logger.error(f"Online feature retrieval failed: {e}")
            return {}
    
    def get_historical_features(
        self,
        entity_df: pd.DataFrame,
        features: List[str],
    ) -> pd.DataFrame:
        """Get historical features for training."""
        try:
            training_data = self.store.get_historical_features(
                entity_df=entity_df,
                features=features,
            )
            return training_data.to_df()
        except Exception as e:
            logger.error(f"Historical feature retrieval failed: {e}")
            return pd.DataFrame()


def generate_sample_features() -> Dict[str, pd.DataFrame]:
    """Generate sample feature data for testing."""
    np.random.seed(42)
    n_merchants = 100
    n_customers = 1000
    n_devices = 500
    n_upi_handles = 2000
    n_orders = 5000
    n_calls = 1000
    
    now = datetime.utcnow()
    
    # Merchant features
    merchants = pd.DataFrame({
        "merchant_id": [f"M{i:06d}" for i in range(n_merchants)],
        "merchant_category": np.random.choice(
            ["electronics", "fashion", "grocery", "food", "travel", "services"],
            n_merchants
        ),
        "merchant_gmv_30d": np.random.lognormal(12, 1.5, n_merchants).astype(int),
        "merchant_txn_count_30d": np.random.poisson(500, n_merchants),
        "merchant_chargeback_rate_30d": np.random.beta(1, 99, n_merchants),
        "merchant_refund_rate_30d": np.random.beta(2, 98, n_merchants),
        "merchant_avg_order_value": np.random.lognormal(8, 0.5, n_merchants).astype(int),
        "merchant_churn_risk_score": np.random.beta(2, 5, n_merchants),
        "merchant_onboarding_date": [
            (now - timedelta(days=np.random.randint(30, 1000))).timestamp()
            for _ in range(n_merchants)
        ],
        "merchant_risk_tier": np.random.choice(
            ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            n_merchants,
            p=[0.6, 0.25, 0.1, 0.05]
        ),
        "merchant_fraud_rate_30d": np.random.beta(1, 199, n_merchants),
        "merchant_dispute_win_rate_90d": np.random.beta(50, 10, n_merchants),
        "event_timestamp": [now] * n_merchants,
        "created_timestamp": [now] * n_merchants,
    })
    
    # Customer features
    customers = pd.DataFrame({
        "customer_id": [f"C{i:08d}" for i in range(n_customers)],
        "customer_txn_count_30d": np.random.poisson(10, n_customers),
        "customer_avg_txn_amount": np.random.lognormal(7, 0.8, n_customers).astype(int),
        "customer_return_rate_90d": np.random.beta(1, 19, n_customers),
        "customer_chargeback_count_90d": np.random.poisson(0.1, n_customers),
        "customer_device_count_30d": np.random.poisson(2, n_customers) + 1,
        "customer_upi_handle_count_30d": np.random.poisson(1.5, n_customers) + 1,
        "customer_first_txn_date": [
            (now - timedelta(days=np.random.randint(1, 1000))).timestamp()
            for _ in range(n_customers)
        ],
        "customer_risk_tier": np.random.choice(
            ["LOW", "MEDIUM", "HIGH"],
            n_customers,
            p=[0.7, 0.2, 0.1]
        ),
        "customer_lifetime_value": np.random.lognormal(9, 1, n_customers).astype(int),
        "customer_preferred_categories": [
            str(list(np.random.choice(
                ["electronics", "fashion", "grocery", "food", "travel"],
                np.random.randint(1, 4), replace=False
            ))) for _ in range(n_customers)
        ],
        "customer_preferred_payment_methods": [
            str(list(np.random.choice(
                ["upi", "card", "wallet", "netbanking"],
                np.random.randint(1, 3), replace=False
            ))) for _ in range(n_customers)
        ],
        "event_timestamp": [now] * n_customers,
        "created_timestamp": [now] * n_customers,
    })
    
    # Device features
    devices = pd.DataFrame({
        "device_fingerprint": [f"dev_{i:010d}" for i in range(n_devices)],
        "device_os": np.random.choice(["android", "ios", "web"], n_devices, p=[0.7, 0.2, 0.1]),
        "device_model": np.random.choice(
            ["pixel", "samsung", "xiaomi", "iphone", "oneplus", "other"],
            n_devices
        ),
        "device_sensor_entropy": np.random.beta(5, 2, n_devices),
        "device_app_tamper_score": np.random.beta(1, 10, n_devices),
        "device_emulator_score": np.random.beta(1, 50, n_devices),
        "device_root_score": np.random.beta(1, 30, n_devices),
        "device_vpn_probability": np.random.beta(1, 20, n_devices),
        "device_first_seen": [
            (now - timedelta(days=np.random.randint(1, 500))).timestamp()
            for _ in range(n_devices)
        ],
        "device_merchant_count_30d": np.random.poisson(5, n_devices),
        "device_customer_count_30d": np.random.poisson(2, n_devices) + 1,
        "device_upi_handle_count_30d": np.random.poisson(3, n_devices),
        "device_fraud_reports_30d": np.random.poisson(0.05, n_devices),
        "event_timestamp": [now] * n_devices,
        "created_timestamp": [now] * n_devices,
    })
    
    # UPI handle features
    upi_handles = pd.DataFrame({
        "upi_handle": [f"user{i}@upi" for i in range(n_upi_handles)],
        "upi_txn_count_5min": np.random.poisson(0.1, n_upi_handles),
        "upi_txn_count_1hr": np.random.poisson(2, n_upi_handles),
        "upi_refund_count_1hr": np.random.poisson(0.05, n_upi_handles),
        "upi_amount_zscore_1hr": np.random.normal(0, 1, n_upi_handles),
        "upi_unique_merchants_1hr": np.random.poisson(1, n_upi_handles),
        "upi_unique_devices_1hr": np.random.poisson(1, n_upi_handles),
        "upi_first_seen": [
            (now - timedelta(days=np.random.randint(1, 365))).timestamp()
            for _ in range(n_upi_handles)
        ],
        "upi_chargeback_count_24hr": np.random.poisson(0.01, n_upi_handles),
        "upi_dispute_rate_24hr": np.random.beta(1, 200, n_upi_handles),
        "event_timestamp": [now] * n_upi_handles,
        "created_timestamp": [now] * n_upi_handles,
    })
    
    # Order features
    orders = pd.DataFrame({
        "order_id": [f"ORD{i:010d}" for i in range(n_orders)],
        "order_amount_paise": np.random.lognormal(8, 0.7, n_orders).astype(int),
        "order_category": np.random.choice(
            ["electronics", "fashion", "grocery", "food", "travel", "services"],
            n_orders
        ),
        "order_payment_method": np.random.choice(
            ["upi", "card", "wallet", "netbanking"],
            n_orders,
            p=[0.5, 0.2, 0.2, 0.1]
        ),
        "order_shipping_address_hash": [hashlib.sha256(f"addr{i}".encode()).hexdigest()[:16] for i in range(n_orders)],
        "order_billing_address_hash": [hashlib.sha256(f"bill{i}".encode()).hexdigest()[:16] for i in range(n_orders)],
        "order_device_fingerprint": np.random.choice(devices["device_fingerprint"], n_orders),
        "order_customer_id": np.random.choice(customers["customer_id"], n_orders),
        "order_merchant_id": np.random.choice(merchants["merchant_id"], n_orders),
        "order_timestamp": [
            (now - timedelta(days=np.random.randint(0, 90))).timestamp()
            for _ in range(n_orders)
        ],
        "order_delivery_status": np.random.choice(
            ["delivered", "shipped", "processing", "returned", "cancelled"],
            n_orders,
            p=[0.7, 0.1, 0.1, 0.05, 0.05]
        ),
        "order_return_status": np.random.choice(
            ["none", "requested", "approved", "rejected", "received"],
            n_orders,
            p=[0.85, 0.05, 0.05, 0.03, 0.02]
        ),
        "order_chargeback_status": np.random.choice(
            ["none", "disputed", "won", "lost"],
            n_orders,
            p=[0.95, 0.03, 0.01, 0.01]
        ),
        "event_timestamp": [now] * n_orders,
        "created_timestamp": [now] * n_orders,
    })
    
    # Call session features
    calls = pd.DataFrame({
        "call_session_id": [f"CALL{i:010d}" for i in range(n_calls)],
        "call_duration_seconds": np.random.exponential(180, n_calls).astype(int),
        "call_caller_id": [f"+91{np.random.randint(7000000000, 9999999999)}" for _ in range(n_calls)],
        "call_callee_id": [f"+91{np.random.randint(7000000000, 9999999999)}" for _ in range(n_calls)],
        "call_direction": np.random.choice(["inbound", "outbound"], n_calls),
        "call_codec": np.random.choice(["G.711", "Opus", "AMR-WB", "EVS"], n_calls),
        "call_network_type": np.random.choice(["wifi", "4g", "5g", "3g"], n_calls),
        "call_stir_shaken_attestation": np.random.choice(["A", "B", "C", "none"], n_calls),
        "call_ani_validated": np.random.choice(["true", "false"], n_calls, p=[0.8, 0.2]),
        "call_urgency_keywords": [
            str(list(np.random.choice(
                ["urgent", "immediate", "emergency", "asap", "critical", ""],
                np.random.randint(0, 3)
            ))) for _ in range(n_calls)
        ],
        "call_secrecy_keywords": [
            str(list(np.random.choice(
                ["confidential", "secret", "private", "don't tell", ""],
                np.random.randint(0, 2)
            ))) for _ in range(n_calls)
        ],
        "call_authority_keywords": [
            str(list(np.random.choice(
                ["police", "government", "bank", "ceo", "director", "manager", ""],
                np.random.randint(0, 2)
            ))) for _ in range(n_calls)
        ],
        "call_timestamp": [
            (now - timedelta(hours=np.random.randint(0, 720))).timestamp()
            for _ in range(n_calls)
        ],
        "event_timestamp": [now] * n_calls,
        "created_timestamp": [now] * n_calls,
    })
    
    return {
        "merchant": merchants,
        "customer": customers,
        "device": devices,
        "upi_handle": upi_handles,
        "order": orders,
        "call_session": calls,
    }


def write_offline_features(
    features: Dict[str, pd.DataFrame],
    output_dir: str = "data/feast/offline"
):
    """Write feature DataFrames to parquet files."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    for name, df in features.items():
        path = f"{output_dir}/{name}_features.parquet"
        df.to_parquet(path, index=False)
        logger.info(f"Wrote {len(df)} rows to {path}")


def load_features_to_online_store(
    features: Dict[str, pd.DataFrame],
    redis_host: str = "localhost",
    redis_port: int = 6379,
):
    """Load features to Redis online store."""
    import redis
    import json
    
    r = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)
    
    for name, df in features.items():
        table_name = f"{name}_features"
        for _, row in df.iterrows():
            # Get entity key
            entity_col = f"{name}_id" if name != "upi_handle" else "upi_handle"
            if entity_col not in df.columns:
                entity_col = f"{name}_fingerprint" if name == "device" else f"{name}_id"
            entity_key = row[entity_col]
            
            # Prepare feature data (exclude metadata columns)
            feature_data = {
                k: v for k, v in row.items()
                if k not in ["event_timestamp", "created_timestamp", entity_col]
            }
            
            # Store as hash
            r.hset(table_name, entity_key, json.dumps(feature_data, default=str))
        
        logger.info(f"Loaded {len(df)} rows to Redis table {table_name}")


if __name__ == "__main__":
    import hashlib
    
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample features
    features = generate_sample_features()
    
    # Write to offline store
    write_offline_features(features)
    
    # Optionally load to online store
    # load_features_to_online_store(features)
    
    print("Sample features generated successfully!")
"""
Locust load test for UPI Fraud Detector API.
Target: 100 RPS sustained load.

Run with:
    locust -f tests/load/locustfile.py --host=http://localhost:8000 --users=50 --spawn-rate=10 --run-time=60s --headless
"""

import random
from locust import HttpUser, task, between, constant


class FraudDetectorUser(HttpUser):
    """Simulates a client sending transactions for fraud scoring."""
    
    wait_time = constant(0.01)  # ~100 RPS per user at 100 users
    
    # Transaction templates
    LEGITIMATE_TXN = {
        "amount_paise": 25000,
        "merchant_id": "M_trusted_123",
        "customer_id": "C_regular_456",
        "device_fingerprint": "dev_known_789",
        "upi_handle": "user@trusted",
        "velocity_5min": 1,
        "velocity_1hr": 3,
        "refund_count_1hr": 0,
        "refund_rate": 0.0,
        "same_device_refunds": 0,
        "settlement_verified": 1,
        "qr_mismatch": 0,
        "remote_access_app": 0,
        "device_emulator_score": 0.01,
        "device_root_score": 0.01,
        "device_fraud_reports_30d": 0,
        "vpn_probability": 0.01,
        "hour": 14,
        "is_night": 0,
        "new_upi_handle": 0,
        "new_merchant": 0,
        "new_device": 0,
        "unusual_hour": 0,
        "session_duration_sec": 45,
        "merchant_id_txn_count_5min": 0,
        "device_fingerprint_txn_count_5min": 0,
        "upi_handle_txn_count_5min": 0,
    }
    
    FRAUD_TXN = {
        "amount_paise": 45000,
        "merchant_id": "M_new999",
        "customer_id": "C_fake111",
        "device_fingerprint": "dev_emulator1",
        "upi_handle": "upi_throwaway999",
        "velocity_5min": 12,
        "velocity_1hr": 25,
        "refund_count_1hr": 3,
        "refund_rate": 0.25,
        "same_device_refunds": 2,
        "settlement_verified": 0,
        "qr_mismatch": 1,
        "remote_access_app": 1,
        "device_emulator_score": 0.85,
        "device_root_score": 0.7,
        "device_fraud_reports_30d": 5,
        "vpn_probability": 0.7,
        "hour": 23,
        "is_night": 1,
        "new_upi_handle": 1,
        "new_merchant": 1,
        "new_device": 1,
        "unusual_hour": 1,
        "session_duration_sec": 5,
        "merchant_id_txn_count_5min": 0,
        "device_fingerprint_txn_count_5min": 0,
        "upi_handle_txn_count_5min": 0,
    }
    
    CHALLENGE_TXN = {
        "amount_paise": 35000,
        "merchant_id": "M_somewhat_new",
        "customer_id": "C_known",
        "device_fingerprint": "dev_known",
        "upi_handle": "user@bank",
        "velocity_5min": 3,
        "velocity_1hr": 8,
        "refund_count_1hr": 0,
        "refund_rate": 0.0,
        "same_device_refunds": 0,
        "settlement_verified": 1,
        "qr_mismatch": 0,
        "remote_access_app": 0,
        "device_emulator_score": 0.3,
        "device_root_score": 0.1,
        "device_fraud_reports_30d": 0,
        "vpn_probability": 0.2,
        "hour": 20,
        "is_night": 0,
        "new_upi_handle": 0,
        "new_merchant": 1,
        "new_device": 0,
        "unusual_hour": 0,
        "session_duration_sec": 30,
        "merchant_id_txn_count_5min": 0,
        "device_fingerprint_txn_count_5min": 0,
        "upi_handle_txn_count_5min": 0,
    }
    
    def on_start(self):
        """Verify service is healthy before starting."""
        response = self.client.get("/health")
        if response.status_code != 200:
            raise Exception(f"Service unhealthy: {response.status_code}")
    
    @task(70)
    def score_legitimate(self):
        """Score a legitimate transaction (70% of traffic)."""
        txn = self.LEGITIMATE_TXN.copy()
        txn["txn_id"] = f"LOAD_LEGIT_{random.randint(1, 1000000)}"
        
        with self.client.post("/score", json=txn, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("action") not in ["ALLOW", "CHALLENGE", "BLOCK"]:
                    response.failure(f"Unexpected action: {data.get('action')}")
            else:
                response.failure(f"Status {response.status_code}: {response.text}")
    
    @task(20)
    def score_fraud(self):
        """Score a fraud transaction (20% of traffic)."""
        txn = self.FRAUD_TXN.copy()
        txn["txn_id"] = f"LOAD_FRAUD_{random.randint(1, 1000000)}"
        
        with self.client.post("/score", json=txn, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("action") not in ["CHALLENGE", "BLOCK"]:
                    response.failure(f"Fraud not detected: {data.get('action')}")
            else:
                response.failure(f"Status {response.status_code}: {response.text}")
    
    @task(10)
    def score_challenge(self):
        """Score a challenge-worthy transaction (10% of traffic)."""
        txn = self.CHALLENGE_TXN.copy()
        txn["txn_id"] = f"LOAD_CHAL_{random.randint(1, 1000000)}"
        
        with self.client.post("/score", json=txn, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("action") not in ["ALLOW", "CHALLENGE", "BLOCK"]:
                    response.failure(f"Unexpected action: {data.get('action')}")
            else:
                response.failure(f"Status {response.status_code}: {response.text}")
    
    @task(5)
    def batch_score(self):
        """Test batch scoring endpoint (5% of traffic)."""
        batch_size = random.randint(5, 20)
        transactions = []
        
        for i in range(batch_size):
            txn_type = random.choices(
                ["legit", "fraud", "challenge"],
                weights=[70, 20, 10]
            )[0]
            
            if txn_type == "legit":
                base = self.LEGITIMATE_TXN
            elif txn_type == "fraud":
                base = self.FRAUD_TXN
            else:
                base = self.CHALLENGE_TXN
            
            txn = base.copy()
            txn["txn_id"] = f"LOAD_BATCH_{random.randint(1, 1000000)}_{i}"
            transactions.append(txn)
        
        with self.client.post("/batch_score", json={"transactions": transactions}, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                if len(results) != batch_size:
                    response.failure(f"Batch size mismatch: expected {batch_size}, got {len(results)}")
            else:
                response.failure(f"Status {response.status_code}: {response.text}")
    
    @task(1)
    def health_check(self):
        """Periodic health check (1% of traffic)."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(1)
    def metrics_check(self):
        """Check Prometheus metrics endpoint (1% of traffic)."""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Metrics endpoint failed: {response.status_code}")


class SpikeUser(FraudDetectorUser):
    """User that sends burst traffic to test spike handling."""
    
    wait_time = between(0.001, 0.01)  # Very fast bursts
    
    @task
    def burst_score(self):
        """Send rapid requests."""
        txn = self.LEGITIMATE_TXN.copy()
        txn["txn_id"] = f"SPIKE_{random.randint(1, 1000000)}"
        self.client.post("/score", json=txn)


# Custom load shape for ramp-up testing
from locust import LoadTestShape


class StepLoadShape(LoadTestShape):
    """
    Step load pattern:
    - 0-60s: 10 users (10 RPS)
    - 60-120s: 50 users (50 RPS)
    - 120-180s: 100 users (100 RPS)
    - 180-240s: 50 users (50 RPS)
    - 240-300s: 10 users (10 RPS)
    """
    
    stages = [
        {"duration": 60, "users": 10, "spawn_rate": 2},
        {"duration": 120, "users": 50, "spawn_rate": 5},
        {"duration": 180, "users": 100, "spawn_rate": 10},
        {"duration": 240, "users": 50, "spawn_rate": 10},
        {"duration": 300, "users": 10, "spawn_rate": 10},
    ]
    
    def tick(self):
        run_time = self.get_run_time()
        
        for stage in self.stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["spawn_rate"])
        
        return None


class ConstantLoadShape(LoadTestShape):
    """Constant 100 RPS for 5 minutes."""
    
    def tick(self):
        run_time = self.get_run_time()
        if run_time < 300:  # 5 minutes
            return (100, 20)  # 100 users, spawn 20/sec
        return None
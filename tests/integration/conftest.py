"""
Pytest configuration for integration tests.
"""

import pytest
import pytest_asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running"
    )


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def client():
    """Create async HTTP client for testing with model pre-loaded."""
    from services.upi_fraud_detector.serve import app
    from services.upi_fraud_detector.detector import UpiFraudDetector
    from httpx import ASGITransport, AsyncClient

    # Pre-load model
    model_path = Path("models/upi_fraud_lgbm")
    if model_path.exists():
        app.state.detector = UpiFraudDetector.load(str(model_path))
        print(f"Loaded model for testing: {app.state.detector.model_version}")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# Fixtures for test transactions
@pytest.fixture
def legitimate_transaction() -> dict:
    """A transaction that should be ALLOWED (low risk)."""
    return {
        "txn_id": "INT_LEGIT_001",
        "amount_paise": 50000,
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


@pytest.fixture
def fraud_transaction() -> dict:
    """A transaction that should be BLOCKED (high risk)."""
    return {
        "txn_id": "INT_FRAUD_001",
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


@pytest.fixture
def challenge_transaction() -> dict:
    """A transaction that should be CHALLENGED (medium risk)."""
    return {
        "txn_id": "INT_CHALLENGE_001",
        "amount_paise": 25000,
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


@pytest.fixture
def mixed_batch(legitimate_transaction, fraud_transaction, challenge_transaction) -> list:
    """A batch with mixed transaction types."""
    return [
        legitimate_transaction,
        fraud_transaction,
        challenge_transaction,
        {**legitimate_transaction, "txn_id": "INT_LEGIT_002"},
        {**fraud_transaction, "txn_id": "INT_FRAUD_002"},
    ]
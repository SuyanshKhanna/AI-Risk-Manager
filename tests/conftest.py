"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta

from libs.common.schemas import (
    Action, FraudVector, UpiTxnRequest, UpiTxnResponse,
    VoiceAuthRequest, KycLivenessRequest, FeedbackLabel,
)


@pytest.fixture
def sample_upi_request():
    """Sample UPI transaction request."""
    return UpiTxnRequest(
        request_id="test_req_123",
        merchant_id="M123456",
        txn_id="TXN789012",
        amount_paise=50000,
        upi_handle="customer@upi",
        device_fingerprint="dev_abc123",
        qr_image_b64="base64_qr_image",
        screenshot_b64="base64_screenshot",
        screenshot_ts_ms=int(datetime.now(timezone.utc).timestamp() * 1000),
    )


@pytest.fixture
def sample_voice_request():
    """Sample voice authentication request."""
    return VoiceAuthRequest(
        request_id="test_voice_123",
        merchant_id="M123456",
        call_session_id="CALL_001",
        audio_b64="base64_audio_data",
        caller_id="+919876543210",
        enrolled_voiceprint_id="vp_123",
    )


@pytest.fixture
def sample_kyc_request():
    """Sample KYC liveness request."""
    return KycLivenessRequest(
        request_id="test_kyc_123",
        merchant_id="M123456",
        session_id="KYC_SESS_001",
        video_b64="base64_video_data",
        document_image_b64="base64_doc_image",
        challenge_type="blink",
    )


@pytest.fixture
def sample_feedback_label():
    """Sample feedback label."""
    return FeedbackLabel(
        request_id="req_123",
        vector=FraudVector.UPI,
        true_label=1,
        label_source="chargeback",
        confidence=0.95,
    )


@pytest.fixture
def synthetic_fraud_data():
    """Generate synthetic fraud detection dataset."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Create fraud signal from first few features
    fraud_signal = (
        2.0 * (X[:, 0] > 1.5) +
        1.5 * (X[:, 1] < -1.0) +
        1.0 * (X[:, 2] > 2.0) +
        0.5 * np.random.randn(n_samples)
    )
    
    fraud_prob = 1 / (1 + np.exp(-fraud_signal))
    fraud_prob = np.clip(fraud_prob, 0.001, 0.3)
    y = np.random.binomial(1, fraud_prob)
    
    # Gates
    R = np.ones(n_samples, dtype=int)
    O = np.where(y == 1, np.random.binomial(1, 0.7, n_samples), 0)
    D = np.where(O == 1, np.random.binomial(1, 0.8, n_samples), 0)
    groups = np.random.randint(1, 10, n_samples)
    
    # Observed labels
    Y_obs = np.zeros(n_samples)
    labeled_mask = (R == 1) & (O == 1) & (D == 1)
    Y_obs[labeled_mask] = y[labeled_mask]
    
    # Add noise
    noise_mask = labeled_mask & (np.random.random(n_samples) < 0.1)
    Y_obs[noise_mask] = 1 - Y_obs[noise_mask]
    
    return {
        "X": pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])]),
        "y": pd.Series(y, name="label"),
        "Y_obs": Y_obs,
        "R": R,
        "O": O,
        "D": D,
        "groups": groups,
        "labeled_mask": labeled_mask,
    }


@pytest.fixture
def synthetic_sequence_data():
    """Generate synthetic sequence data for TFT."""
    np.random.seed(42)
    n_samples = 500
    seq_len = 30
    input_dim = 100
    
    sequences = np.random.randn(n_samples, seq_len, input_dim).astype(np.float32)
    static = np.random.randn(n_samples, input_dim).astype(np.float32)
    y = np.random.randint(0, 2, n_samples)
    
    return {
        "sequences": sequences,
        "static": static,
        "labels": y,
    }


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton state between tests."""
    yield
    # Cleanup if needed


@pytest.fixture
def mock_feature_store():
    """Mock feature store for testing."""
    from unittest.mock import MagicMock
    
    mock = MagicMock()
    mock.get_online_features.return_value = {
        "merchant:merchant_gmv_30d": [10000000],
        "merchant:merchant_chargeback_rate_30d": [0.01],
        "device:device_fraud_reports_30d": [0],
    }
    return mock


# Pytest markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "adversarial: Adversarial robustness tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")
"""Unit tests for UPI Fraud Detector (LightGBM MVP)."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.upi_fraud_detector.detector import (
    UpiFraudDetector, ModelConfig, load_training_data, DecisionEngine
)


class TestModelConfig:
    """Test ModelConfig defaults."""
    
    def test_default_config(self):
        config = ModelConfig()
        assert config.test_size == 0.2
        assert config.val_size == 0.1
        assert config.calibrate is True
        assert config.challenge_threshold == 0.3
        assert config.block_threshold == 0.7
        assert config.fp_cost_per_blocked_txn == 150.0
        assert config.lgb_params is not None
        assert config.lgb_params["objective"] == "binary"


class TestUpiFraudDetector:
    """Test UPI Fraud Detector training and prediction."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Create small synthetic dataset for testing."""
        np.random.seed(42)
        n = 1000
        fraud_ratio = 0.02
        n_fraud = int(n * fraud_ratio)
        
        # Legitimate transactions
        n_legit = n - n_fraud
        legit_data = {
            "amount_paise": np.random.lognormal(8.5, 0.7, n_legit).astype(int),
            "velocity_5min": np.random.poisson(0.5, n_legit),
            "velocity_1hr": np.random.poisson(3, n_legit),
            "refund_count_1hr": np.random.poisson(0.02, n_legit),
            "refund_rate": np.random.beta(1, 50, n_legit),
            "same_device_refunds": np.zeros(n_legit, dtype=int),
            "settlement_verified": np.ones(n_legit, dtype=int),
            "qr_mismatch": np.zeros(n_legit, dtype=int),
            "remote_access_app": np.zeros(n_legit, dtype=int),
            "device_emulator_score": np.random.beta(1, 50, n_legit),
            "device_root_score": np.random.beta(1, 30, n_legit),
            "device_fraud_reports_30d": np.random.poisson(0.01, n_legit),
            "vpn_probability": np.random.beta(1, 20, n_legit),
            "hour": np.random.randint(6, 22, n_legit),
            "is_night": np.zeros(n_legit, dtype=int),
            "new_upi_handle": np.zeros(n_legit, dtype=int),
            "new_merchant": np.zeros(n_legit, dtype=int),
            "new_device": np.zeros(n_legit, dtype=int),
            "unusual_hour": np.zeros(n_legit, dtype=int),
            "session_duration_sec": np.zeros(n_legit, dtype=int),
            # Rolling features
            "merchant_id_txn_count_5min": np.random.poisson(10, n_legit),
            "merchant_id_txn_count_1hr": np.random.poisson(50, n_legit),
            "device_fingerprint_txn_count_5min": np.random.poisson(0.5, n_legit),
            "device_fingerprint_txn_count_1hr": np.random.poisson(2, n_legit),
            "upi_handle_txn_count_5min": np.random.poisson(0.2, n_legit),
            "upi_handle_txn_count_1hr": np.random.poisson(1, n_legit),
        }
        
        # Fraud transactions (fake screenshot pattern)
        fraud_data = {}
        for k, v in legit_data.items():
            if k in ["velocity_5min", "velocity_1hr", "refund_count_1hr"]:
                fraud_data[k] = np.random.poisson(v.mean() * 10, n_fraud).astype(v.dtype)
            elif k in ["settlement_verified"]:
                fraud_data[k] = np.zeros(n_fraud, dtype=v.dtype)
            elif k in ["device_emulator_score", "vpn_probability"]:
                fraud_data[k] = np.random.beta(5, 2, n_fraud).astype(v.dtype)
            elif k in ["is_night", "new_upi_handle", "new_merchant", "new_device"]:
                fraud_data[k] = np.ones(n_fraud, dtype=v.dtype)
            else:
                fraud_data[k] = v[:n_fraud].copy()
        
        # Combine
        combined = {}
        for k in legit_data:
            combined[k] = np.concatenate([legit_data[k], fraud_data[k]])
        
        df = pd.DataFrame(combined)
        y = pd.Series(np.concatenate([np.zeros(n_legit), np.ones(n_fraud)]), name="label")
        
        # Shuffle
        idx = np.random.permutation(n)
        df = df.iloc[idx].reset_index(drop=True)
        y = y.iloc[idx].reset_index(drop=True)
        
        return df, y
    
    def test_detector_init(self):
        """Test detector initialization."""
        config = ModelConfig()
        detector = UpiFraudDetector(config)
        assert detector.config == config
        assert detector.model is None
        assert detector.explainer is None
    
    def test_train_and_predict(self, synthetic_data):
        """Test full train -> predict cycle."""
        X, y = synthetic_data
        
        config = ModelConfig(
            test_size=0.3,
            val_size=0.1,
            calibrate=True,
            challenge_threshold=0.3,
            block_threshold=0.7,
        )
        
        detector = UpiFraudDetector(config)
        metrics = detector.train(X, y)
        
        # Check metrics exist
        assert "pr_auc" in metrics
        assert "roc_auc" in metrics
        assert "challenge_precision" in metrics
        assert "block_recall" in metrics
        
        # Check model is trained
        assert detector.model is not None
        assert detector.explainer is not None
        assert detector.decision_engine is not None
        assert detector.feature_columns is not None
        
        # Test prediction
        decisions = detector.predict(X.head(10))
        assert len(decisions) == 10
        for d in decisions:
            assert "score" in d
            assert "action" in d
            assert "reasons" in d
            assert "shap_top3" in d
            assert d["action"] in ["ALLOW", "CHALLENGE", "BLOCK"]
            assert 0 <= d["score"] <= 1
            assert isinstance(d["reasons"], list)
            assert len(d["reasons"]) > 0
    
    def test_save_load(self, synthetic_data, tmp_path):
        """Test model save/load roundtrip."""
        X, y = synthetic_data
        
        config = ModelConfig(test_size=0.3, val_size=0.1, calibrate=True)
        detector = UpiFraudDetector(config)
        detector.train(X, y)
        
        # Save
        model_dir = tmp_path / "test_model"
        detector.save(str(model_dir))
        
        # Check files exist
        assert (model_dir / "model.txt").exists()
        assert (model_dir / "config.json").exists()
        assert (model_dir / "calibrated_model.pkl").exists()
        
        # Load
        loaded = UpiFraudDetector.load(str(model_dir))
        
        assert loaded.model_version == detector.model_version
        assert loaded.feature_columns == detector.feature_columns
        assert loaded.config.challenge_threshold == detector.config.challenge_threshold
        
        # Predict with loaded model
        decisions = loaded.predict(X.head(5))
        assert len(decisions) == 5
        for d in decisions:
            assert "score" in d
            assert d["action"] in ["ALLOW", "CHALLENGE", "BLOCK"]


class TestDecisionEngine:
    """Test DecisionEngine rule layer."""
    
    def test_rule_thresholds(self):
        """Test that CHALLENGE/BLOCK thresholds work with mocked model."""
        from unittest.mock import MagicMock
        import pandas as pd
        import numpy as np
        
        # Create a mock model that returns predictable scores
        mock_model = MagicMock()
        # For high risk: score 0.9 -> BLOCK
        # For low risk: score 0.1 -> ALLOW
        mock_model.predict_proba = MagicMock(side_effect=[
            np.array([[0.1, 0.9]]),  # high risk
            np.array([[0.9, 0.1]]),  # low risk
        ])
        
        config = ModelConfig(
            challenge_threshold=0.3,
            block_threshold=0.7,
            calibrate=False,
        )
        
        detector = UpiFraudDetector(config)
        detector.model = MagicMock()
        detector.model.best_iteration = 100
        detector.calibrated_model = mock_model
        detector.feature_columns = ["feat_0", "feat_1"]
        detector.explainer = MagicMock()
        detector.explainer.__call__ = MagicMock(return_value=MagicMock(
            values=np.array([[0.1, -0.2]]),
        ))
        
        detector.decision_engine = DecisionEngine(config, mock_model, detector.explainer)
        
        # Test high-risk sample (score 0.9 -> BLOCK)
        high_risk = pd.DataFrame({"feat_0": [1], "feat_1": [1]})
        decisions = detector.predict(high_risk)
        d = decisions[0]
        assert d["action"] == "BLOCK"
        assert any("High fraud risk" in r for r in d["reasons"])
        
        # Test low-risk sample (score 0.1 -> ALLOW)
        mock_model.predict_proba = MagicMock(return_value=np.array([[0.9, 0.1]]))
        low_risk = pd.DataFrame({"feat_0": [0], "feat_1": [0]})
        decisions = detector.predict(low_risk)
        d = decisions[0]
        assert d["action"] == "ALLOW"
        assert any("Low fraud risk" in r for r in d["reasons"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
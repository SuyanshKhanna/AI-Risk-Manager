"""Tests for STR Estimator."""


import numpy as np
import pytest

from libs.str_estimator import (
    LightGBMOutcome,
    LightGBMPropensity,
    STRClassifier,
    STRConfig,
    STREstimator,
)


class TestSTRConfig:
    """Test STR configuration."""

    def test_default_config(self):
        config = STRConfig()
        assert config.propensity_model == "lightgbm"
        assert config.outcome_model == "lightgbm"
        assert config.corruption_model == "beta_binomial"
        assert config.shrinkage_enabled is True
        assert config.n_folds == 5


class TestLightGBMPropensity:
    """Test LightGBM propensity model."""

    def test_fit_predict(self):
        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, 100)

        model = LightGBMPropensity()
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert probs.shape == (100, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)
        assert np.all(probs >= 0) and np.all(probs <= 1)


class TestLightGBMOutcome:
    """Test LightGBM outcome model."""

    def test_fit_predict(self):
        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float32)
        y = np.random.randint(0, 2, 100)

        model = LightGBMOutcome()
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (100,)
        assert np.all(np.isin(preds, [0, 1]))

        probs = model.predict_proba(X)
        assert probs.shape == (100, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)


class TestSTREstimator:
    """Test STR Estimator."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 1000
        self.n_features = 20

        # Generate synthetic data with known fraud patterns
        self.X = np.random.randn(self.n_samples, self.n_features).astype(np.float32)

        # True fraud signal
        fraud_signal = (
            1.5 * (self.X[:, 0] > 1.0) +
            1.0 * (self.X[:, 1] < -0.5) +
            0.5 * np.random.randn(self.n_samples)
        )
        true_fraud_prob = 1 / (1 + np.exp(-fraud_signal))
        true_fraud_prob = np.clip(true_fraud_prob, 0.001, 0.3)
        self.y_true = np.random.binomial(1, true_fraud_prob)

        # Gates
        self.R = np.ones(self.n_samples, dtype=int)  # All authorized
        self.O = np.where(self.y_true == 1, np.random.binomial(1, 0.7, self.n_samples), 0)
        self.D = np.where(self.O == 1, np.random.binomial(1, 0.8, self.n_samples), 0)
        self.groups = np.random.randint(1, 10, self.n_samples)

        # Observed labels (only where R=O=D=1)
        self.Y_obs = np.zeros(self.n_samples)
        labeled_mask = (self.R == 1) & (self.O == 1) & (self.D == 1)
        self.Y_obs[labeled_mask] = self.y_true[labeled_mask]

        # Add some label noise (corruption)
        noise_mask = labeled_mask & (np.random.random(self.n_samples) < 0.1)
        self.Y_obs[noise_mask] = 1 - self.Y_obs[noise_mask]

    def test_fit_basic(self):
        """Test basic STR fitting."""
        config = STRConfig(
            propensity_model="lightgbm",
            outcome_model="lightgbm",
            corruption_model="beta_binomial",
            shrinkage_enabled=False,  # Disable for simpler test
        )

        estimator = STREstimator(config)
        estimator.fit(self.X, self.Y_obs, self.R, self.O, self.D, self.groups)

        assert estimator.is_fitted
        assert hasattr(estimator, 'pseudo_labels')
        assert hasattr(estimator, 'pi_combined')
        assert hasattr(estimator, 'corruption_rates')

        # Check pseudo-labels are in valid range
        assert np.all(estimator.pseudo_labels >= 0)
        assert np.all(estimator.pseudo_labels <= 1)

        # Check propensities are in valid range
        assert np.all(estimator.pi_combined > 0)
        assert np.all(estimator.pi_combined <= 1)

    def test_fit_with_shrinkage(self):
        """Test STR fitting with empirical Bayes shrinkage."""
        config = STRConfig(
            propensity_model="lightgbm",
            outcome_model="lightgbm",
            corruption_model="beta_binomial",
            shrinkage_enabled=True,
            shrinkage_min_samples=50,
        )

        estimator = STREstimator(config)
        estimator.fit(self.X, self.Y_obs, self.R, self.O, self.D, self.groups)

        assert estimator.is_fitted
        # Shrinkage should produce different pseudo-labels than raw
        # (though with synthetic data the difference may be small)

    def test_get_training_weights(self):
        """Test inverse propensity weight computation."""
        config = STRConfig(shrinkage_enabled=False)
        estimator = STREstimator(config)
        estimator.fit(self.X, self.Y_obs, self.R, self.O, self.D, self.groups)

        weights = estimator.get_training_weights()
        assert weights.shape == (self.n_samples,)
        assert np.all(weights > 0)
        # Weights should be normalized to mean 1
        assert np.isclose(weights.mean(), 1.0, rtol=0.1)

    def test_predict_pseudo_labels(self):
        """Test predicting pseudo-labels for new data."""
        config = STRConfig(shrinkage_enabled=False)
        estimator = STREstimator(config)
        estimator.fit(self.X, self.Y_obs, self.R, self.O, self.D, self.groups)

        # New data
        X_new = np.random.randn(100, self.n_features).astype(np.float32)
        pseudo_new = estimator.predict_pseudo_labels(X_new)

        assert pseudo_new.shape == (100,)
        assert np.all(pseudo_new >= 0)
        assert np.all(pseudo_new <= 1)

    def test_estimate_variance(self):
        """Test variance estimation."""
        config = STRConfig(shrinkage_enabled=False, variance_estimator="plugin")
        estimator = STREstimator(config)
        estimator.fit(self.X, self.Y_obs, self.R, self.O, self.D, self.groups)

        variance = estimator.estimate_variance(self.X, estimator.pseudo_labels)
        assert variance.shape == (self.n_samples,)
        assert np.all(variance >= 0)

    def test_optimal_training_delay(self):
        """Test optimal training delay estimation."""
        config = STRConfig(
            shrinkage_enabled=False,
            optimize_training_delay=True,
            max_training_delay_days=60,
        )
        estimator = STREstimator(config)

        # Add delay days
        delay_days = np.random.randint(1, 90, self.n_samples)

        # This should not raise an error
        optimal_delay = estimator.optimal_training_delay(
            self.X, self.Y_obs, self.R, self.O, self.D, delay_days
        )

        assert isinstance(optimal_delay, int)
        assert 7 <= optimal_delay <= 60


class TestSTRClassifier:
    """Test STR Classifier (sklearn-compatible)."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n_samples = 500
        self.n_features = 15

        self.X = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        fraud_signal = 1.0 * (self.X[:, 0] > 0.5) + 0.5 * np.random.randn(self.n_samples)
        fraud_prob = 1 / (1 + np.exp(-fraud_signal))
        fraud_prob = np.clip(fraud_prob, 0.01, 0.4)
        self.y = np.random.binomial(1, fraud_prob)

        self.R = np.ones(self.n_samples, dtype=int)
        self.O = np.where(self.y == 1, np.random.binomial(1, 0.7, self.n_samples), 0)
        self.D = np.where(self.O == 1, np.random.binomial(1, 0.8, self.n_samples), 0)
        self.groups = np.random.randint(1, 5, self.n_samples)

    def test_fit_predict(self):
        """Test classifier fit and predict."""
        clf = STRClassifier(
            str_config=STRConfig(shrinkage_enabled=False),
            calibration="isotonic",
        )

        clf.fit(self.X, self.y, R=self.R, O=self.O, D=self.D, groups=self.groups)

        # Test predictions
        preds = clf.predict(self.X[:50])
        assert preds.shape == (50,)
        assert np.all(np.isin(preds, [0, 1]))

        probs = clf.predict_proba(self.X[:50])
        assert probs.shape == (50, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)

    def test_get_str_diagnostics(self):
        """Test diagnostic information."""
        clf = STRClassifier(str_config=STRConfig(shrinkage_enabled=False))
        clf.fit(self.X, self.y, R=self.R, O=self.O, D=self.D, groups=self.groups)

        diagnostics = clf.get_str_diagnostics()
        assert "corruption_rates" in diagnostics
        assert "propensity_means" in diagnostics
        assert "pseudo_label_stats" in diagnostics
        assert "mean" in diagnostics["pseudo_label_stats"]


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_labeled_samples(self):
        """Test error when no samples have R=O=D=1."""
        np.random.seed(42)
        X = np.random.randn(100, 10).astype(np.float32)
        Y_obs = np.zeros(100)
        R = np.zeros(100, dtype=int)  # None authorized
        O = np.zeros(100, dtype=int)
        D = np.zeros(100, dtype=int)

        estimator = STREstimator(STRConfig(shrinkage_enabled=False))

        with pytest.raises(ValueError, match="No labeled samples"):
            estimator.fit(X, Y_obs, R, O, D)

    def test_not_fitted_errors(self):
        """Test errors when using unfitted estimator."""
        estimator = STREstimator()

        with pytest.raises(ValueError, match="not fitted"):
            estimator.get_training_weights()

        with pytest.raises(ValueError, match="not fitted"):
            estimator.predict_pseudo_labels(np.random.randn(10, 5))

    def test_mismatched_dimensions(self):
        """Test error on dimension mismatch."""
        estimator = STREstimator(STRConfig(shrinkage_enabled=False))

        X = np.random.randn(100, 10)
        Y_obs = np.random.randint(0, 2, 50)  # Wrong size
        R = np.ones(100, dtype=int)
        O = np.ones(100, dtype=int)
        D = np.ones(100, dtype=int)

        with pytest.raises(AssertionError):
            estimator.fit(X, Y_obs, R, O, D)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

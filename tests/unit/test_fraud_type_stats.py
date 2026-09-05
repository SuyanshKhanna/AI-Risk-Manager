"""
Tests for the /fraud_type_stats endpoint and the fraud_type_stats module.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from services.upi_fraud_detector.fraud_type_stats import (
    FraudTypeResult,
    FRAUD_TYPE_STATS,
    get_all_fraud_type_stats,
    get_fraud_type_stat,
)


# ──────────────────────────────────────────────────────────────────────────────
# Module-level unit tests
# ──────────────────────────────────────────────────────────────────────────────


class TestFraudTypeStats:
    """Unit tests for the fraud_type_stats data module."""

    def test_all_stats_returns_list(self):
        stats = get_all_fraud_type_stats()
        assert isinstance(stats, list)
        assert len(stats) >= 1

    def test_every_entry_is_fraud_type_result(self):
        for s in get_all_fraud_type_stats():
            assert isinstance(s, FraudTypeResult)

    def test_implemented_types_have_positive_count(self):
        """Implemented fraud types must have a real (non-zero) TP count."""
        for s in get_all_fraud_type_stats():
            if s.implemented:
                assert s.count_caught > 0, (
                    f"{s.type_name}: implemented=True but count_caught={s.count_caught}"
                )

    def test_implemented_types_have_detection_method(self):
        """Every implemented type must supply a detection_method."""
        for s in get_all_fraud_type_stats():
            if s.implemented:
                assert s.detection_method is not None and s.detection_method.strip() != "", (
                    f"{s.type_name}: implemented=True but detection_method is missing"
                )

    def test_unimplemented_types_have_zero_count(self):
        """Unimplemented types must never report a non-zero count."""
        for s in get_all_fraud_type_stats():
            if not s.implemented:
                assert s.count_caught == 0, (
                    f"{s.type_name}: implemented=False but count_caught={s.count_caught} (no invented numbers!)"
                )

    def test_unimplemented_types_have_no_detection_method(self):
        """Unimplemented types must not carry a detection_method."""
        for s in get_all_fraud_type_stats():
            if not s.implemented:
                assert s.detection_method is None, (
                    f"{s.type_name}: implemented=False but detection_method is set"
                )

    def test_no_duplicate_type_names(self):
        names = [s.type_name for s in get_all_fraud_type_stats()]
        assert len(names) == len(set(names)), "Duplicate type_name found"

    def test_get_fraud_type_stat_known(self):
        result = get_fraud_type_stat("fake_screenshot")
        assert result is not None
        assert result.type_name == "fake_screenshot"
        assert result.implemented is True
        assert result.count_caught == 45

    def test_get_fraud_type_stat_unknown(self):
        result = get_fraud_type_stat("nonexistent_type_xyz")
        assert result is None

    def test_known_implemented_types_present(self):
        """Smoke-check that the expected fraud types exist."""
        names = {s.type_name for s in get_all_fraud_type_stats()}
        expected_implemented = {
            "fake_screenshot",
            "device_emulator",
            "velocity_burst",
            "qr_tampering",
            "vpn_proxy",
            "same_device_refund",
            "known_fraud_device",
            "refund_abuse",
            "remote_access_attack",
        }
        assert expected_implemented.issubset(names)

    def test_known_unimplemented_types_present(self):
        names = {s.type_name for s in get_all_fraud_type_stats()}
        expected_unimplemented = {"qr_overlay_image", "sim_swap", "account_takeover"}
        assert expected_unimplemented.issubset(names)

    def test_total_implemented_count_sane(self):
        """
        Total TP count across implemented types should be >= test-set fraud
        (100) because multiple signals can fire on one transaction, but it
        should not wildly exceed it either.
        """
        total = sum(
            s.count_caught for s in get_all_fraud_type_stats() if s.implemented
        )
        assert total >= 100, "Total should be at least as large as the test-set fraud count"
        assert total < 1000, "Total is suspiciously large — double-check the counts"


# ──────────────────────────────────────────────────────────────────────────────
# HTTP endpoint tests (no live model required — endpoint is model-agnostic)
# ──────────────────────────────────────────────────────────────────────────────


class TestFraudTypeStatsEndpoint:
    """Integration tests for GET /fraud_type_stats via the FastAPI test client."""

    @pytest.fixture(scope="class")
    def client(self):
        """
        Return a TestClient without loading the real LightGBM model.
        The /fraud_type_stats endpoint reads static data only and does
        not call get_detector(), so no model is needed.
        """
        from services.upi_fraud_detector.serve import app
        return TestClient(app, raise_server_exceptions=True)

    def test_endpoint_returns_200(self, client):
        resp = client.get("/fraud_type_stats")
        assert resp.status_code == 200, resp.text

    def test_response_has_required_top_level_keys(self, client):
        data = client.get("/fraud_type_stats").json()
        required = {
            "total_fraud_caught",
            "test_set_size",
            "test_set_fraud_total",
            "threshold_used",
            "note",
            "fraud_types",
        }
        assert required.issubset(data.keys())

    def test_fraud_types_is_non_empty_list(self, client):
        data = client.get("/fraud_type_stats").json()
        assert isinstance(data["fraud_types"], list)
        assert len(data["fraud_types"]) >= 1

    def test_every_fraud_type_has_required_fields(self, client):
        data = client.get("/fraud_type_stats").json()
        for ft in data["fraud_types"]:
            assert "type_name" in ft
            assert "display_name" in ft
            assert "count_caught" in ft
            assert "implemented" in ft

    def test_implemented_false_has_no_detection_method(self, client):
        """
        Unimplemented entries must not carry a detection_method and must
        report count_caught = 0.  This is the critical anti-fabrication check.
        """
        data = client.get("/fraud_type_stats").json()
        for ft in data["fraud_types"]:
            if not ft["implemented"]:
                assert ft["count_caught"] == 0, (
                    f"{ft['type_name']}: implemented=false but count_caught={ft['count_caught']}"
                )
                # detection_method must be absent or null — never a fake string
                assert ft.get("detection_method") is None, (
                    f"{ft['type_name']}: implemented=false but has a detection_method"
                )

    def test_implemented_true_has_detection_method(self, client):
        data = client.get("/fraud_type_stats").json()
        for ft in data["fraud_types"]:
            if ft["implemented"]:
                dm = ft.get("detection_method")
                assert dm is not None and isinstance(dm, str) and len(dm) > 10, (
                    f"{ft['type_name']}: implemented=true but detection_method is missing or empty"
                )

    def test_total_fraud_caught_equals_sum_of_implemented(self, client):
        data = client.get("/fraud_type_stats").json()
        computed = sum(
            ft["count_caught"] for ft in data["fraud_types"] if ft["implemented"]
        )
        assert data["total_fraud_caught"] == computed

    def test_test_set_fraud_total_is_100(self, client):
        data = client.get("/fraud_type_stats").json()
        assert data["test_set_fraud_total"] == 100

    def test_test_set_size_is_10101(self, client):
        data = client.get("/fraud_type_stats").json()
        assert data["test_set_size"] == 10101

    def test_fake_screenshot_count(self, client):
        data = client.get("/fraud_type_stats").json()
        entry = next(ft for ft in data["fraud_types"] if ft["type_name"] == "fake_screenshot")
        assert entry["count_caught"] == 45
        assert entry["implemented"] is True

    def test_sim_swap_is_not_implemented(self, client):
        data = client.get("/fraud_type_stats").json()
        entry = next(ft for ft in data["fraud_types"] if ft["type_name"] == "sim_swap")
        assert entry["implemented"] is False
        assert entry["count_caught"] == 0
        assert entry.get("detection_method") is None

    def test_no_content_type_error_on_get(self, client):
        """GET should not require Content-Type: application/json."""
        resp = client.get("/fraud_type_stats")
        assert resp.status_code != 415


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

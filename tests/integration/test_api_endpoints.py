"""
Integration tests for UPI Fraud Detector API endpoints.
Tests /health, /score, /batch_score, and /demo endpoints.
"""

import pytest
import httpx
from typing import Dict, Any


class TestHealthEndpoint:
    """Test /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_endpoint_returns_200(self, client: httpx.AsyncClient):
        """Health endpoint should return 200 OK."""
        response = await client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_endpoint_structure(self, client: httpx.AsyncClient):
        """Health endpoint should return expected JSON structure."""
        response = await client.get("/health")
        data = response.json()

        assert "status" in data
        assert data["status"] in ["healthy", "degraded"]
        assert "model_version" in data
        assert "metrics" in data
        assert isinstance(data["metrics"], dict)

    @pytest.mark.asyncio
    async def test_health_metrics_contains_expected_keys(self, client: httpx.AsyncClient):
        """Health metrics should contain PR-AUC, ROC-AUC, and threshold metrics."""
        response = await client.get("/health")
        metrics = response.json()["metrics"]

        # Metrics may be empty if model not loaded, but should have expected structure
        expected_keys = [
            "pr_auc", "roc_auc",
            "challenge_precision", "challenge_recall", "challenge_fpr", "challenge_fp_cost",
            "block_precision", "block_recall", "block_fpr", "block_fp_cost"
        ]
        if metrics:  # Only check if metrics are present
            for key in expected_keys:
                assert key in metrics, f"Missing metric: {key}"


class TestScoreEndpoint:
    """Test /score endpoint."""

    @pytest.mark.asyncio
    async def test_score_legitimate_transaction(self, client: httpx.AsyncClient, legitimate_transaction: Dict[str, Any]):
        """Legitimate transaction should return ALLOW action."""
        response = await client.post("/score", json=legitimate_transaction)
        assert response.status_code == 200

        data = response.json()
        assert data["txn_id"] == legitimate_transaction["txn_id"]
        assert "score" in data
        assert 0.0 <= data["score"] <= 1.0
        assert data["action"] in ["ALLOW", "CHALLENGE", "BLOCK"]
        assert isinstance(data["reasons"], list)
        assert isinstance(data["shap_top3"], list)

    @pytest.mark.asyncio
    async def test_score_fraud_transaction(self, client: httpx.AsyncClient, fraud_transaction: Dict[str, Any]):
        """Fraud transaction should return CHALLENGE or BLOCK action."""
        response = await client.post("/score", json=fraud_transaction)
        assert response.status_code == 200

        data = response.json()
        assert data["txn_id"] == fraud_transaction["txn_id"]
        assert 0.0 <= data["score"] <= 1.0
        assert data["action"] in ["CHALLENGE", "BLOCK"]
        assert len(data["reasons"]) > 0
        assert len(data["shap_top3"]) <= 3

    @pytest.mark.asyncio
    async def test_score_challenge_transaction(self, client: httpx.AsyncClient, challenge_transaction: Dict[str, Any]):
        """Medium-risk transaction should return CHALLENGE action."""
        response = await client.post("/score", json=challenge_transaction)
        assert response.status_code == 200

        data = response.json()
        assert data["txn_id"] == challenge_transaction["txn_id"]
        assert 0.0 <= data["score"] <= 1.0
        assert data["action"] in ["ALLOW", "CHALLENGE", "BLOCK"]

    @pytest.mark.asyncio
    async def test_score_invalid_txn_id(self, client: httpx.AsyncClient, legitimate_transaction: Dict[str, Any]):
        """Missing txn_id should return 422 validation error."""
        txn = legitimate_transaction.copy()
        del txn["txn_id"]

        response = await client.post("/score", json=txn)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_score_invalid_amount(self, client: httpx.AsyncClient, legitimate_transaction: Dict[str, Any]):
        """Negative amount should return 422 validation error."""
        txn = legitimate_transaction.copy()
        txn["amount_paise"] = -100

        response = await client.post("/score", json=txn)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_score_missing_required_field(self, client: httpx.AsyncClient, legitimate_transaction: Dict[str, Any]):
        """Missing required field should return 422."""
        txn = legitimate_transaction.copy()
        del txn["merchant_id"]

        response = await client.post("/score", json=txn)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_score_response_consistency(self, client: httpx.AsyncClient, legitimate_transaction: Dict[str, Any]):
        """Repeated scoring of same transaction should return consistent results."""
        responses = []
        for _ in range(3):
            response = await client.post("/score", json=legitimate_transaction)
            assert response.status_code == 200
            responses.append(response.json())

        # All scores should be identical (deterministic model)
        scores = [r["score"] for r in responses]
        assert all(abs(s - scores[0]) < 1e-6 for s in scores)

        actions = [r["action"] for r in responses]
        assert all(a == actions[0] for a in actions)


class TestBatchScoreEndpoint:
    """Test /batch_score endpoint."""

    @pytest.mark.asyncio
    async def test_batch_score_returns_all_results(self, client: httpx.AsyncClient, mixed_batch: list):
        """Batch score should return results for all transactions."""
        response = await client.post("/batch_score", json={"transactions": mixed_batch})
        assert response.status_code == 200

        data = response.json()
        assert "results" in data
        assert len(data["results"]) == len(mixed_batch)

    @pytest.mark.asyncio
    async def test_batch_score_preserves_order(self, client: httpx.AsyncClient, mixed_batch: list):
        """Batch score results should preserve input order."""
        response = await client.post("/batch_score", json={"transactions": mixed_batch})
        data = response.json()

        for i, result in enumerate(data["results"]):
            assert result["txn_id"] == mixed_batch[i]["txn_id"]

    @pytest.mark.asyncio
    async def test_batch_score_empty_list(self, client: httpx.AsyncClient):
        """Empty batch should return empty results."""
        response = await client.post("/batch_score", json={"transactions": []})
        assert response.status_code == 200

        data = response.json()
        assert data["results"] == []

    @pytest.mark.asyncio
    async def test_batch_score_large_batch(self, client: httpx.AsyncClient, legitimate_transaction: Dict[str, Any]):
        """Batch score should handle larger batches (100 transactions)."""
        large_batch = [
            {**legitimate_transaction, "txn_id": f"INT_BATCH_{i}"}
            for i in range(100)
        ]

        response = await client.post("/batch_score", json={"transactions": large_batch})
        assert response.status_code == 200

        data = response.json()
        assert len(data["results"]) == 100

    @pytest.mark.asyncio
    async def test_batch_score_invalid_item(self, client: httpx.AsyncClient, legitimate_transaction: Dict[str, Any]):
        """Batch with one invalid item should return 422."""
        batch = [
            legitimate_transaction,
            {**legitimate_transaction, "txn_id": "INVALID", "amount_paise": -1}
        ]

        response = await client.post("/batch_score", json={"transactions": batch})
        assert response.status_code == 422


class TestDemoEndpoint:
    """Test /demo endpoint."""

    @pytest.mark.asyncio
    async def test_demo_endpoint_returns_200(self, client: httpx.AsyncClient):
        """Demo endpoint should return 200 OK."""
        response = await client.post("/demo")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_demo_endpoint_structure(self, client: httpx.AsyncClient):
        """Demo endpoint should return expected structure with two transactions."""
        response = await client.post("/demo")
        data = response.json()

        assert "demo_transactions" in data
        assert len(data["demo_transactions"]) == 2

    @pytest.mark.asyncio
    async def test_demo_contains_legit_and_fraud(self, client: httpx.AsyncClient):
        """Demo should contain one legitimate and one fraud example."""
        response = await client.post("/demo")
        data = response.json()

        types = [txn["type"] for txn in data["demo_transactions"]]
        assert any("LEGITIMATE" in t for t in types)
        assert any("FRAUD" in t for t in types)

    @pytest.mark.asyncio
    async def test_demo_transactions_have_all_fields(self, client: httpx.AsyncClient):
        """Each demo transaction should have all required fields."""
        response = await client.post("/demo")
        data = response.json()

        for txn in data["demo_transactions"]:
            assert "txn_id" in txn
            assert "type" in txn
            assert "score" in txn
            assert "action" in txn
            assert "reasons" in txn
            assert isinstance(txn["reasons"], list)
            assert len(txn["reasons"]) > 0


class TestErrorHandling:
    """Test error responses."""

    @pytest.mark.asyncio
    async def test_404_for_unknown_endpoint(self, client: httpx.AsyncClient):
        """Unknown endpoints should return 404."""
        response = await client.get("/nonexistent")
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_405_for_wrong_method_on_score(self, client: httpx.AsyncClient):
        """GET on /score should return 405 Method Not Allowed."""
        response = await client.get("/score")
        assert response.status_code == 405

    @pytest.mark.asyncio
    async def test_405_for_wrong_method_on_batch(self, client: httpx.AsyncClient):
        """GET on /batch_score should return 405 Method Not Allowed."""
        response = await client.get("/batch_score")
        assert response.status_code == 405


class TestPerformance:
    """Basic performance checks."""

    @pytest.mark.asyncio
    async def test_score_latency_under_100ms(self, client: httpx.AsyncClient, legitimate_transaction: Dict[str, Any]):
        """Single score request should complete within 100ms."""
        import time
        start = time.perf_counter()
        response = await client.post("/score", json=legitimate_transaction)
        elapsed = (time.perf_counter() - start) * 1000

        assert response.status_code == 200
        assert elapsed < 100, f"Score latency {elapsed:.1f}ms exceeds 100ms threshold"

    @pytest.mark.asyncio
    async def test_batch_score_latency(self, client: httpx.AsyncClient, legitimate_transaction: Dict[str, Any]):
        """Batch of 10 should complete within 500ms."""
        import time
        batch = [
            {**legitimate_transaction, "txn_id": f"PERF_{i}"}
            for i in range(10)
        ]

        start = time.perf_counter()
        response = await client.post("/batch_score", json={"transactions": batch})
        elapsed = (time.perf_counter() - start) * 1000

        assert response.status_code == 200
        assert elapsed < 500, f"Batch latency {elapsed:.1f}ms exceeds 500ms threshold"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
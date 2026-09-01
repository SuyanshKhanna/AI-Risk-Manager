"""Tests for common library."""

from datetime import datetime, timezone

import pytest

from libs.common import (
    Action,
    AuthConfig,
    FraudVector,
    ServiceMetrics,
    create_jwt_token,
    current_timestamp_iso,
    current_timestamp_ms,
    decode_jwt_token,
    generate_api_key,
    generate_request_id,
    hash_string,
    parse_timestamp,
    retry_with_backoff,
    safe_json_dumps,
    safe_json_loads,
    schemas,
    utils,
)


class TestUtils:
    """Test utility functions."""

    def test_generate_request_id(self):
        rid = generate_request_id()
        assert rid.startswith("req_")
        assert len(rid) == 20  # req_ + 16 hex chars

    def test_current_timestamp_ms(self):
        ts = current_timestamp_ms()
        assert isinstance(ts, int)
        assert ts > 1e12  # Should be in milliseconds

    def test_current_timestamp_iso(self):
        ts = current_timestamp_iso()
        assert isinstance(ts, str)
        assert "T" in ts

    def test_parse_timestamp_int_ms(self):
        ts = 1700000000000  # milliseconds
        dt = parse_timestamp(ts)
        assert isinstance(dt, datetime)
        assert dt.tzinfo == timezone.utc

    def test_parse_timestamp_int_sec(self):
        ts = 1700000000  # seconds
        dt = parse_timestamp(ts)
        assert isinstance(dt, datetime)

    def test_parse_timestamp_iso_string(self):
        ts = "2024-01-01T12:00:00Z"
        dt = parse_timestamp(ts)
        assert isinstance(dt, datetime)
        assert dt.year == 2024

    def test_safe_json_dumps(self):
        obj = {"a": 1, "b": datetime.now(timezone.utc)}
        result = safe_json_dumps(obj)
        assert isinstance(result, str)
        assert "a" in result

    def test_safe_json_loads(self):
        result = safe_json_loads('{"a": 1}')
        assert result == {"a": 1}

        result = safe_json_loads("invalid")
        assert result is None

    def test_hash_string(self):
        h = hash_string("test")
        assert len(h) == 16
        assert h == hash_string("test")
        assert h != hash_string("test2")

    def test_timer_context_manager(self):
        with utils.timer() as get_elapsed:
            pass
        elapsed = get_elapsed()
        assert elapsed >= 0

    @pytest.mark.asyncio
    async def test_retry_with_backoff(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        async def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary error")
            return "success"

        result = await failing_func()
        assert result == "success"
        assert call_count == 3


class TestSchemas:
    """Test Pydantic schemas."""

    def test_action_enum(self):
        assert Action.ALLOW == "ALLOW"
        assert Action.BLOCK == "BLOCK"
        assert Action.CHALLENGE == "CHALLENGE"

    def test_fraud_vector_enum(self):
        assert FraudVector.UPI == "upi"
        assert FraudVector.VOICE == "voice"
        assert FraudVector.KYC == "kyc"

    def test_upi_txn_request(self):
        req = schemas.UpiTxnRequest(
            request_id="test_123",
            merchant_id="M123",
            txn_id="T456",
            amount_paise=50000,
            upi_handle="user@upi",
            device_fingerprint="dev_abc",
        )
        assert req.merchant_id == "M123"
        assert req.amount_paise == 50000

    def test_upi_txn_response(self):
        resp = schemas.UpiTxnResponse(
            request_id="test_123",
            risk_score=0.75,
            action=Action.BLOCK,
            reasons=["High risk"],
            model_version="1.0",
            latency_ms=25.5,
        )
        assert resp.risk_score == 0.75
        assert resp.action == Action.BLOCK

    def test_feedback_label(self):
        label = schemas.FeedbackLabel(
            request_id="req_123",
            vector=FraudVector.UPI,
            true_label=1,
            label_source="chargeback",
            confidence=0.9,
        )
        assert label.true_label == 1
        assert label.vector == FraudVector.UPI


class TestAuth:
    """Test authentication utilities."""

    def test_create_and_decode_jwt(self):
        config = AuthConfig(JWT_SECRET="test-secret")
        token = create_jwt_token("service1", scopes=["read", "write"], config=config)
        assert isinstance(token, str)

        payload = decode_jwt_token(token, config=config)
        assert payload.sub == "service1"
        assert "read" in payload.scopes
        assert "write" in payload.scopes

    def test_jwt_expiry(self):
        config = AuthConfig(JWT_SECRET="test-secret")
        token = create_jwt_token("service1", expiry_minutes=0, config=config)

        # Should fail - token already expired
        import time
        time.sleep(0.1)
        with pytest.raises(Exception):
            decode_jwt_token(token, config=config)

    def test_generate_api_key(self):
        key_id, full_key = generate_api_key()
        assert key_id.startswith("ak_") is False  # Our prefix is ak_
        assert full_key.startswith("ak_")
        assert "." in full_key

    def test_hash_and_verify_api_key(self):
        key_id, full_key = generate_api_key()
        key_hash = hash_api_key(full_key)

        assert verify_api_key(full_key, key_hash)
        assert not verify_api_key("wrong_key", key_hash)

    def test_create_service_token(self):
        config = AuthConfig(JWT_SECRET="test-secret")
        token = create_service_token(
            "upi_fraud_detector",
            vectors=[FraudVector.UPI, FraudVector.VOICE],
            config=config,
        )

        payload = decode_jwt_token(token, config=config)
        assert payload.sub == "upi_fraud_detector"
        assert payload.metadata["type"] == "service"
        assert "upi" in payload.metadata["vectors"]


class TestMetrics:
    """Test metrics utilities."""

    def test_service_metrics_init(self):
        svc_metrics = ServiceMetrics("test_service")
        assert svc_metrics.service_name == "test_service"
        assert svc_metrics.requests_total is not None

    def test_record_request(self):
        svc_metrics = ServiceMetrics("test_service")
        svc_metrics.record_request("GET", "/health", 200, 0.01)
        # Metrics recorded internally


class TestRetryLogic:
    """Test retry decorator."""

    def test_retry_success_first_try(self):
        call_count = 0

        @retry_with_backoff(max_retries=3, base_delay=0.01)
        def success_func():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = success_func()
        assert result == "ok"
        assert call_count == 1

    def test_retry_exhausted(self):
        call_count = 0

        @retry_with_backoff(max_retries=2, base_delay=0.01)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError):
            always_fail()
        assert call_count == 3  # Initial + 2 retries


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

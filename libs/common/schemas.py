"""Shared Pydantic schemas for all services."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Action(str, Enum):
    """Decision actions."""
    ALLOW = "ALLOW"
    CHALLENGE = "CHALLENGE"
    BLOCK = "BLOCK"
    AUTO_SUBMIT = "AUTO_SUBMIT"
    SHADOW_BAN = "SHADOW_BAN"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class RiskTier(str, Enum):
    """Merchant risk tiers."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FraudVector(str, Enum):
    """Fraud vector types."""
    UPI = "upi"
    VOICE = "voice"
    KYC = "kyc"
    CHARGEBACK = "chargeback"
    RETURN = "return"
    REVIEW = "review"


# Base request/response models
class BaseRequest(BaseModel):
    """Base request with common fields."""
    request_id: str = Field(..., description="Unique request identifier")
    merchant_id: str = Field(..., description="Merchant identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseResponse(BaseModel):
    """Base response with common fields."""
    request_id: str
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score [0,1]")
    action: Action
    reasons: list[str] = Field(default_factory=list)
    shap_values: dict[str, float] = Field(default_factory=dict)
    model_version: str
    latency_ms: float
    vector: FraudVector


# UPI-specific models
class UpiTxnRequest(BaseRequest):
    """UPI transaction scoring request."""
    txn_id: str
    amount_paise: int = Field(..., gt=0)
    upi_handle: str
    device_fingerprint: str
    qr_image_b64: str | None = None
    screenshot_b64: str | None = None
    screenshot_ts_ms: int | None = None
    pg_callback_status: str | None = None
    pg_callback_ts_ms: int | None = None


class UpiTxnResponse(BaseResponse):
    """UPI transaction scoring response."""
    vector: FraudVector = FraudVector.UPI
    settlement_verified: bool = False
    qr_tamper_detected: bool = False
    velocity_anomaly: bool = False


# Voice-specific models
class VoiceAuthRequest(BaseRequest):
    """Voice authentication request."""
    call_session_id: str
    audio_b64: str  # Base64 encoded 16kHz audio
    caller_id: str | None = None
    enrolled_voiceprint_id: str | None = None
    challenge_digits: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VoiceAuthResponse(BaseResponse):
    """Voice authentication response."""
    vector: FraudVector = FraudVector.VOICE
    liveness_score: float = 0.0
    voiceprint_match: float = 0.0
    codec_detected: str | None = None


# KYC-specific models
class KycLivenessRequest(BaseRequest):
    """KYC liveness check request."""
    session_id: str
    video_b64: str  # Base64 encoded video frames
    document_image_b64: str | None = None
    challenge_type: str | None = None  # blink, smile, head_turn
    metadata: dict[str, Any] = Field(default_factory=dict)


class KycLivenessResponse(BaseResponse):
    """KYC liveness check response."""
    vector: FraudVector = FraudVector.KYC
    liveness_passed: bool = False
    deepfake_detected: bool = False
    injection_detected: bool = False
    document_forged: bool = False
    face_quality_score: float = 0.0


# Chargeback-specific models
class ChargebackRequest(BaseRequest):
    """Chargeback dispute request."""
    dispute_id: str
    txn_id: str
    amount_paise: int
    reason_code: str
    issuer_id: str
    card_network: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChargebackResponse(BaseResponse):
    """Chargeback dispute response."""
    vector: FraudVector = FraudVector.CHARGEBACK
    win_probability: float = 0.0
    evidence_package: dict[str, Any] | None = None
    auto_submit: bool = False


# Return-specific models
class ReturnRiskRequest(BaseRequest):
    """Return risk scoring request."""
    return_id: str
    order_id: str
    damage_images_b64: list[str] = Field(default_factory=list)
    return_reason: str
    customer_return_history: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReturnRiskResponse(BaseResponse):
    """Return risk scoring response."""
    vector: FraudVector = FraudVector.RETURN
    image_forensics_score: float = 0.0
    customer_risk_score: float = 0.0
    requires_video: bool = False
    requires_inspection: bool = False


# Review-specific models
class ReviewRingRequest(BaseRequest):
    """Review ring detection request."""
    review_id: str
    product_id: str
    reviewer_id: str
    review_text: str
    rating: int = Field(..., ge=1, le=5)
    session_data: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReviewRingResponse(BaseResponse):
    """Review ring detection response."""
    vector: FraudVector = FraudVector.REVIEW
    ring_probability: float = 0.0
    cluster_id: str | None = None
    is_purchase_verified: bool = False
    behavioral_anomaly: float = 0.0


# Decision engine models
class DecisionRequest(BaseModel):
    """Unified decision request."""
    request_id: str
    merchant_id: str
    vector_scores: dict[FraudVector, float]
    vector_actions: dict[FraudVector, Action]
    vector_reasons: dict[FraudVector, list[str]]
    context: dict[str, Any] = Field(default_factory=dict)


class DecisionResponse(BaseModel):
    """Unified decision response."""
    request_id: str
    final_action: Action
    final_score: float
    applied_rules: list[str]
    vector_contributions: dict[FraudVector, float]
    shap_values: dict[str, float]
    model_version: str
    latency_ms: float


# Feedback/Label models
class FeedbackLabel(BaseModel):
    """Feedback label for continuous learning."""
    request_id: str
    vector: FraudVector
    true_label: int  # 0 = legitimate, 1 = fraud
    label_source: str  # "chargeback", "manual_review", "investigation", "synthetic"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Health check models
class HealthResponse(BaseModel):
    """Health check response."""
    status: str  # "healthy", "degraded", "unhealthy"
    service: str
    version: str
    uptime_seconds: float
    dependencies: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)


# Metrics models
class MetricsSnapshot(BaseModel):
    """Metrics snapshot for monitoring."""
    service: str
    timestamp: datetime
    request_count: int
    error_count: int
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    score_distribution: dict[str, int]  # binned scores
    action_distribution: dict[Action, int]
    drift_detected: bool = False
    drift_features: list[str] = Field(default_factory=list)

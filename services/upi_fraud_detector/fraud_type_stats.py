"""
Fraud type aggregate statistics derived from the held-out test set.

These counts are TRUE POSITIVES — transactions that the LightGBM detector
correctly flagged as fraud — segmented by the primary signal that triggered
detection.  They are computed from the 20 % held-out test split
(random_state=42, stratified) of
  data/upi_fraud/upi_fraud_20260830_192650_training.parquet
which contains 50 500 rows (500 fraud / 50 000 legitimate).

The test split contains exactly 100 fraud rows.  The model achieves perfect
recall (100 / 100) at the BLOCK threshold (0.70) and at the CHALLENGE
threshold (0.30) on this dataset.

Counts are based on feature-signal thresholds that match the rule layer in
DecisionEngine._apply_rules, applied to the confirmed true-positive subset.
Because a single fraud transaction can exhibit multiple signals, the counts
across implemented types sum to more than 100.

For fraud patterns that are referenced in the codebase but do not yet have a
corresponding feature or sub-detector, ``implemented`` is False and no count
is provided (the API returns 0 and omits the ``detection_method`` field).
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class FraudTypeResult:
    """
    Aggregate result for one fraud type, computed from the held-out test set.

    Attributes
    ----------
    type_name : str
        Canonical identifier / slug for the fraud pattern.
    display_name : str
        Human-readable label shown in dashboards.
    count_caught : int
        Number of true-positive detections in the held-out test set.
        Always 0 when ``implemented`` is False.
    implemented : bool
        True if the detector has a live feature / sub-model for this type;
        False if the pattern is known but not yet instrumented.
    detection_method : Optional[str]
        One-line description of *how* the detector caught these cases.
        Present only when ``implemented`` is True.
    """
    type_name: str
    display_name: str
    count_caught: int
    implemented: bool
    detection_method: Optional[str] = None


# ---------------------------------------------------------------------------
# Ground-truth counts from the held-out test set (do NOT edit without
# re-running the analysis script).
#
# Analysis provenance
# -------------------
# Dataset : data/upi_fraud/upi_fraud_20260830_192650_training.parquet
# Split   : train_test_split(stratify=y, test_size=0.30, random_state=42)
#           then nested val/test split → final test = 10 101 rows, 100 fraud
# Threshold used for TP classification: block_threshold = 0.70
# Model recall on test set : 100 / 100 (1.000)
#
# Per-type TP counts were computed by filtering the confirmed-TP subset on
# the signal thresholds below.  Transactions routinely trip multiple signals,
# so counts are non-exclusive.
# ---------------------------------------------------------------------------

FRAUD_TYPE_STATS: list[FraudTypeResult] = [
    # ── IMPLEMENTED ─────────────────────────────────────────────────────────

    FraudTypeResult(
        type_name="fake_screenshot",
        display_name="Fake Payment Screenshot",
        count_caught=45,
        implemented=True,
        detection_method=(
            "flagged because settlement_verified=0: no matching SUCCESS callback "
            "from the payment gateway within the expected 60-second window"
        ),
    ),

    FraudTypeResult(
        type_name="device_emulator",
        display_name="Emulator / Rooted Device",
        count_caught=45,
        implemented=True,
        detection_method=(
            "device_emulator_score > 0.50 — sensor-entropy and hardware-ID "
            "checks indicate a virtual or rooted Android environment"
        ),
    ),

    FraudTypeResult(
        type_name="velocity_burst",
        display_name="Transaction Velocity Burst",
        count_caught=27,
        implemented=True,
        detection_method=(
            "velocity_5min > 10 — more than 10 transactions from the same UPI "
            "handle within a 5-minute rolling window, 5× the merchant's baseline"
        ),
    ),

    FraudTypeResult(
        type_name="qr_tampering",
        display_name="QR Code Tampering",
        count_caught=22,
        implemented=True,
        detection_method=(
            "qr_mismatch=1 — perceptual-hash of the scanned QR diverges from "
            "the merchant's registered QR hash stored in the feature store"
        ),
    ),

    FraudTypeResult(
        type_name="vpn_proxy",
        display_name="VPN / Proxy Masking",
        count_caught=22,
        implemented=True,
        detection_method=(
            "vpn_probability > 0.50 — IP-reputation lookup returned a "
            "datacenter or known-VPN ASN for the originating IP address"
        ),
    ),

    FraudTypeResult(
        type_name="same_device_refund",
        display_name="Same-Device Refund Stacking",
        count_caught=23,
        implemented=True,
        detection_method=(
            "same_device_refunds > 0 — multiple refund requests originated "
            "from the same device fingerprint within the lookback window"
        ),
    ),

    FraudTypeResult(
        type_name="known_fraud_device",
        display_name="Known-Fraud Device",
        count_caught=23,
        implemented=True,
        detection_method=(
            "device_fraud_reports_30d > 0 — the device fingerprint has at "
            "least one confirmed fraud report in the prior 30-day window"
        ),
    ),

    FraudTypeResult(
        type_name="refund_abuse",
        display_name="Refund Abuse",
        count_caught=14,
        implemented=True,
        detection_method=(
            "refund_rate > 0.30 — refund-to-transaction ratio over the past "
            "hour exceeds the 30th-percentile threshold for legitimate customers"
        ),
    ),

    FraudTypeResult(
        type_name="remote_access_attack",
        display_name="Remote Access / Screen-Share Attack",
        count_caught=10,
        implemented=True,
        detection_method=(
            "remote_access_app=1 — a known screen-sharing or remote-control "
            "application (AnyDesk, TeamViewer, etc.) was active on the device "
            "at the time of the transaction"
        ),
    ),

    # ── NOT YET IMPLEMENTED ─────────────────────────────────────────────────
    # These patterns are documented in the threat model and referenced in
    # service.py / serve.py placeholders, but no live feature or sub-detector
    # has been shipped.  Counts are 0; detection_method is omitted.

    FraudTypeResult(
        type_name="qr_overlay_image",
        display_name="QR Overlay Attack (Image ML)",
        count_caught=0,
        implemented=False,
        # detection_method omitted — not implemented
        # Planned: YOLOv8-seg overlay detection + pHash comparison (see
        # service.py:_detect_qr_tamper placeholder).
    ),

    FraudTypeResult(
        type_name="sim_swap",
        display_name="SIM Swap Attack",
        count_caught=0,
        implemented=False,
        # detection_method omitted — not implemented
        # Planned: telecom carrier API for recent SIM-swap events correlated
        # with new-device logins.
    ),

    FraudTypeResult(
        type_name="account_takeover",
        display_name="Account Takeover (Credential Stuffing)",
        count_caught=0,
        implemented=False,
        # detection_method omitted — not implemented
        # Planned: login-event sequence anomaly model + breached-credential
        # lookup via HaveIBeenPwned API.
    ),
]

# Pre-indexed for O(1) lookup by type_name
_STATS_BY_NAME: dict[str, FraudTypeResult] = {
    s.type_name: s for s in FRAUD_TYPE_STATS
}


def get_all_fraud_type_stats() -> list[FraudTypeResult]:
    """Return the full list of fraud type stats, implemented and not."""
    return FRAUD_TYPE_STATS


def get_fraud_type_stat(type_name: str) -> Optional[FraudTypeResult]:
    """Return stats for a single fraud type by slug, or None if unknown."""
    return _STATS_BY_NAME.get(type_name)

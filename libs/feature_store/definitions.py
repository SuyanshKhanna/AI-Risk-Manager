"""Feature store definitions using Feast."""

from feast import Entity, FeatureView, Field, FileSource, RedisSource
from feast.types import Float64, Int64, String, UnixTimestamp
from datetime import timedelta


# Entities
merchant = Entity(
    name="merchant",
    join_keys=["merchant_id"],
    description="Merchant entity",
)

customer = Entity(
    name="customer",
    join_keys=["customer_id"],
    description="Customer entity",
)

device = Entity(
    name="device",
    join_keys=["device_fingerprint"],
    description="Device entity",
)

upi_handle = Entity(
    name="upi_handle",
    join_keys=["upi_handle"],
    description="UPI handle entity",
)

order = Entity(
    name="order",
    join_keys=["order_id"],
    description="Order entity",
)

call_session = Entity(
    name="call_session",
    join_keys=["call_session_id"],
    description="Call session entity",
)


# Data sources (offline)
merchant_source = FileSource(
    path="data/feast/offline/merchant_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

customer_source = FileSource(
    path="data/feast/offline/customer_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

device_source = FileSource(
    path="data/feast/offline/device_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

upi_handle_source = FileSource(
    path="data/feast/offline/upi_handle_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

order_source = FileSource(
    path="data/feast/offline/order_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

call_session_source = FileSource(
    path="data/feast/offline/call_session_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)


# Data sources (online - Redis)
merchant_online = RedisSource(
    table_name="merchant_features",
    redis_connection_options={
        "host": "localhost",
        "port": 6379,
        "db": 0,
    },
)

customer_online = RedisSource(
    table_name="customer_features",
    redis_connection_options={
        "host": "localhost",
        "port": 6379,
        "db": 0,
    },
)

device_online = RedisSource(
    table_name="device_features",
    redis_connection_options={
        "host": "localhost",
        "port": 6379,
        "db": 0,
    },
)

upi_handle_online = RedisSource(
    table_name="upi_handle_features",
    redis_connection_options={
        "host": "localhost",
        "port": 6379,
        "db": 0,
    },
)

order_online = RedisSource(
    table_name="order_features",
    redis_connection_options={
        "host": "localhost",
        "port": 6379,
        "db": 0,
    },
)

call_session_online = RedisSource(
    table_name="call_session_features",
    redis_connection_options={
        "host": "localhost",
        "port": 6379,
        "db": 0,
    },
)


# Feature Views

merchant_risk_profile = FeatureView(
    name="merchant_risk_profile",
    entities=[merchant],
    ttl=timedelta(days=90),
    schema=[
        Field(name="merchant_category", dtype=String),
        Field(name="merchant_gmv_30d", dtype=Float64),
        Field(name="merchant_txn_count_30d", dtype=Int64),
        Field(name="merchant_chargeback_rate_30d", dtype=Float64),
        Field(name="merchant_refund_rate_30d", dtype=Float64),
        Field(name="merchant_avg_order_value", dtype=Float64),
        Field(name="merchant_churn_risk_score", dtype=Float64),
        Field(name="merchant_onboarding_date", dtype=UnixTimestamp),
        Field(name="merchant_risk_tier", dtype=String),
        Field(name="merchant_fraud_rate_30d", dtype=Float64),
        Field(name="merchant_dispute_win_rate_90d", dtype=Float64),
    ],
    online=True,
    source=merchant_source,
    tags={"team": "fraud", "domain": "merchant"},
)

customer_behavior = FeatureView(
    name="customer_behavior",
    entities=[customer],
    ttl=timedelta(days=90),
    schema=[
        Field(name="customer_txn_count_30d", dtype=Int64),
        Field(name="customer_avg_txn_amount", dtype=Float64),
        Field(name="customer_return_rate_90d", dtype=Float64),
        Field(name="customer_chargeback_count_90d", dtype=Int64),
        Field(name="customer_device_count_30d", dtype=Int64),
        Field(name="customer_upi_handle_count_30d", dtype=Int64),
        Field(name="customer_first_txn_date", dtype=UnixTimestamp),
        Field(name="customer_risk_tier", dtype=String),
        Field(name="customer_lifetime_value", dtype=Float64),
        Field(name="customer_preferred_categories", dtype=String),  # JSON array
        Field(name="customer_preferred_payment_methods", dtype=String),  # JSON array
    ],
    online=True,
    source=customer_source,
    tags={"team": "fraud", "domain": "customer"},
)

device_fingerprint = FeatureView(
    name="device_fingerprint",
    entities=[device],
    ttl=timedelta(days=90),
    schema=[
        Field(name="device_os", dtype=String),
        Field(name="device_model", dtype=String),
        Field(name="device_sensor_entropy", dtype=Float64),
        Field(name="device_app_tamper_score", dtype=Float64),
        Field(name="device_emulator_score", dtype=Float64),
        Field(name="device_root_score", dtype=Float64),
        Field(name="device_vpn_probability", dtype=Float64),
        Field(name="device_first_seen", dtype=UnixTimestamp),
        Field(name="device_merchant_count_30d", dtype=Int64),
        Field(name="device_customer_count_30d", dtype=Int64),
        Field(name="device_upi_handle_count_30d", dtype=Int64),
        Field(name="device_fraud_reports_30d", dtype=Int64),
    ],
    online=True,
    source=device_source,
    tags={"team": "fraud", "domain": "device"},
)

upi_handle_velocity = FeatureView(
    name="upi_handle_velocity",
    entities=[upi_handle],
    ttl=timedelta(days=7),
    schema=[
        Field(name="upi_txn_count_5min", dtype=Int64),
        Field(name="upi_txn_count_1hr", dtype=Int64),
        Field(name="upi_refund_count_1hr", dtype=Int64),
        Field(name="upi_amount_zscore_1hr", dtype=Float64),
        Field(name="upi_unique_merchants_1hr", dtype=Int64),
        Field(name="upi_unique_devices_1hr", dtype=Int64),
        Field(name="upi_first_seen", dtype=UnixTimestamp),
        Field(name="upi_chargeback_count_24hr", dtype=Int64),
        Field(name="upi_dispute_rate_24hr", dtype=Float64),
    ],
    online=True,
    source=upi_handle_source,
    tags={"team": "fraud", "domain": "upi"},
)

order_features = FeatureView(
    name="order_features",
    entities=[order],
    ttl=timedelta(days=90),
    schema=[
        Field(name="order_amount_paise", dtype=Int64),
        Field(name="order_category", dtype=String),
        Field(name="order_payment_method", dtype=String),
        Field(name="order_shipping_address_hash", dtype=String),
        Field(name="order_billing_address_hash", dtype=String),
        Field(name="order_device_fingerprint", dtype=String),
        Field(name="order_customer_id", dtype=String),
        Field(name="order_merchant_id", dtype=String),
        Field(name="order_timestamp", dtype=UnixTimestamp),
        Field(name="order_delivery_status", dtype=String),
        Field(name="order_return_status", dtype=String),
        Field(name="order_chargeback_status", dtype=String),
    ],
    online=True,
    source=order_source,
    tags={"team": "fraud", "domain": "order"},
)

call_session_features = FeatureView(
    name="call_session_features",
    entities=[call_session],
    ttl=timedelta(days=30),
    schema=[
        Field(name="call_duration_seconds", dtype=Int64),
        Field(name="call_caller_id", dtype=String),
        Field(name="call_callee_id", dtype=String),
        Field(name="call_direction", dtype=String),  # inbound/outbound
        Field(name="call_codec", dtype=String),
        Field(name="call_network_type", dtype=String),
        Field(name="call_stir_shaken_attestation", dtype=String),
        Field(name="call_ani_validated", dtype=String),  # boolean as string
        Field(name="call_urgency_keywords", dtype=String),  # JSON array
        Field(name="call_secrecy_keywords", dtype=String),  # JSON array
        Field(name="call_authority_keywords", dtype=String),  # JSON array
        Field(name="call_timestamp", dtype=UnixTimestamp),
    ],
    online=True,
    source=call_session_source,
    tags={"team": "fraud", "domain": "voice"},
)

# Derived/On-demand feature views (computed at request time)
from feast import OnDemandFeatureView
from feast.feature_view import FeatureView as FV

# These would be defined with @on_demand_feature_view decorator
# in the actual implementation file


def get_all_entities():
    """Get all entity definitions."""
    return [
        merchant, customer, device, upi_handle, order, call_session
    ]


def get_all_feature_views():
    """Get all feature view definitions."""
    return [
        merchant_risk_profile,
        customer_behavior,
        device_fingerprint,
        upi_handle_velocity,
        order_features,
        call_session_features,
    ]


def get_all_sources():
    """Get all data source definitions."""
    return [
        merchant_source, customer_source, device_source,
        upi_handle_source, order_source, call_session_source,
        merchant_online, customer_online, device_online,
        upi_handle_online, order_online, call_session_online,
    ]
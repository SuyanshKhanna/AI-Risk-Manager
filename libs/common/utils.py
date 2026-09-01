"""Common utilities."""

import hashlib
import json
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from typing import Any

import structlog
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# Structured logging
logger = structlog.get_logger(__name__)


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req_{uuid.uuid4().hex[:16]}"


def generate_trace_id() -> str:
    """Generate a trace ID for distributed tracing."""
    return uuid.uuid4().hex


def hash_string(s: str) -> str:
    """Generate SHA256 hash of a string."""
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def current_timestamp_ms() -> int:
    """Current timestamp in milliseconds."""
    return int(time.time() * 1000)


def current_timestamp_iso() -> str:
    """Current timestamp in ISO format with UTC."""
    return datetime.now(timezone.utc).isoformat()


def parse_timestamp(ts: Any) -> datetime | None:
    """Parse various timestamp formats to datetime."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, (int, float)):
        # Assume milliseconds if > 1e12, else seconds
        if ts > 1e12:
            return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(ts, str):
        for fmt in [
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ]:
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                continue
    return None


def safe_json_dumps(obj: Any, default: Any = None) -> str:
    """Safely serialize to JSON with custom default handler."""
    def default_handler(o):
        if hasattr(o, 'isoformat'):
            return o.isoformat()
        if hasattr(o, '__dict__'):
            return o.__dict__
        return str(o)
    return json.dumps(obj, default=default_handler or default)


def safe_json_loads(s: str) -> Any:
    """Safely deserialize JSON."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


@contextmanager
def timer() -> Callable[[], float]:
    """Context manager that returns elapsed time in milliseconds."""
    start = time.perf_counter()
    yield lambda: (time.perf_counter() - start) * 1000


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 0.1,
    max_delay: float = 5.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,),
):
    """Decorator for retry with exponential backoff."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            "Retry attempt",
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            delay=delay,
                            error=str(e),
                        )
                        time.sleep(delay)
                        delay = min(delay * exponential_base, max_delay)
                    else:
                        logger.error(
                            "Max retries exceeded",
                            max_retries=max_retries,
                            error=str(e),
                        )
                        raise
            raise last_exception
        return wrapper
    return decorator


class MetricsTimer:
    """Context manager for timing operations with Prometheus metrics."""

    def __init__(self, histogram: Histogram, labels: dict[str, str] | None = None):
        self.histogram = histogram
        self.labels = labels or {}
        self.start_time = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        if self.labels:
            self.histogram.labels(**self.labels).observe(elapsed_ms / 1000)
        else:
            self.histogram.observe(elapsed_ms / 1000)


def get_prometheus_registry() -> CollectorRegistry:
    """Get or create Prometheus registry."""
    return CollectorRegistry()


# Common Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['service', 'method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_latency_seconds',
    'HTTP request latency in seconds',
    ['service', 'method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

GRPC_REQUEST_COUNT = Counter(
    'grpc_requests_total',
    'Total gRPC requests',
    ['service', 'method', 'status']
)

GRPC_REQUEST_LATENCY = Histogram(
    'grpc_request_latency_seconds',
    'gRPC request latency in seconds',
    ['service', 'method'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

MODEL_INFERENCE_LATENCY = Histogram(
    'model_inference_latency_seconds',
    'Model inference latency in seconds',
    ['service', 'model'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

MODEL_SCORE_DISTRIBUTION = Histogram(
    'model_score_distribution',
    'Distribution of model risk scores',
    ['service', 'vector'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections',
    ['service']
)

QUEUE_SIZE = Gauge(
    'queue_size',
    'Current queue size',
    ['service', 'queue_name']
)

CACHE_HIT_RATE = Gauge(
    'cache_hit_rate',
    'Cache hit rate',
    ['service', 'cache_name']
)

DRIFT_DETECTED = Gauge(
    'drift_detected',
    'Whether drift was detected',
    ['service', 'feature']
)

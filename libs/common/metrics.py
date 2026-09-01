"""Metrics and observability utilities."""

import time
from collections.abc import Callable
from contextlib import contextmanager
from functools import wraps
from typing import Any

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.grpc import GrpcInstrumentorClient, GrpcInstrumentorServer
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
)

from .utils import get_prometheus_registry

logger = structlog.get_logger(__name__)

# Global registry
_registry: CollectorRegistry | None = None
_tracer_provider: TracerProvider | None = None


def init_metrics(service_name: str, registry: CollectorRegistry = None) -> CollectorRegistry:
    """Initialize Prometheus metrics registry."""
    global _registry
    _registry = registry or get_prometheus_registry()
    return _registry


def init_tracing(service_name: str, otlp_endpoint: str = "http://localhost:4317") -> TracerProvider:
    """Initialize OpenTelemetry tracing."""
    global _tracer_provider

    resource = Resource(attributes={SERVICE_NAME: service_name})
    _tracer_provider = TracerProvider(resource=resource)

    # OTLP exporter
    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    _tracer_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Set as global tracer provider
    trace.set_tracer_provider(_tracer_provider)

    # Auto-instrumentation
    FastAPIInstrumentor.instrument()
    GrpcInstrumentorClient().instrument()
    GrpcInstrumentorServer().instrument()
    RequestsInstrumentor().instrument()
    RedisInstrumentor().instrument()

    return _tracer_provider


def get_tracer(name: str):
    """Get a tracer instance."""
    return trace.get_tracer(name)


# Metrics builders
def create_counter(name: str, description: str, labels: list = None) -> Counter:
    """Create a Prometheus counter."""
    return Counter(name, description, labels or [], registry=_registry)


def create_histogram(
    name: str,
    description: str,
    labels: list = None,
    buckets: list = None,
) -> Histogram:
    """Create a Prometheus histogram."""
    return Histogram(
        name, description, labels or [], buckets=buckets, registry=_registry
    )


def create_gauge(name: str, description: str, labels: list = None) -> Gauge:
    """Create a Prometheus gauge."""
    return Gauge(name, description, labels or [], registry=_registry)


def create_summary(name: str, description: str, labels: list = None) -> Summary:
    """Create a Prometheus summary."""
    return Summary(name, description, labels or [], registry=_registry)


# Standard metrics for all services
class ServiceMetrics:
    """Standard metrics for a service."""

    def __init__(self, service_name: str):
        self.service_name = service_name
        self._init_metrics()

    def _init_metrics(self):
        prefix = f"{self.service_name}_"

        # Request metrics
        self.requests_total = create_counter(
            f"{prefix}requests_total",
            "Total requests",
            ["method", "endpoint", "status"]
        )
        self.request_latency = create_histogram(
            f"{prefix}request_latency_seconds",
            "Request latency in seconds",
            ["method", "endpoint"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )

        # gRPC metrics
        self.grpc_requests_total = create_counter(
            f"{prefix}grpc_requests_total",
            "Total gRPC requests",
            ["method", "status"]
        )
        self.grpc_request_latency = create_histogram(
            f"{prefix}grpc_request_latency_seconds",
            "gRPC request latency in seconds",
            ["method"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        )

        # Model metrics
        self.model_inference_latency = create_histogram(
            f"{prefix}model_inference_latency_seconds",
            "Model inference latency in seconds",
            ["model"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        )
        self.model_score_distribution = create_histogram(
            f"{prefix}model_score_distribution",
            "Distribution of model risk scores",
            ["vector"],
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        self.model_predictions = create_counter(
            f"{prefix}model_predictions_total",
            "Total model predictions",
            ["vector", "action"]
        )

        # Feature store metrics
        self.feature_store_latency = create_histogram(
            f"{prefix}feature_store_latency_seconds",
            "Feature store latency in seconds",
            ["operation"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
        )
        self.feature_store_errors = create_counter(
            f"{prefix}feature_store_errors_total",
            "Feature store errors",
            ["operation", "error_type"]
        )

        # Cache metrics
        self.cache_hits = create_counter(
            f"{prefix}cache_hits_total",
            "Cache hits",
            ["cache_name"]
        )
        self.cache_misses = create_counter(
            f"{prefix}cache_misses_total",
            "Cache misses",
            ["cache_name"]
        )
        self.cache_hit_rate = create_gauge(
            f"{prefix}cache_hit_rate",
            "Cache hit rate",
            ["cache_name"]
        )

        # Queue metrics
        self.queue_size = create_gauge(
            f"{prefix}queue_size",
            "Current queue size",
            ["queue_name"]
        )
        self.queue_latency = create_histogram(
            f"{prefix}queue_latency_seconds",
            "Queue processing latency in seconds",
            ["queue_name"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
        )

        # Drift metrics
        self.drift_detected = create_gauge(
            f"{prefix}drift_detected",
            "Whether drift was detected",
            ["feature"]
        )
        self.drift_score = create_gauge(
            f"{prefix}drift_score",
            "Drift score (PSI/KS)",
            ["feature"]
        )

        # Business metrics
        self.fraud_detected = create_counter(
            f"{prefix}fraud_detected_total",
            "Fraud cases detected",
            ["vector", "action"]
        )
        self.false_positive_estimate = create_gauge(
            f"{prefix}false_positive_estimate",
            "Estimated false positive rate",
            ["vector"]
        )
        self.loss_prevented = create_counter(
            f"{prefix}loss_prevented_total",
            "Estimated loss prevented in paise",
            ["vector"]
        )

        # System metrics
        self.active_connections = create_gauge(
            f"{prefix}active_connections",
            "Active connections"
        )
        self.memory_usage = create_gauge(
            f"{prefix}memory_usage_bytes",
            "Memory usage in bytes"
        )
        self.cpu_usage = create_gauge(
            f"{prefix}cpu_usage_percent",
            "CPU usage percent"
        )

    def record_request(self, method: str, endpoint: str, status: int, latency: float):
        """Record HTTP request metrics."""
        self.requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
        self.request_latency.labels(method=method, endpoint=endpoint).observe(latency)

    def record_grpc_request(self, method: str, status: str, latency: float):
        """Record gRPC request metrics."""
        self.grpc_requests_total.labels(method=method, status=status).inc()
        self.grpc_request_latency.labels(method=method).observe(latency)

    def record_inference(self, model: str, latency: float):
        """Record model inference metrics."""
        self.model_inference_latency.labels(model=model).observe(latency)

    def record_score(self, vector: str, score: float, action: str):
        """Record model score and action."""
        self.model_score_distribution.labels(vector=vector).observe(score)
        self.model_predictions.labels(vector=vector, action=action).inc()

    def record_fraud(self, vector: str, action: str, loss_paise: int = 0):
        """Record fraud detection."""
        self.fraud_detected.labels(vector=vector, action=action).inc()
        if loss_paise > 0:
            self.loss_prevented.labels(vector=vector).inc(loss_paise)

    def record_cache(self, cache_name: str, hit: bool):
        """Record cache hit/miss."""
        if hit:
            self.cache_hits.labels(cache_name=cache_name).inc()
        else:
            self.cache_misses.labels(cache_name=cache_name).inc()

        # Update hit rate
        total = self.cache_hits.labels(cache_name=cache_name)._value.get() + \
                self.cache_misses.labels(cache_name=cache_name)._value.get()
        if total > 0:
            hit_rate = self.cache_hits.labels(cache_name=cache_name)._value.get() / total
            self.cache_hit_rate.labels(cache_name=cache_name).set(hit_rate)


# Decorators for automatic metrics
def track_latency(histogram: Histogram, labels: dict[str, str] = None):
    """Decorator to track function latency."""
    def decorator(func: Callable):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                if labels:
                    histogram.labels(**labels).observe(elapsed)
                else:
                    histogram.observe(elapsed)

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                elapsed = time.perf_counter() - start
                if labels:
                    histogram.labels(**labels).observe(elapsed)
                else:
                    histogram.observe(elapsed)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


def track_requests(counter: Counter, histogram: Histogram, labels: dict[str, str] = None):
    """Decorator to track request count and latency."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            status = "success"
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception:
                status = "error"
                raise
            finally:
                elapsed = time.perf_counter() - start
                if labels:
                    counter.labels(**labels, status=status).inc()
                    histogram.labels(**labels).observe(elapsed)
                else:
                    counter.labels(status=status).inc()
                    histogram.observe(elapsed)
        return wrapper
    return decorator


@contextmanager
def trace_operation(name: str, attributes: dict[str, Any] = None):
    """Context manager for tracing operations."""
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise


def get_metrics_output() -> bytes:
    """Get Prometheus metrics output."""
    if _registry:
        return generate_latest(_registry)
    return b""


def get_content_type() -> str:
    """Get Prometheus content type."""
    return CONTENT_TYPE_LATEST


# Health check metrics
HEALTH_CHECK_DURATION = create_histogram(
    "health_check_duration_seconds",
    "Health check duration in seconds",
    ["service", "check"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

HEALTH_CHECK_STATUS = create_gauge(
    "health_check_status",
    "Health check status (1=healthy, 0=unhealthy)",
    ["service", "check"]
)

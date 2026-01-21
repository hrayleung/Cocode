"""Production monitoring and metrics collection."""

from .metrics import (
    MetricsCollector,
    get_metrics,
    record_search_latency,
    record_index_operation,
    record_cache_access,
    record_embedding_request,
)

__all__ = [
    "MetricsCollector",
    "get_metrics",
    "record_cache_access",
    "record_embedding_request",
    "record_index_operation",
    "record_search_latency",
]

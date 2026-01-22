"""Metrics collection for production monitoring.

Provides lightweight metrics tracking for search latency, indexing operations,
cache performance, and embedding requests. Designed for easy integration with
Prometheus or other monitoring systems.
"""

import functools
import threading
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class LatencyStats:
    """Statistics for latency measurements.
    
    Note: This class is not thread-safe on its own. When used within
    MetricsCollector, thread-safety is provided by the parent's lock.
    """
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0

    def record(self, ms: float) -> None:
        """Record a latency measurement. Caller must hold external lock."""
        self.count += 1
        self.total_ms += ms
        self.min_ms = min(self.min_ms, ms)
        self.max_ms = max(self.max_ms, ms)

    @property
    def avg_ms(self) -> float:
        """Get average latency. Caller must hold external lock."""
        return self.total_ms / self.count if self.count > 0 else 0.0

    def to_dict(self) -> dict:
        """Export as dict. Caller must hold external lock."""
        return {
            "count": self.count,
            "avg_ms": round(self.avg_ms, 2),
            "min_ms": round(self.min_ms, 2) if self.count > 0 else 0,
            "max_ms": round(self.max_ms, 2),
        }


@dataclass
class CounterStats:
    """Simple counter with success/failure tracking."""
    success: int = 0
    failure: int = 0

    @property
    def total(self) -> int:
        return self.success + self.failure

    @property
    def success_rate(self) -> float:
        return self.success / self.total if self.total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "success": self.success,
            "failure": self.failure,
            "success_rate": round(self.success_rate, 3),
        }


@dataclass
class MetricsCollector:
    """Thread-safe metrics collector for production monitoring."""

    search_latency: LatencyStats = field(default_factory=LatencyStats)
    index_latency: LatencyStats = field(default_factory=LatencyStats)
    embedding_latency: LatencyStats = field(default_factory=LatencyStats)
    cache_hits: int = 0
    cache_misses: int = 0
    index_operations: CounterStats = field(default_factory=CounterStats)
    search_operations: CounterStats = field(default_factory=CounterStats)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def cache_hit_rate(self) -> float:
        with self._lock:
            total = self.cache_hits + self.cache_misses
            return self.cache_hits / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        """Export all metrics as a dictionary."""
        with self._lock:
            cache_hits = self.cache_hits
            cache_misses = self.cache_misses
            total = cache_hits + cache_misses
            hit_rate = round(cache_hits / total, 3) if total > 0 else 0.0
            return {
                "search": {
                    "latency": self.search_latency.to_dict(),
                    "operations": self.search_operations.to_dict(),
                },
                "index": {
                    "latency": self.index_latency.to_dict(),
                    "operations": self.index_operations.to_dict(),
                },
                "embedding": {
                    "latency": self.embedding_latency.to_dict(),
                },
                "cache": {
                    "hits": cache_hits,
                    "misses": cache_misses,
                    "hit_rate": hit_rate,
                },
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.search_latency = LatencyStats()
            self.index_latency = LatencyStats()
            self.embedding_latency = LatencyStats()
            self.cache_hits = 0
            self.cache_misses = 0
            self.index_operations = CounterStats()
            self.search_operations = CounterStats()


# Global singleton
_metrics: MetricsCollector | None = None
_metrics_lock = threading.Lock()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector singleton."""
    global _metrics
    if _metrics is None:
        with _metrics_lock:
            if _metrics is None:
                _metrics = MetricsCollector()
    return _metrics


def record_search_latency(ms: float, success: bool = True) -> None:
    """Record a search operation latency."""
    m = get_metrics()
    with m._lock:
        m.search_latency.record(ms)
        if success:
            m.search_operations.success += 1
        else:
            m.search_operations.failure += 1


def record_index_operation(ms: float, success: bool = True) -> None:
    """Record an indexing operation."""
    m = get_metrics()
    with m._lock:
        m.index_latency.record(ms)
        if success:
            m.index_operations.success += 1
        else:
            m.index_operations.failure += 1


def record_cache_access(hit: bool) -> None:
    """Record a cache access (hit or miss)."""
    m = get_metrics()
    with m._lock:
        if hit:
            m.cache_hits += 1
        else:
            m.cache_misses += 1


def record_embedding_request(ms: float) -> None:
    """Record an embedding API request latency."""
    m = get_metrics()
    with m._lock:
        m.embedding_latency.record(ms)


def timed(record_fn: Callable[[float], None]):
    """Decorator to time a function and record its latency."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                record_fn(elapsed_ms)
        return wrapper
    return decorator

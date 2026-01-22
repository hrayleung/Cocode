"""Thread-safe LRU cache with TTL for embeddings."""

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """Cache entry with value and timestamp."""
    value: list[float]
    timestamp: float


class EmbeddingCache:
    """Thread-safe LRU cache with TTL for query embeddings.
    
    Reduces redundant embedding API calls by caching query→embedding mappings.
    Uses content hash as key to handle equivalent queries.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 3600.0,  # 1 hour default
    ):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._hits = 0
        self._misses = 0

    def _make_key(self, text: str) -> str:
        """Create cache key from text content."""
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def get(self, text: str) -> list[float] | None:
        """Get cached embedding if exists and not expired."""
        key = self._make_key(text)
        now = time.monotonic()
        
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            
            # Check TTL
            if (now - entry.timestamp) > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, text: str, embedding: list[float]) -> None:
        """Cache an embedding."""
        key = self._make_key(text)
        now = time.monotonic()
        
        with self._lock:
            # Remove if exists (to update position)
            if key in self._cache:
                del self._cache[key]
            
            # Evict oldest if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            
            self._cache[key] = CacheEntry(value=embedding, timestamp=now)

    def get_stats(self) -> dict[str, int | float]:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }

    def clear(self) -> None:
        """Clear all cached embeddings."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


# Global singleton for query embedding cache
_query_embedding_cache: EmbeddingCache | None = None
_cache_lock = threading.Lock()


def get_embedding_cache() -> EmbeddingCache:
    """Get or create the global embedding cache."""
    global _query_embedding_cache
    if _query_embedding_cache is None:
        with _cache_lock:
            if _query_embedding_cache is None:
                _query_embedding_cache = EmbeddingCache()
    return _query_embedding_cache

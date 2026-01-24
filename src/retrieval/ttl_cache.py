"""Thread-safe TTL cache for database existence checks."""

import heapq
import threading
import time


class TTLCache:
    """Thread-safe cache with separate TTLs for positive and negative results."""

    def __init__(self, ttl_hit: float = 300.0, ttl_miss: float = 30.0, max_size: int = 10000):
        self._cache: dict[str, tuple[bool, float]] = {}
        self._lock = threading.Lock()
        self._ttl_hit = ttl_hit
        self._ttl_miss = ttl_miss
        self._max_size = max_size

    def get(self, key: str) -> bool | None:
        """Get cached value if not expired, else None.

        Opportunistically cleans up expired entries to prevent accumulation.
        """
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            exists, ts = cached
            ttl = self._ttl_hit if exists else self._ttl_miss
            if (now - ts) >= ttl:
                # Expired - remove from cache and clean up other expired entries
                del self._cache[key]
                self._cleanup_expired(now, max_cleanup=10)
                return None
            return exists

    def _cleanup_expired(self, now: float, max_cleanup: int = 10) -> None:
        """Opportunistically remove expired entries (internal, assumes lock held).

        Args:
            now: Current monotonic time
            max_cleanup: Maximum number of expired entries to remove
        """
        removed = 0
        # Check a sample of entries for expiration
        for k, (exists, ts) in list(self._cache.items()):
            if removed >= max_cleanup:
                break
            ttl = self._ttl_hit if exists else self._ttl_miss
            if (now - ts) >= ttl:
                del self._cache[k]
                removed += 1

    def set(self, key: str, exists: bool) -> None:
        """Cache a value with current timestamp."""
        with self._lock:
            # Evict oldest entries if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                # Remove ~10% of oldest entries using heapq for O(n log k) performance
                to_remove = max(1, self._max_size // 10)
                # Find k smallest timestamps (oldest entries)
                oldest = heapq.nsmallest(
                    to_remove,
                    self._cache.items(),
                    key=lambda item: item[1][1]  # Sort by timestamp
                )
                for k, _ in oldest:
                    del self._cache[k]
            self._cache[key] = (exists, time.monotonic())

    def invalidate(self, key: str) -> None:
        """Remove a key from cache."""
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()

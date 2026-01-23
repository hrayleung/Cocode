"""Thread-safe TTL cache for database existence checks."""

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
        """Get cached value if not expired, else None."""
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(key)
            if cached is None:
                return None
            exists, ts = cached
            ttl = self._ttl_hit if exists else self._ttl_miss
            if (now - ts) >= ttl:
                # Expired - remove from cache
                del self._cache[key]
                return None
            return exists

    def set(self, key: str, exists: bool) -> None:
        """Cache a value with current timestamp."""
        with self._lock:
            # Evict oldest entries if at capacity
            if len(self._cache) >= self._max_size and key not in self._cache:
                # Remove ~10% of oldest entries
                to_remove = max(1, self._max_size // 10)
                sorted_keys = sorted(self._cache.keys(), key=lambda k: self._cache[k][1])
                for k in sorted_keys[:to_remove]:
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

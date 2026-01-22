"""Thread-safe TTL cache for database existence checks."""

import threading
import time


class TTLCache:
    """Thread-safe cache with separate TTLs for positive and negative results."""

    def __init__(self, ttl_hit: float = 300.0, ttl_miss: float = 30.0):
        self._cache: dict[str, tuple[bool, float]] = {}
        self._lock = threading.Lock()
        self._ttl_hit = ttl_hit
        self._ttl_miss = ttl_miss

    def get(self, key: str) -> bool | None:
        """Get cached value if not expired, else None."""
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(key)
            if not cached:
                return None
            exists, ts = cached
            ttl = self._ttl_hit if exists else self._ttl_miss
            return exists if (now - ts) < ttl else None

    def set(self, key: str, exists: bool) -> None:
        """Cache a value with current timestamp."""
        with self._lock:
            self._cache[key] = (exists, time.monotonic())

    def invalidate(self, key: str) -> None:
        """Remove a key from cache."""
        with self._lock:
            self._cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached values."""
        with self._lock:
            self._cache.clear()

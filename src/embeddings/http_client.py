"""Shared HTTP client management for embedding providers."""

import atexit
import threading

import httpx


class SharedHttpClient:
    """Thread-safe singleton HTTP client with connection pooling."""

    def __init__(
        self,
        timeout: float = 120.0,
        max_connections: int = 20,
        max_keepalive: int = 10,
        keepalive_expiry: float = 30.0,
    ):
        self._client: httpx.Client | None = None
        self._lock = threading.Lock()
        self._timeout = timeout
        self._max_connections = max_connections
        self._max_keepalive = max_keepalive
        self._keepalive_expiry = keepalive_expiry
        atexit.register(self.close)

    def get(self) -> httpx.Client:
        """Get or create the shared HTTP client."""
        with self._lock:
            if self._client is None:
                self._client = httpx.Client(
                    timeout=httpx.Timeout(self._timeout),
                    limits=httpx.Limits(
                        max_connections=self._max_connections,
                        max_keepalive_connections=self._max_keepalive,
                        keepalive_expiry=self._keepalive_expiry,
                    ),
                )
            return self._client

    def close(self) -> None:
        """Close the HTTP client (best-effort)."""
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                finally:
                    self._client = None


# Shared instance for embedding providers
_embedding_client = SharedHttpClient()


def get_embedding_client() -> httpx.Client:
    """Get the shared HTTP client for embedding requests."""
    return _embedding_client.get()


def close_embedding_client() -> None:
    """Close the shared embedding HTTP client."""
    _embedding_client.close()

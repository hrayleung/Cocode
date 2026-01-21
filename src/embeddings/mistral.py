"""Mistral Codestral Embed embeddings.

Truncates embeddings to settings.embedding_dimensions when the API returns
a longer vector to maintain compatibility with the stored pgvector index.
"""

from config.settings import settings
from .http_client import get_embedding_client

MISTRAL_API_URL = "https://api.mistral.ai/v1/embeddings"


def _fit_dimensions(vec: list[float]) -> list[float]:
    """Fit embedding to configured pgvector dimension."""
    target = settings.embedding_dimensions
    if target <= 0 or len(vec) == target:
        return vec
    if len(vec) > target:
        return vec[:target]
    raise ValueError(f"Embedding dimension mismatch: got {len(vec)}, expected {target}")


def _make_request(texts: list[str], timeout: float = 120.0) -> list[dict]:
    """Make authenticated request to Mistral API."""
    if not settings.mistral_api_key:
        raise ValueError("MISTRAL_API_KEY not configured")
    response = get_embedding_client().post(
        MISTRAL_API_URL,
        headers={"Authorization": f"Bearer {settings.mistral_api_key}", "Content-Type": "application/json"},
        json={"model": settings.mistral_embed_model, "input": texts},
        timeout=timeout,
    )
    response.raise_for_status()
    return sorted(response.json()["data"], key=lambda x: x["index"])


def get_embedding(text: str) -> list[float]:
    """Get embedding for a single text."""
    return _fit_dimensions(_make_request([text], timeout=60.0)[0]["embedding"])


def get_embeddings_batch(texts: list[str], batch_size: int = 512) -> list[list[float]]:
    """Get embeddings for multiple texts."""
    if not texts:
        return []
    result = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        result.extend([_fit_dimensions(d["embedding"]) for d in _make_request(chunk)])
    return result


def is_available() -> bool:
    """Check if Mistral embeddings are configured."""
    return bool(settings.mistral_api_key)

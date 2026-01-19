"""Mistral Codestral Embed embeddings.

NOTE: Mistral embeddings have a fixed output dimension from the API.
Cocode stores embeddings in pgvector with a configured dimension
(`settings.embedding_dimensions`).

To keep retrieval compatible with the stored index, we truncate embeddings
to `settings.embedding_dimensions` when the API returns a longer vector.
"""

import httpx
from config.settings import settings

MISTRAL_API_URL = "https://api.mistral.ai/v1/embeddings"


def _fit_dimensions(vec: list[float]) -> list[float]:
    """Fit an embedding to the configured pgvector dimension."""

    target = settings.embedding_dimensions
    if target <= 0:
        return vec

    if len(vec) == target:
        return vec

    if len(vec) > target:
        return vec[:target]

    # If the model returns fewer dims than required, we must fail loudly.
    raise ValueError(f"Embedding dimension mismatch: got {len(vec)}, expected {target}")


def get_embedding(text: str) -> list[float]:
    """Get embedding for a single text."""
    if not settings.mistral_api_key:
        raise ValueError("MISTRAL_API_KEY not configured")

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            MISTRAL_API_URL,
            headers={"Authorization": f"Bearer {settings.mistral_api_key}", "Content-Type": "application/json"},
            json={
                "model": settings.mistral_embed_model,
                "input": [text],
            },
        )
        response.raise_for_status()
        vec = response.json()["data"][0]["embedding"]
        return _fit_dimensions(vec)


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Get embeddings for multiple texts."""
    if not settings.mistral_api_key:
        raise ValueError("MISTRAL_API_KEY not configured")
    if not texts:
        return []

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            MISTRAL_API_URL,
            headers={"Authorization": f"Bearer {settings.mistral_api_key}", "Content-Type": "application/json"},
            json={
                "model": settings.mistral_embed_model,
                "input": texts,
            },
        )
        response.raise_for_status()
        sorted_data = sorted(response.json()["data"], key=lambda x: x["index"])
        return [_fit_dimensions(d["embedding"]) for d in sorted_data]


def is_available() -> bool:
    """Check if Mistral embeddings are configured."""
    return bool(settings.mistral_api_key)

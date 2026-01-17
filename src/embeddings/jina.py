"""Jina embeddings wrapper with late chunking support."""

import logging
import httpx
from config.settings import settings

logger = logging.getLogger(__name__)

JINA_API_URL = "https://api.jina.ai/v1/embeddings"


def get_embedding(text: str, task: str = "retrieval.query") -> list[float]:
    """Get embedding for a single text."""
    if not settings.jina_api_key:
        raise ValueError("JINA_API_KEY not configured")

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            JINA_API_URL,
            headers={"Authorization": f"Bearer {settings.jina_api_key}", "Content-Type": "application/json"},
            json={
                "model": settings.jina_model,
                "input": [text],
                "task": task,
                "dimensions": settings.embedding_dimensions,
                "normalized": True,
            },
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]


def get_embeddings_late_chunking(chunks: list[str], task: str = "retrieval.passage") -> list[list[float]]:
    """Get embeddings for chunks using late chunking (preserves cross-chunk context)."""
    if not settings.jina_api_key:
        raise ValueError("JINA_API_KEY not configured")
    if not chunks:
        return []

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            JINA_API_URL,
            headers={"Authorization": f"Bearer {settings.jina_api_key}", "Content-Type": "application/json"},
            json={
                "model": settings.jina_model,
                "input": chunks,
                "task": task,
                "dimensions": settings.embedding_dimensions,
                "normalized": True,
                "late_chunking": True,
            },
        )
        response.raise_for_status()
        sorted_data = sorted(response.json()["data"], key=lambda x: x["index"])
        return [d["embedding"] for d in sorted_data]


def get_embeddings_batch(texts: list[str], task: str = "retrieval.passage", use_late_chunking: bool = False) -> list[list[float]]:
    """Get embeddings for multiple texts."""
    if not settings.jina_api_key:
        raise ValueError("JINA_API_KEY not configured")
    if not texts:
        return []

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            JINA_API_URL,
            headers={"Authorization": f"Bearer {settings.jina_api_key}", "Content-Type": "application/json"},
            json={
                "model": settings.jina_model,
                "input": texts,
                "task": task,
                "dimensions": settings.embedding_dimensions,
                "normalized": True,
                "late_chunking": use_late_chunking,
            },
        )
        response.raise_for_status()
        sorted_data = sorted(response.json()["data"], key=lambda x: x["index"])
        return [d["embedding"] for d in sorted_data]


def is_available() -> bool:
    """Check if Jina embeddings are configured."""
    return bool(settings.jina_api_key) and settings.use_late_chunking

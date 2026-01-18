"""Jina embeddings wrapper with late chunking support."""

import logging
import httpx
from config.settings import settings

logger = logging.getLogger(__name__)

JINA_API_URL = "https://api.jina.ai/v1/embeddings"

# Security limits to prevent resource exhaustion
MAX_TEXT_LENGTH = 50_000  # 50KB per text
MAX_BATCH_SIZE = 100  # Maximum batch size for embeddings


def get_embedding(text: str, task: str = "retrieval.query") -> list[float]:
    """Get embedding for a single text.

    Args:
        text: Text to embed (max 50KB)
        task: Task type for embedding

    Returns:
        Embedding vector

    Raises:
        ValueError: If text exceeds maximum length or API key not configured
    """
    if not settings.jina_api_key:
        raise ValueError("JINA_API_KEY not configured")

    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"Text exceeds maximum length of {MAX_TEXT_LENGTH} characters")

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
    """Get embeddings for chunks using late chunking (preserves cross-chunk context).

    Args:
        chunks: List of text chunks (each max 50KB)
        task: Task type for embedding

    Returns:
        List of embedding vectors

    Raises:
        ValueError: If chunks exceed limits or API key not configured
    """
    if not settings.jina_api_key:
        raise ValueError("JINA_API_KEY not configured")
    if not chunks:
        return []

    # Validate chunk lengths
    for i, chunk in enumerate(chunks):
        if len(chunk) > MAX_TEXT_LENGTH:
            raise ValueError(f"Chunk at index {i} exceeds maximum length of {MAX_TEXT_LENGTH} characters")

    if len(chunks) > MAX_BATCH_SIZE:
        raise ValueError(f"Number of chunks exceeds maximum of {MAX_BATCH_SIZE}")

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
    """Get embeddings for multiple texts.

    Args:
        texts: List of texts to embed (each max 50KB)
        task: Task type for embedding
        use_late_chunking: Whether to use late chunking

    Returns:
        List of embedding vectors

    Raises:
        ValueError: If texts exceed limits or API key not configured
    """
    if not settings.jina_api_key:
        raise ValueError("JINA_API_KEY not configured")
    if not texts:
        return []

    # Validate text lengths
    for i, text in enumerate(texts):
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text at index {i} exceeds maximum length of {MAX_TEXT_LENGTH} characters")

    if len(texts) > MAX_BATCH_SIZE:
        raise ValueError(f"Number of texts exceeds maximum of {MAX_BATCH_SIZE}")

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

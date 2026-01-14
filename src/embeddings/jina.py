"""Jina embeddings wrapper with late chunking support.

Late chunking embeds the full document first, then extracts chunk embeddings.
This preserves cross-chunk context, improving retrieval by ~24%.
"""

import logging

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

JINA_API_URL = "https://api.jina.ai/v1/embeddings"


def get_embedding(
    text: str,
    task: str = "retrieval.query",
) -> list[float]:
    """Get embedding for a single text (query).

    Args:
        text: Text to embed
        task: Task type - "retrieval.query" for queries, "retrieval.passage" for docs

    Returns:
        Embedding vector
    """
    if not settings.jina_api_key:
        raise ValueError("JINA_API_KEY not configured")

    with httpx.Client(timeout=60.0) as client:
        response = client.post(
            JINA_API_URL,
            headers={
                "Authorization": f"Bearer {settings.jina_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.jina_model,
                "input": [text],
                "task": task,
                "dimensions": settings.embedding_dimensions,
                "normalized": True,
            },
        )
        response.raise_for_status()
        data = response.json()
        return data["data"][0]["embedding"]


def get_embeddings_late_chunking(
    chunks: list[str],
    task: str = "retrieval.passage",
) -> list[list[float]]:
    """Get embeddings for chunks using late chunking.

    Late chunking concatenates all chunks, embeds as a single document,
    then returns individual chunk embeddings. This preserves cross-chunk
    context for better retrieval quality.

    Args:
        chunks: List of text chunks from the SAME document
        task: Task type - typically "retrieval.passage" for indexing

    Returns:
        List of embedding vectors, one per chunk
    """
    if not settings.jina_api_key:
        raise ValueError("JINA_API_KEY not configured")

    if not chunks:
        return []

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            JINA_API_URL,
            headers={
                "Authorization": f"Bearer {settings.jina_api_key}",
                "Content-Type": "application/json",
            },
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
        data = response.json()

        # Sort by index to maintain order
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [d["embedding"] for d in sorted_data]


def get_embeddings_batch(
    texts: list[str],
    task: str = "retrieval.passage",
    use_late_chunking: bool = False,
) -> list[list[float]]:
    """Get embeddings for multiple texts.

    Args:
        texts: List of texts to embed
        task: Task type
        use_late_chunking: If True, use late chunking (only for same-document chunks)

    Returns:
        List of embedding vectors
    """
    if not settings.jina_api_key:
        raise ValueError("JINA_API_KEY not configured")

    if not texts:
        return []

    with httpx.Client(timeout=120.0) as client:
        response = client.post(
            JINA_API_URL,
            headers={
                "Authorization": f"Bearer {settings.jina_api_key}",
                "Content-Type": "application/json",
            },
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
        data = response.json()

        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [d["embedding"] for d in sorted_data]


def is_available() -> bool:
    """Check if Jina embeddings are configured and available."""
    return bool(settings.jina_api_key) and settings.use_late_chunking

"""Jina embeddings wrapper with late chunking support."""

import logging

from config.settings import settings
from .http_client import get_embedding_client

logger = logging.getLogger(__name__)

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
MAX_TEXT_LENGTH = 50_000
MAX_BATCH_SIZE = 100


def _make_request(payload: dict, timeout: float = 120.0) -> dict:
    """Make authenticated request to Jina API."""
    if not settings.jina_api_key:
        raise ValueError("JINA_API_KEY not configured")
    response = get_embedding_client().post(
        JINA_API_URL,
        headers={"Authorization": f"Bearer {settings.jina_api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def _validate_texts(texts: list[str], context: str = "Text") -> None:
    """Validate text lengths."""
    for i, text in enumerate(texts):
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"{context} at index {i} exceeds maximum length of {MAX_TEXT_LENGTH} characters")


def _extract_embeddings(data: list[dict]) -> list[list[float]]:
    """Extract embeddings from API response, sorted by index."""
    return [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]


def get_embedding(text: str, task: str = "retrieval.query") -> list[float]:
    """Get embedding for a single text."""
    _validate_texts([text])
    data = _make_request({
        "model": settings.jina_model,
        "input": [text],
        "task": task,
        "dimensions": settings.embedding_dimensions,
        "normalized": True,
    }, timeout=60.0)
    return data["data"][0]["embedding"]


def get_embeddings_late_chunking(chunks: list[str], task: str = "retrieval.passage") -> list[list[float]]:
    """Get embeddings using late chunking (preserves cross-chunk context)."""
    if not chunks:
        return []

    total_chars = sum(len(c) for c in chunks)
    if total_chars > JINA_LATE_CHUNKING_CHAR_LIMIT:
        raise ValueError(
            f"Batch text length ({total_chars} chars) exceeds late_chunking limit "
            f"({JINA_LATE_CHUNKING_CHAR_LIMIT} chars). Pass shorter texts or disable late chunking."
        )

    _validate_texts(chunks, "Chunk")

    data = _make_request({
        "model": settings.jina_model,
        "input": chunks,
        "task": task,
        "dimensions": settings.embedding_dimensions,
        "normalized": True,
        "late_chunking": True,
    })
    return _extract_embeddings(data["data"])


# Approximate token limit for Jina late chunking (8192 tokens ~ 30K chars per request)
JINA_LATE_CHUNKING_CHAR_LIMIT = 30000


def get_embeddings_batch(texts: list[str], task: str = "retrieval.passage", use_late_chunking: bool = False) -> list[list[float]]:
    """Get embeddings for multiple texts."""
    if not texts:
        return []
    _validate_texts(texts)
    
    result = []
    for i in range(0, len(texts), MAX_BATCH_SIZE):
        batch = texts[i : i + MAX_BATCH_SIZE]
        
        # Validate per-batch length for late chunking (limit applies per request)
        if use_late_chunking:
            batch_chars = sum(len(t) for t in batch)
            if batch_chars > JINA_LATE_CHUNKING_CHAR_LIMIT:
                raise ValueError(
                    f"Batch text length ({batch_chars} chars) exceeds late_chunking limit "
                    f"({JINA_LATE_CHUNKING_CHAR_LIMIT} chars). Pass shorter texts or disable late chunking."
                )
        
        data = _make_request({
            "model": settings.jina_model,
            "input": batch,
            "task": task,
            "dimensions": settings.embedding_dimensions,
            "normalized": True,
            "late_chunking": use_late_chunking,
        })
        result.extend(_extract_embeddings(data["data"]))
    return result


def is_available() -> bool:
    """Check if Jina embeddings are configured."""
    return bool(settings.jina_api_key) and settings.use_late_chunking

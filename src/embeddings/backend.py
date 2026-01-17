"""Shared embedding-backend selection.

This module centralizes the decision of whether to use Jina or OpenAI embeddings
so indexing and query paths stay in the same embedding space.
"""

import logging
import threading

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

JINA_API_URL = "https://api.jina.ai/v1/embeddings"

_use_jina_cached: bool | None = None
_use_jina_lock = threading.Lock()


def should_use_jina() -> bool:
    """Return True iff Jina embeddings are configured and validated.

    The result is cached process-wide to keep indexing and query consistent.
    """
    if not settings.jina_api_key or not settings.use_late_chunking:
        return False

    global _use_jina_cached
    if _use_jina_cached is not None:
        return _use_jina_cached

    with _use_jina_lock:
        if _use_jina_cached is not None:
            return _use_jina_cached
        _use_jina_cached = _validate_jina()
        return _use_jina_cached


def _validate_jina() -> bool:
    try:
        response = httpx.post(
            JINA_API_URL,
            headers={
                "Authorization": f"Bearer {settings.jina_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.jina_model,
                "input": ["test"],
                "task": "retrieval.query",
                "dimensions": settings.embedding_dimensions,
                "normalized": True,
            },
            timeout=10.0,
        )

        if response.status_code == 403:
            logger.warning("Jina API key is invalid (403 Forbidden). Falling back to OpenAI.")
            return False
        if response.status_code != 200:
            logger.warning(
                f"Jina API returned {response.status_code}. Falling back to OpenAI."
            )
            return False

        return True
    except Exception as e:
        logger.warning(f"Jina API validation failed: {e}. Falling back to OpenAI.")
        return False

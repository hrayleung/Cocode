"""Shared embedding-backend selection.

This module centralizes the decision of which embedding provider to use
so indexing and query paths stay in the same embedding space.
"""

import logging
import threading
from typing import Literal

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
MISTRAL_API_URL = "https://api.mistral.ai/v1/embeddings"

_selected_provider: Literal["jina", "mistral", "openai"] | None = None
_provider_lock = threading.Lock()


def get_selected_provider() -> Literal["jina", "mistral", "openai"]:
    """Get the selected embedding provider, validating API keys."""
    global _selected_provider
    if _selected_provider is not None:
        return _selected_provider

    with _provider_lock:
        if _selected_provider is not None:
            return _selected_provider

        requested = settings.embedding_provider.lower()

        if requested == "mistral" and _validate_mistral():
            _selected_provider = "mistral"
        elif requested == "jina" and settings.use_late_chunking and _validate_jina():
            _selected_provider = "jina"
        else:
            _selected_provider = "openai"

        logger.info(f"Using {_selected_provider} embeddings")
        return _selected_provider


def get_embedding_provider():
    """Get the embedding provider instance."""
    from src.embeddings.provider import JinaProvider, MistralProvider, OpenAIProvider

    provider = get_selected_provider()

    providers = {
        "mistral": MistralProvider,
        "jina": JinaProvider,
        "openai": OpenAIProvider,
    }
    return providers.get(provider, OpenAIProvider)()


def should_use_jina() -> bool:
    """Return True iff Jina is the selected provider."""
    return get_selected_provider() == "jina"


def should_use_mistral() -> bool:
    """Return True iff Mistral is the selected provider."""
    return get_selected_provider() == "mistral"


def _validate_jina() -> bool:
    if not settings.jina_api_key:
        return False
    try:
        response = httpx.post(
            JINA_API_URL,
            headers={"Authorization": f"Bearer {settings.jina_api_key}", "Content-Type": "application/json"},
            json={"model": settings.jina_model, "input": ["test"], "task": "retrieval.query", "dimensions": settings.embedding_dimensions, "normalized": True},
            timeout=10.0,
        )
        if response.status_code != 200:
            logger.warning(f"Jina API returned {response.status_code}")
            return False
        return True
    except Exception as e:
        logger.warning(f"Jina API validation failed: {e}")
        return False


def _validate_mistral() -> bool:
    if not settings.mistral_api_key:
        return False
    try:
        response = httpx.post(
            MISTRAL_API_URL,
            headers={"Authorization": f"Bearer {settings.mistral_api_key}", "Content-Type": "application/json"},
            json={"model": settings.mistral_embed_model, "input": ["test"]},
            timeout=10.0,
        )
        if response.status_code != 200:
            logger.warning(f"Mistral API returned {response.status_code}")
            return False
        return True
    except Exception as e:
        logger.warning(f"Mistral API validation failed: {e}")
        return False

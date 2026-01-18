"""Shared embedding-backend selection."""

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


def _validate_api(url: str, api_key: str, payload: dict) -> bool:
    """Validate an embedding API by making a test request."""
    if not api_key:
        return False

    try:
        response = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=10.0,
        )
        if response.status_code == 200:
            return True
        logger.warning(f"API at {url} returned {response.status_code}")
        return False
    except Exception as e:
        logger.warning(f"API validation failed for {url}: {e}")
        return False


def get_selected_provider() -> Literal["jina", "mistral", "openai"]:
    """Get the selected embedding provider, validating API keys."""
    global _selected_provider
    if _selected_provider:
        return _selected_provider

    with _provider_lock:
        if _selected_provider:
            return _selected_provider

        requested = settings.embedding_provider.lower()

        if requested == "mistral" and _validate_api(
            MISTRAL_API_URL, settings.mistral_api_key,
            {"model": settings.mistral_embed_model, "input": ["test"]}
        ):
            _selected_provider = "mistral"
        elif requested == "jina" and settings.use_late_chunking and _validate_api(
            JINA_API_URL, settings.jina_api_key,
            {"model": settings.jina_model, "input": ["test"], "task": "retrieval.query",
             "dimensions": settings.embedding_dimensions, "normalized": True}
        ):
            _selected_provider = "jina"
        else:
            _selected_provider = "openai"

        logger.info(f"Using {_selected_provider} embeddings")
        return _selected_provider


def get_embedding_provider():
    """Get the embedding provider instance."""
    from src.embeddings.provider import JinaProvider, MistralProvider, OpenAIProvider

    providers = {"mistral": MistralProvider, "jina": JinaProvider, "openai": OpenAIProvider}
    return providers.get(get_selected_provider(), OpenAIProvider)()


def should_use_jina() -> bool:
    return get_selected_provider() == "jina"


def should_use_mistral() -> bool:
    return get_selected_provider() == "mistral"

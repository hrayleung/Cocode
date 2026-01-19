"""Embedding provider selection and validation.

This module handles automatic selection and validation of embedding providers
(Jina, Mistral, or OpenAI) based on configuration and API availability.
"""

import logging
import threading
from typing import Literal

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

# API endpoints for embedding providers
JINA_API_URL = "https://api.jina.ai/v1/embeddings"
MISTRAL_API_URL = "https://api.mistral.ai/v1/embeddings"

# Thread-safe singleton for provider selection
_selected_provider: Literal["jina", "mistral", "openai"] | None = None
_provider_lock = threading.Lock()


def _validate_api(url: str, api_key: str, payload: dict) -> bool:
    """Validate an embedding API by making a test request.

    Args:
        url: API endpoint URL
        api_key: API key for authentication
        payload: Test payload to send

    Returns:
        True if API is accessible and responds successfully
    """
    if not api_key:
        return False

    try:
        response = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=10.0,
        )
        return response.status_code == 200
    except Exception as e:
        logger.warning(f"API validation failed for {url}: {e}")
        return False


def get_selected_provider() -> Literal["jina", "mistral", "openai"]:
    """Get the selected embedding provider, validating API keys on first call.

    Selection priority (based on EMBEDDING_PROVIDER setting):
    1. Mistral Codestral Embed (if configured and available)
    2. Jina with late chunking (if configured and available)
    3. OpenAI (fallback)

    Returns:
        Provider name: "jina", "mistral", or "openai"
    """
    global _selected_provider

    if _selected_provider is not None:
        return _selected_provider

    with _provider_lock:
        if _selected_provider is not None:
            return _selected_provider

        requested = settings.embedding_provider.lower()

        # Try Mistral if requested
        if requested == "mistral":
            mistral_payload = {"model": settings.mistral_embed_model, "input": ["test"]}
            if _validate_api(MISTRAL_API_URL, settings.mistral_api_key, mistral_payload):
                _selected_provider = "mistral"
                logger.info("Using Mistral embeddings")
                return _selected_provider

        # Try Jina if requested (requires late chunking enabled)
        if requested == "jina" and settings.use_late_chunking:
            jina_payload = {
                "model": settings.jina_model,
                "input": ["test"],
                "task": "retrieval.query",
                "dimensions": settings.embedding_dimensions,
                "normalized": True
            }
            if _validate_api(JINA_API_URL, settings.jina_api_key, jina_payload):
                _selected_provider = "jina"
                logger.info("Using Jina embeddings with late chunking")
                return _selected_provider

        # Fall back to OpenAI
        _selected_provider = "openai"
        logger.info("Using OpenAI embeddings")
        return _selected_provider


def get_embedding_provider():
    """Get the embedding provider instance.

    Returns:
        Configured embedding provider (JinaProvider, MistralProvider, or OpenAIProvider)
    """
    from src.embeddings.provider import JinaProvider, MistralProvider, OpenAIProvider

    providers = {
        "mistral": MistralProvider,
        "jina": JinaProvider,
        "openai": OpenAIProvider
    }
    provider_class = providers.get(get_selected_provider(), OpenAIProvider)
    return provider_class()


def should_use_jina() -> bool:
    """Check if Jina is the selected embedding provider."""
    return get_selected_provider() == "jina"


def should_use_mistral() -> bool:
    """Check if Mistral is the selected embedding provider."""
    return get_selected_provider() == "mistral"

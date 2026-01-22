"""Embedding provider protocol and implementations.

This module defines the EmbeddingProvider protocol and concrete implementations
for OpenAI, Jina, and Mistral embedding services.
"""

import threading
from typing import Protocol


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers.

    All embedding providers must implement these methods to be used
    interchangeably throughout the codebase.
    """

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for a single text."""
        ...

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embedding vectors for multiple texts."""
        ...


# Model registry for reference (dimensions used by settings.embedding_dimensions)
MODEL_DIMENSIONS: dict[str, dict[str, int]] = {
    "openai": {"text-embedding-3-large": 3072, "text-embedding-3-small": 1536, "text-embedding-ada-002": 1536},
    "jina": {"jina-embeddings-v3": 1024, "jina-embeddings-v2-base-code": 768},
    "mistral": {"codestral-embed": 1024},
}


class OpenAIProvider:
    """OpenAI text-embedding-3-large provider."""

    def get_embedding(self, text: str) -> list[float]:
        from src.embeddings.openai import get_embedding
        return get_embedding(text)

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        from src.embeddings.openai import get_embeddings_batch
        return get_embeddings_batch(texts)


class JinaProvider:
    """Jina embedding provider with late chunking support."""

    def get_embedding(self, text: str) -> list[float]:
        from src.embeddings.jina import get_embedding
        return get_embedding(text)

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        from src.embeddings.jina import get_embeddings_batch
        return get_embeddings_batch(texts, use_late_chunking=True)


class MistralProvider:
    """Mistral Codestral Embed provider optimized for code."""

    def get_embedding(self, text: str) -> list[float]:
        from src.embeddings.mistral import get_embedding
        return get_embedding(text)

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        from src.embeddings.mistral import get_embeddings_batch
        return get_embeddings_batch(texts)


# Thread-safe singleton for provider instance
_provider: EmbeddingProvider | None = None
_provider_lock = threading.Lock()


def get_provider() -> EmbeddingProvider:
    """Get the configured embedding provider (thread-safe singleton).

    Returns:
        Configured embedding provider instance
    """
    global _provider

    if _provider is not None:
        return _provider

    with _provider_lock:
        if _provider is None:
            from src.embeddings.backend import get_embedding_provider
            _provider = get_embedding_provider()

    return _provider

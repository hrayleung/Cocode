"""Embedding provider protocol and implementations."""

import threading
from typing import Protocol


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    def get_embedding(self, text: str) -> list[float]: ...
    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIProvider:
    """OpenAI embedding provider."""
    def get_embedding(self, text: str) -> list[float]:
        from src.embeddings.openai import get_embedding
        return get_embedding(text)

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        from src.embeddings.openai import get_embeddings_batch
        return get_embeddings_batch(texts)


class JinaProvider:
    """Jina embedding provider with late chunking."""
    def get_embedding(self, text: str) -> list[float]:
        from src.embeddings.jina import get_embedding
        return get_embedding(text)

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        from src.embeddings.jina import get_embeddings_batch
        return get_embeddings_batch(texts, use_late_chunking=True)


_provider: EmbeddingProvider | None = None
_provider_lock = threading.Lock()


def get_provider() -> EmbeddingProvider:
    """Get the configured embedding provider (thread-safe singleton)."""
    global _provider
    if _provider is not None:
        return _provider

    with _provider_lock:
        if _provider is None:
            from config.settings import settings
            _provider = JinaProvider() if (settings.use_late_chunking and settings.jina_api_key) else OpenAIProvider()
    return _provider

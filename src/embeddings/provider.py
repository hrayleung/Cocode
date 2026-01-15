"""Embedding provider protocol and implementations."""

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text."""
        ...

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts."""
        ...


class OpenAIProvider:
    """OpenAI embedding provider."""

    def get_embedding(self, text: str) -> list[float]:
        from src.embeddings.openai import get_embedding
        return get_embedding(text)

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        from src.embeddings.openai import get_embeddings_batch
        return get_embeddings_batch(texts)


class JinaProvider:
    """Jina embedding provider."""

    def get_embedding(self, text: str) -> list[float]:
        from src.embeddings.jina import get_embedding
        return get_embedding(text)

    def get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        from src.embeddings.jina import get_embeddings_batch
        return get_embeddings_batch(texts, use_late_chunking=True)


_provider: EmbeddingProvider | None = None


def get_provider() -> EmbeddingProvider:
    """Get the configured embedding provider."""
    global _provider
    if _provider is None:
        from config.settings import settings
        if settings.use_late_chunking and settings.jina_api_key:
            _provider = JinaProvider()
        else:
            _provider = OpenAIProvider()
    return _provider

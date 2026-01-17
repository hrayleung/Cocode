"""Embeddings module."""

from .openai import get_embedding, get_embeddings_batch
from . import jina
from .provider import EmbeddingProvider, OpenAIProvider, JinaProvider, get_provider

__all__ = [
    "get_embedding", "get_embeddings_batch", "jina",
    "EmbeddingProvider", "OpenAIProvider", "JinaProvider", "get_provider",
]

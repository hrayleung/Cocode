"""Embeddings module for OpenAI and Jina embeddings."""

from .openai import get_embedding, get_embeddings_batch
from . import jina

__all__ = ["get_embedding", "get_embeddings_batch", "jina"]

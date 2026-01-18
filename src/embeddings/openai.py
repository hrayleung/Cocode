"""OpenAI embeddings wrapper."""

import openai
from config.settings import settings

_client: openai.OpenAI | None = None

# Security limits to prevent resource exhaustion
MAX_TEXT_LENGTH = 50_000  # 50KB per text
MAX_BATCH_SIZE = 100  # Maximum batch size for embeddings


def get_client() -> openai.OpenAI:
    """Get or create the OpenAI client."""
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=settings.openai_api_key)
    return _client


def get_embedding(text: str, add_query_context: bool = False) -> list[float]:
    """Get embedding for a single text.

    Args:
        text: Text to embed (max 50KB)
        add_query_context: Whether to add query context

    Returns:
        Embedding vector

    Raises:
        ValueError: If text exceeds maximum length
    """
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"Text exceeds maximum length of {MAX_TEXT_LENGTH} characters")

    if add_query_context:
        text = f"# Code search query\n\n{text}"

    response = get_client().embeddings.create(
        model=settings.embedding_model,
        input=text,
        dimensions=settings.embedding_dimensions,
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Get embeddings for multiple texts in batches.

    Args:
        texts: List of texts to embed
        batch_size: Batch size (max 100)

    Returns:
        List of embedding vectors

    Raises:
        ValueError: If batch_size exceeds maximum or texts contain oversized content
    """
    if batch_size > MAX_BATCH_SIZE:
        raise ValueError(f"Batch size exceeds maximum of {MAX_BATCH_SIZE}")

    # Validate text lengths
    for i, text in enumerate(texts):
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text at index {i} exceeds maximum length of {MAX_TEXT_LENGTH} characters")

    client = get_client()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
            dimensions=settings.embedding_dimensions,
        )
        sorted_data = sorted(response.data, key=lambda x: x.index)
        all_embeddings.extend([d.embedding for d in sorted_data])

    return all_embeddings

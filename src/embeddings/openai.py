"""OpenAI embeddings wrapper."""

import openai

from config.settings import settings

# Initialize client
_client: openai.OpenAI | None = None


def get_client() -> openai.OpenAI:
    """Get or create the OpenAI client."""
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=settings.openai_api_key)
    return _client


def get_embedding(text: str, add_query_context: bool = False) -> list[float]:
    """Get embedding for a single text.

    Args:
        text: Text to embed
        add_query_context: If True, prepend query context for code search
    """
    if add_query_context:
        # Add context prefix for code search queries (matches contextual chunks)
        text = f"# Code search query\n\n{text}"

    client = get_client()
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=text,
        dimensions=settings.embedding_dimensions,
    )
    return response.data[0].embedding


def get_embeddings_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Get embeddings for multiple texts in batches.

    Args:
        texts: List of texts to embed
        batch_size: Maximum texts per API call (OpenAI limit is 2048)

    Returns:
        List of embedding vectors
    """
    client = get_client()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
            dimensions=settings.embedding_dimensions,
        )
        # Sort by index to maintain order
        sorted_data = sorted(response.data, key=lambda x: x.index)
        all_embeddings.extend([d.embedding for d in sorted_data])

    return all_embeddings

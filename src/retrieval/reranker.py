"""Cohere reranking for search results."""

import logging

import cohere

from config.settings import settings

from .vector_search import SearchResult

logger = logging.getLogger(__name__)

# Initialize client lazily
_client: cohere.Client | None = None


def get_client() -> cohere.Client:
    """Get or create the Cohere client."""
    global _client
    if _client is None:
        _client = cohere.Client(api_key=settings.cohere_api_key)
    return _client


def rerank_results(
    query: str,
    results: list[SearchResult],
    top_n: int = 10,
) -> list[SearchResult]:
    """Rerank search results using Cohere reranker.

    Args:
        query: Original search query
        results: Search results to rerank
        top_n: Number of results to return after reranking

    Returns:
        Reranked search results
    """
    if not results:
        return []

    if not settings.cohere_api_key:
        # No API key, return original results
        return results[:top_n]

    client = get_client()

    # Prepare documents for reranking
    documents = [r.content for r in results]

    try:
        response = client.rerank(
            model=settings.rerank_model,
            query=query,
            documents=documents,
            top_n=min(top_n, len(results)),
            return_documents=False,
        )

        # Map back to SearchResult objects
        reranked = []
        for item in response.results:
            original = results[item.index]
            reranked.append(
                SearchResult(
                    filename=original.filename,
                    location=original.location,
                    content=original.content,
                    score=item.relevance_score,
                )
            )

        return reranked

    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return results[:top_n]

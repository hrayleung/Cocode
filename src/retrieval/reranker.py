"""Cohere reranking for search results."""

import logging
import threading

import cohere
from config.settings import settings
from src.models import SearchResult

logger = logging.getLogger(__name__)

_client: cohere.Client | None = None
_client_lock = threading.Lock()


def get_client() -> cohere.Client:
    """Get or create the Cohere client (thread-safe singleton)."""
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            _client = cohere.Client(api_key=settings.cohere_api_key)
    return _client


def rerank_results(query: str, results: list[SearchResult], top_n: int = 10) -> list[SearchResult]:
    """Rerank search results using Cohere reranker."""
    if not results:
        return []
    if not settings.cohere_api_key:
        return results[:top_n]

    try:
        response = get_client().rerank(
            model=settings.rerank_model,
            query=query,
            documents=[r.content for r in results],
            top_n=min(top_n, len(results)),
            return_documents=False,
        )

        return [
            SearchResult(
                filename=results[item.index].filename,
                location=results[item.index].location,
                content=results[item.index].content,
                score=item.relevance_score,
            )
            for item in response.results
        ]
    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return results[:top_n]

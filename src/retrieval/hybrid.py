"""Hybrid search combining vector and BM25 with Reciprocal Rank Fusion.

This module implements the core hybrid search pipeline that combines:
- Vector similarity search (semantic understanding)
- BM25 full-text search (keyword matching)
- Symbol-level search (function/class awareness)

Results are combined using Reciprocal Rank Fusion (RRF) and optionally
reranked using Cohere's reranker for improved relevance.
"""

import logging
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from config.settings import settings
from src.embeddings import get_provider, OpenAIProvider
from src.exceptions import SearchError
from src.models import SearchResult

from .bm25_search import BM25Config, bm25_search
from .centrality import get_centrality_scores
from .file_categorizer import apply_category_boosting
from .reranker import rerank_results
from .symbol_search import symbol_hybrid_search
from .vector_search import vector_search

logger = logging.getLogger(__name__)

_SEARCH_EXECUTOR: ThreadPoolExecutor | None = None
_SEARCH_EXECUTOR_LOCK = threading.Lock()

# Parallel backends are I/O-heavy (DB + embedding HTTP). Keep the pool modest to
# avoid excess memory/thread overhead while still allowing concurrent queries.
_SEARCH_EXECUTOR_MAX_WORKERS = 8


def _get_search_executor() -> ThreadPoolExecutor:
    global _SEARCH_EXECUTOR
    if _SEARCH_EXECUTOR is not None:
        return _SEARCH_EXECUTOR
    with _SEARCH_EXECUTOR_LOCK:
        if _SEARCH_EXECUTOR is None:
            _SEARCH_EXECUTOR = ThreadPoolExecutor(
                max_workers=_SEARCH_EXECUTOR_MAX_WORKERS,
                thread_name_prefix="cocode-search",
            )
    return _SEARCH_EXECUTOR


def apply_centrality_boost(
    results: list[SearchResult],
    repo_name: str,
    weight: float = 1.0,
) -> None:
    """Apply centrality-based boosting to search results in-place.

    Files that are imported by many others (structurally central) get
    higher scores. Peripheral files (tests, scripts) get lower scores.

    Args:
        results: Search results to modify (in-place)
        repo_name: Repository name for centrality lookup
        weight: Boost strength (0=disabled, 1=full effect)
    """
    if not results or weight == 0:
        return

    filenames = list({r.filename for r in results})
    scores = get_centrality_scores(repo_name, filenames)

    for result in results:
        centrality = scores.get(result.filename, 1.0)
        result.score *= 1.0 + (centrality - 1.0) * weight


def get_query_embedding(query: str) -> list[float]:
    """Get embedding for a search query using the configured provider.

    Args:
        query: Search query text

    Returns:
        Embedding vector for the query
    """
    from src.embeddings.openai import get_embedding as get_embedding_openai

    provider = get_provider()
    if isinstance(provider, OpenAIProvider):
        return get_embedding_openai(query, add_query_context=True)
    return provider.get_embedding(query)


def _normalize_rrf_weights(weights: list[float] | None, n: int) -> list[float]:
    if weights is None:
        return [1.0] * n
    total = sum(weights)
    if total > 0:
        return [w / total for w in weights]
    return [1.0 / n] * n


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 40,
    weights: list[float] | None = None,
) -> list[SearchResult]:
    """Combine multiple result lists using Reciprocal Rank Fusion (RRF).

    This is result-level RRF: it fuses identical *results* (filename+location+content).
    Use `reciprocal_rank_fusion_by_key` when you want to fuse by a higher-level key
    such as filename (file-level evidence aggregation).
    """
    if not result_lists:
        return []

    weights = _normalize_rrf_weights(weights, len(result_lists))

    from src.rust_bridge import reciprocal_rank_fusion_weighted as rust_rrf

    rust_input = []
    key_to_result: dict[str, SearchResult] = {}

    for results in result_lists:
        rust_list = []
        for rank, result in enumerate(results):
            key = f"{result.filename}:{result.location}:{hash(result.content.strip())}"
            rust_list.append((key, result.score))
            if key not in key_to_result or result.score > key_to_result[key].score:
                key_to_result[key] = result
        rust_input.append(rust_list)

    fused_scores = rust_rrf(rust_input, weights, k=float(k))

    return [
        SearchResult(
            filename=key_to_result[key].filename,
            location=key_to_result[key].location,
            content=key_to_result[key].content,
            score=score,
        )
        for key, score in fused_scores
    ]


def reciprocal_rank_fusion_by_key(
    result_lists: list[list[SearchResult]],
    *,
    key_fn: Callable[[SearchResult], str],
    k: int = 40,
    weights: list[float] | None = None,
) -> list[SearchResult]:
    """RRF that fuses by an arbitrary key (e.g., filename for file-level ranking).

    This is useful when different backends surface different *spans* from the same file.
    Fusing by filename rewards cross-signal agreement, which tends to improve "key file"
    selection compared to max-chunk scoring.
    """
    if not result_lists:
        return []

    weights = _normalize_rrf_weights(weights, len(result_lists))

    from src.rust_bridge import reciprocal_rank_fusion_weighted as rust_rrf

    best_rep: dict[str, SearchResult] = {}
    best_rank: dict[str, int] = {}
    best_weight: dict[str, float] = {}

    rust_input = []
    for weight, results in zip(weights, result_lists, strict=True):
        rust_list = []
        for rank, result in enumerate(results):
            key = key_fn(result)
            rust_list.append((key, result.score))

            if (
                key not in best_rep
                or rank < best_rank[key]
                or (rank == best_rank[key] and weight > best_weight[key])
            ):
                best_rep[key] = result
                best_rank[key] = rank
                best_weight[key] = weight
        rust_input.append(rust_list)

    fused_scores = rust_rrf(rust_input, weights, k=float(k))

    return [
        SearchResult(
            filename=best_rep[key].filename,
            location=best_rep[key].location,
            content=best_rep[key].content,
            score=score,
        )
        for key, score in fused_scores
    ]


@dataclass
class SearchOutcome:
    """Result of a search backend execution."""

    results: list[SearchResult] = field(default_factory=list)
    failed: bool = False
    attempted: bool = True


class _EmbeddingCache:
    """Thread-safe lazy cache for query embedding.

    Ensures the embedding is computed only once, even when multiple
    search backends request it concurrently.
    """

    def __init__(self, query: str, initial_embedding: list[float] | None = None):
        self._query = query
        self._embedding: list[float] | None = (
            initial_embedding if initial_embedding else None
        )
        self._error: Exception | None = None
        self._lock = threading.Lock()

    def get(self) -> list[float]:
        """Get the embedding, computing it on first access."""
        if self._error:
            raise self._error
        if self._embedding is not None:
            return self._embedding

        with self._lock:
            if self._error:
                raise self._error
            if self._embedding is None:
                try:
                    self._embedding = get_query_embedding(self._query)
                except Exception as e:
                    self._error = e
                    raise
            return self._embedding


def _execute_search(
    search_fn: Callable[[], list[SearchResult]],
    name: str,
) -> SearchOutcome:
    """Execute a search function with error handling.

    Args:
        search_fn: Search function to execute
        name: Name for logging purposes

    Returns:
        SearchOutcome with results or failure status
    """
    try:
        return SearchOutcome(results=search_fn())
    except Exception as e:
        logger.warning(f"{name} search failed: {e}")
        return SearchOutcome(failed=True)


def _run_searches_parallel(
    searches: list[tuple[str, Callable[[], list[SearchResult]]]],
) -> dict[str, SearchOutcome]:
    """Run searches in parallel and return outcomes by name.

    Args:
        searches: List of (name, search_function) tuples

    Returns:
        Dictionary mapping search names to outcomes
    """
    outcomes = {}
    executor = _get_search_executor()
    futures = {name: executor.submit(fn) for name, fn in searches}
    for name, future in futures.items():
        try:
            outcomes[name] = SearchOutcome(results=future.result(timeout=30))
        except Exception as e:
            logger.warning(f"{name} search failed: {e}")
            outcomes[name] = SearchOutcome(failed=True)
    return outcomes


def hybrid_search(
    repo_name: str,
    query: str,
    top_k: int = 10,
    use_reranker: bool = True,
    rerank_candidates: int | None = None,
    vector_weight: float | None = None,
    bm25_weight: float | None = None,
    symbol_weight: float | None = None,
    parallel: bool = True,
    include_symbols: bool = True,
    query_embedding: list[float] | None = None,
) -> list[SearchResult]:
    """Perform hybrid search combining vector, BM25, and symbol searches.

    Pipeline:
    1. Run vector, BM25, and symbol searches in parallel
    2. Combine results using Reciprocal Rank Fusion (RRF)
    3. Apply centrality boosting (important files rank higher)
    4. Apply category boosting (implementation > test)
    5. Optionally rerank with Cohere

    Args:
        repo_name: Repository to search
        query: Search query
        top_k: Number of results to return
        use_reranker: Whether to use Cohere reranking
        rerank_candidates: Number of candidates for reranking
        vector_weight: Weight for vector search (default from settings)
        bm25_weight: Weight for BM25 search (default from settings)
        symbol_weight: Weight for symbol search (default from settings)
        parallel: Whether to run searches in parallel
        include_symbols: Whether to include symbol search

    Returns:
        List of SearchResult sorted by relevance

    Raises:
        SearchError: If all search backends fail
    """
    rerank_count = rerank_candidates or settings.rerank_candidates
    vec_w = vector_weight if vector_weight is not None else settings.vector_weight
    bm25_w = bm25_weight if bm25_weight is not None else settings.bm25_weight
    sym_w = symbol_weight if symbol_weight is not None else settings.symbol_weight

    embedding_cache = _EmbeddingCache(query, initial_embedding=query_embedding)
    bm25_config = BM25Config(k1=settings.bm25_k1, b=settings.bm25_b)
    include_sym = include_symbols and settings.enable_symbol_indexing

    def run_vector():
        return vector_search(repo_name, query, top_k=rerank_count, query_embedding=embedding_cache.get())

    def run_bm25():
        return bm25_search(repo_name, query, top_k=rerank_count, config=bm25_config)

    def run_symbol():
        return symbol_hybrid_search(repo_name, query, top_k=rerank_count, query_embedding=embedding_cache.get())

    # Execute searches
    searches = [("vector", run_vector), ("bm25", run_bm25)]
    if include_sym:
        searches.append(("symbol", run_symbol))

    if parallel:
        outcomes = _run_searches_parallel(searches)
    else:
        outcomes = {name: _execute_search(fn, name.capitalize()) for name, fn in searches}

    if all(outcomes[name].failed for name in outcomes):
        raise SearchError("All search backends failed")

    for name, outcome in outcomes.items():
        if outcome.failed:
            logger.warning(f"{name.capitalize()} search failed, using other backends")

    logger.info("Search results: " + ", ".join(f"{k}={len(v.results)}" for k, v in outcomes.items()))

    # Fuse results using RRF (file-level fusion so different spans from the same
    # file reinforce each other across backends).
    result_lists = [outcomes["vector"].results, outcomes["bm25"].results]
    weights = [vec_w, bm25_w]
    if include_sym and outcomes.get("symbol") and outcomes["symbol"].results:
        result_lists.append(outcomes["symbol"].results)
        weights.append(sym_w)

    candidates = reciprocal_rank_fusion_by_key(
        result_lists,
        key_fn=lambda r: r.filename,
        weights=weights,
    )[:rerank_count]

    # Apply boosting (used for both candidate selection and final ordering)
    if settings.centrality_weight > 0:
        apply_centrality_boost(candidates, repo_name, settings.centrality_weight)

    apply_category_boosting(candidates, sort=True)

    # Optional reranking with Cohere
    if use_reranker and settings.cohere_api_key and candidates:
        logger.debug(f"Reranking {min(len(candidates), top_k * 2)} candidates with Cohere")
        reranked = rerank_results(query, candidates[:top_k * 2], top_n=top_k)

        # Reranking replaces scores, so re-apply boosts to keep implementation
        # preference and structural importance.
        if settings.centrality_weight > 0:
            apply_centrality_boost(reranked, repo_name, settings.centrality_weight)
        apply_category_boosting(reranked, sort=True)

        return reranked[:top_k]

    return candidates[:top_k]


def _format_results(results: list[SearchResult]) -> list[dict]:
    """Format search results for diagnostics output."""
    return [{"filename": r.filename, "location": r.location, "score": r.score} for r in results]


def _timed_search(fn, diagnostics: dict, key: str) -> None:
    """Execute a search function with timing and error handling."""
    import time
    start = time.time()
    try:
        diagnostics["results"][key] = _format_results(fn())
    except Exception as e:
        diagnostics["results"][key] = {"error": str(e)}
    diagnostics["timings"][key] = round(time.time() - start, 3)


def search_with_diagnostics(repo_name: str, query: str, top_k: int = 10) -> dict:
    """Perform search with detailed diagnostics for debugging."""
    from .tokenizer import tokenize_for_search
    from .bm25_search import detect_backend, get_corpus_stats

    diagnostics = {
        "query": query,
        "repo_name": repo_name,
        "tokens": tokenize_for_search(query),
        "bm25_backend": detect_backend().value,
        "timings": {},
        "results": {},
        "corpus_stats": {},
    }

    try:
        diagnostics["corpus_stats"] = get_corpus_stats(repo_name)
    except Exception as e:
        diagnostics["corpus_stats"] = {"error": str(e)}

    query_embedding = None
    try:
        query_embedding = get_query_embedding(query)
    except Exception:
        pass

    _timed_search(
        lambda: vector_search(repo_name, query, top_k=top_k, query_embedding=query_embedding),
        diagnostics, "vector"
    )
    _timed_search(lambda: bm25_search(repo_name, query, top_k=top_k), diagnostics, "bm25")
    _timed_search(
        lambda: hybrid_search(repo_name, query, top_k=top_k, use_reranker=False),
        diagnostics, "hybrid_no_rerank"
    )

    if settings.cohere_api_key and settings.enable_reranker:
        _timed_search(
            lambda: hybrid_search(repo_name, query, top_k=top_k, use_reranker=True),
            diagnostics, "hybrid_reranked"
        )

    return diagnostics

"""Hybrid search combining vector and BM25 with Reciprocal Rank Fusion."""

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


def apply_centrality_boost(
    results: list[SearchResult],
    repo_name: str,
    weight: float = 1.0,
) -> None:
    """Apply centrality-based boosting to search results in-place."""
    if not results or weight == 0:
        return

    filenames = list({r.filename for r in results})
    scores = get_centrality_scores(repo_name, filenames)

    for result in results:
        centrality = scores.get(result.filename, 1.0)
        result.score *= 1.0 + (centrality - 1.0) * weight


def get_query_embedding(query: str) -> list[float]:
    """Get embedding for a search query using the configured provider."""
    from src.embeddings.openai import get_embedding as get_embedding_openai

    provider = get_provider()
    if isinstance(provider, OpenAIProvider):
        return get_embedding_openai(query, add_query_context=True)
    return provider.get_embedding(query)


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 40,
    weights: list[float] | None = None,
) -> list[SearchResult]:
    """Combine multiple result lists using Reciprocal Rank Fusion (RRF)."""
    if not result_lists:
        return []

    if weights is None:
        weights = [1.0] * len(result_lists)
    else:
        total = sum(weights)
        weights = [w / total if total > 0 else 1.0 / len(result_lists) for w in weights]

    scores: dict[str, float] = defaultdict(float)
    results_map: dict[str, SearchResult] = {}

    for weight, results in zip(weights, result_lists):
        for rank, result in enumerate(results):
            key = f"{result.filename}:{result.location}:{hash(result.content.strip())}"
            scores[key] += weight / (k + rank + 1)
            if key not in results_map or result.score > results_map[key].score:
                results_map[key] = result

    return [
        SearchResult(
            filename=results_map[key].filename,
            location=results_map[key].location,
            content=results_map[key].content,
            score=scores[key],
        )
        for key in sorted(scores, key=scores.get, reverse=True)
    ]


@dataclass
class SearchOutcome:
    """Result of a search backend execution."""
    results: list[SearchResult] = field(default_factory=list)
    failed: bool = False
    attempted: bool = True


class _EmbeddingCache:
    """Thread-safe cache for query embedding."""

    def __init__(self, query: str):
        self._query = query
        self._embedding: list[float] | None = None
        self._error: Exception | None = None
        self._lock = threading.Lock()

    def get(self) -> list[float]:
        if self._error:
            raise self._error
        if self._embedding:
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
    """Execute a search function with error handling."""
    try:
        return SearchOutcome(results=search_fn())
    except Exception as e:
        logger.warning(f"{name} search failed: {e}")
        return SearchOutcome(failed=True)


def _run_searches_parallel(
    searches: list[tuple[str, Callable[[], list[SearchResult]]]],
) -> dict[str, SearchOutcome]:
    """Run searches in parallel and return outcomes by name."""
    outcomes = {}
    with ThreadPoolExecutor(max_workers=len(searches)) as executor:
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
) -> list[SearchResult]:
    """Perform hybrid search combining vector, BM25, and symbol searches."""
    rerank_count = rerank_candidates or settings.rerank_candidates
    vec_w = vector_weight if vector_weight is not None else settings.vector_weight
    bm25_w = bm25_weight if bm25_weight is not None else settings.bm25_weight
    sym_w = symbol_weight if symbol_weight is not None else settings.symbol_weight

    embedding_cache = _EmbeddingCache(query)
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

    # Check failures
    if all(outcomes[name].failed for name in outcomes):
        raise SearchError("All search backends failed")

    for name, outcome in outcomes.items():
        if outcome.failed:
            logger.warning(f"{name.capitalize()} search failed, using other backends")

    logger.info("Search results: " + ", ".join(f"{k}={len(v.results)}" for k, v in outcomes.items()))

    # Fuse results
    result_lists = [outcomes["vector"].results, outcomes["bm25"].results]
    weights = [vec_w, bm25_w]
    if include_sym and outcomes.get("symbol") and outcomes["symbol"].results:
        result_lists.append(outcomes["symbol"].results)
        weights.append(sym_w)

    candidates = reciprocal_rank_fusion(result_lists, weights=weights)[:rerank_count]

    if settings.centrality_weight > 0:
        apply_centrality_boost(candidates, repo_name, settings.centrality_weight)

    apply_category_boosting(candidates, sort=True)

    if use_reranker and settings.cohere_api_key and candidates:
        logger.debug(f"Reranking {min(len(candidates), top_k * 2)} candidates with Cohere")
        return rerank_results(query, candidates[:top_k * 2], top_n=top_k)

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

    if settings.cohere_api_key:
        _timed_search(
            lambda: hybrid_search(repo_name, query, top_k=top_k, use_reranker=True),
            diagnostics, "hybrid_reranked"
        )

    return diagnostics

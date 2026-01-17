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
    """Apply centrality-based boosting to search results in-place.

    Files imported by many others get boosted, peripheral files get penalized.
    """
    if not results or weight == 0:
        return

    filenames = list({r.filename for r in results})
    scores = get_centrality_scores(repo_name, filenames)

    for result in results:
        centrality = scores.get(result.filename, 1.0)
        adjusted = 1.0 + (centrality - 1.0) * weight
        result.score *= adjusted


def get_query_embedding(query: str) -> list[float]:
    """Get embedding for a search query using the configured provider."""
    from src.embeddings.openai import get_embedding as get_embedding_openai

    provider = get_provider()

    # OpenAI provider: use query context for better retrieval
    if isinstance(provider, OpenAIProvider):
        return get_embedding_openai(query, add_query_context=True)

    # Jina provider: keep consistent with the index embedding space
    return provider.get_embedding(query)


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 40,
    weights: list[float] | None = None,
) -> list[SearchResult]:
    """Combine multiple result lists using Reciprocal Rank Fusion (RRF).

    RRF score = sum(weight * 1 / (k + rank)) for each list
    """
    if not result_lists:
        return []

    # Normalize weights or use uniform distribution
    if weights is None:
        weights = [1.0] * len(result_lists)
    else:
        total = sum(weights)
        uniform = 1.0 / len(result_lists)
        weights = [w / total if total > 0 else uniform for w in weights]

    scores: dict[str, float] = defaultdict(float)
    results_map: dict[str, SearchResult] = {}

    for weight, results in zip(weights, result_lists):
        for rank, result in enumerate(results):
            content_hash = hash(result.content.strip())
            key = f"{result.filename}:{result.location}:{content_hash}"
            scores[key] += weight * (1.0 / (k + rank + 1))
            if key not in results_map or result.score > results_map[key].score:
                results_map[key] = result

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [
        SearchResult(
            filename=results_map[key].filename,
            location=results_map[key].location,
            content=results_map[key].content,
            score=scores[key],
        )
        for key in sorted_keys
    ]


@dataclass
class SearchOutcome:
    """Result of a search backend execution."""
    results: list[SearchResult] = field(default_factory=list)
    failed: bool = False
    attempted: bool = True


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
    vector_fn: Callable[[], list[SearchResult]],
    bm25_fn: Callable[[], list[SearchResult]],
    symbol_fn: Callable[[], list[SearchResult]] | None = None,
) -> tuple[SearchOutcome, SearchOutcome, SearchOutcome]:
    """Run vector, BM25, and symbol searches in parallel."""
    max_workers = 3 if symbol_fn else 2

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_vector = executor.submit(vector_fn)
        future_bm25 = executor.submit(bm25_fn)
        future_symbol = executor.submit(symbol_fn) if symbol_fn else None

        vector = SearchOutcome()
        bm25 = SearchOutcome()
        symbol = SearchOutcome(attempted=symbol_fn is not None)

        try:
            vector.results = future_vector.result(timeout=30)
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            vector.failed = True

        try:
            bm25.results = future_bm25.result(timeout=30)
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            bm25.failed = True

        if future_symbol:
            try:
                symbol.results = future_symbol.result(timeout=30)
            except Exception as e:
                logger.warning(f"Symbol search failed: {e}")
                symbol.failed = True

    return vector, bm25, symbol


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
    """Perform hybrid search combining vector, BM25, and symbol searches with optional reranking.

    Args:
        repo_name: Repository name
        query: Search query
        top_k: Number of results to return
        use_reranker: Whether to use Cohere reranker
        rerank_candidates: Number of candidates to rerank
        vector_weight: Weight for vector search (default from settings)
        bm25_weight: Weight for BM25 search (default from settings)
        symbol_weight: Weight for symbol search (default from settings)
        parallel: Run searches in parallel
        include_symbols: Include symbol-level search

    Returns:
        Ranked search results
    """
    rerank_count = rerank_candidates or settings.rerank_candidates
    vec_weight = vector_weight if vector_weight is not None else settings.vector_weight
    bm_weight = bm25_weight if bm25_weight is not None else settings.bm25_weight
    sym_weight = symbol_weight if symbol_weight is not None else settings.symbol_weight

    query_embedding: list[float] | None = None
    query_embedding_error: Exception | None = None
    query_embedding_lock = threading.Lock()

    def get_query_embedding_cached() -> list[float]:
        nonlocal query_embedding, query_embedding_error
        if query_embedding_error is not None:
            raise query_embedding_error
        if query_embedding is not None:
            return query_embedding
        with query_embedding_lock:
            if query_embedding_error is not None:
                raise query_embedding_error
            if query_embedding is None:
                try:
                    query_embedding = get_query_embedding(query)
                except Exception as e:
                    query_embedding_error = e
                    raise
            return query_embedding
    bm25_config = BM25Config(k1=settings.bm25_k1, b=settings.bm25_b)

    symbols_attempted = include_symbols and settings.enable_symbol_indexing

    def run_vector() -> list[SearchResult]:
        return vector_search(
            repo_name,
            query,
            top_k=rerank_count,
            query_embedding=get_query_embedding_cached(),
        )

    def run_bm25() -> list[SearchResult]:
        return bm25_search(repo_name, query, top_k=rerank_count, config=bm25_config)

    def run_symbol() -> list[SearchResult]:
        if not symbols_attempted:
            return []
        return symbol_hybrid_search(
            repo_name,
            query,
            top_k=rerank_count,
            query_embedding=get_query_embedding_cached(),
        )

    if parallel:
        vector, bm25, symbol = _run_searches_parallel(
            run_vector, run_bm25, run_symbol if symbols_attempted else None
        )
    else:
        logger.debug(f"Running vector search for '{query}' in {repo_name}")
        vector = _execute_search(run_vector, "Vector")
        logger.debug(f"Running BM25 search for '{query}' in {repo_name}")
        bm25 = _execute_search(run_bm25, "BM25")
        if symbols_attempted:
            logger.debug(f"Running symbol search for '{query}' in {repo_name}")
            symbol = _execute_search(run_symbol, "Symbol")
        else:
            symbol = SearchOutcome(attempted=False)

    # Check if all attempted backends failed
    attempted_backends = [vector, bm25]
    if symbol.attempted:
        attempted_backends.append(symbol)

    all_failed = all(b.failed for b in attempted_backends)

    if all_failed:
        raise SearchError("All search backends failed")
    if vector.failed:
        logger.warning("Vector search failed, using BM25/symbol-only results")
    if bm25.failed:
        logger.warning("BM25 search failed, using vector/symbol-only results")
    if symbol.attempted and symbol.failed:
        logger.warning("Symbol search failed, using vector/BM25-only results")

    logger.info(f"Search results: vector={len(vector.results)}, bm25={len(bm25.results)}, symbol={len(symbol.results)}")

    # Prepare result lists and weights for RRF
    result_lists = [vector.results, bm25.results]
    weights = [vec_weight, bm_weight]

    # Add symbol results if available
    if symbol.attempted and symbol.results:
        result_lists.append(symbol.results)
        weights.append(sym_weight)

    fused_results = reciprocal_rank_fusion(result_lists, weights=weights)

    candidates = fused_results[:rerank_count]

    if settings.centrality_weight > 0:
        apply_centrality_boost(candidates, repo_name, settings.centrality_weight)

    apply_category_boosting(candidates, sort=True)

    if use_reranker and settings.cohere_api_key and candidates:
        rerank_input = candidates[:top_k * 2]
        logger.debug(f"Reranking {len(rerank_input)} candidates with Cohere")
        return rerank_results(query, rerank_input, top_n=top_k)

    return candidates[:top_k]


def _format_results(results: list[SearchResult]) -> list[dict]:
    """Format search results for diagnostics output."""
    return [
        {"filename": r.filename, "location": r.location, "score": r.score}
        for r in results
    ]


def search_with_diagnostics(
    repo_name: str,
    query: str,
    top_k: int = 10,
) -> dict:
    """Perform search with detailed diagnostics for debugging."""
    import time
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

    # Run vector search with timing
    start = time.time()
    try:
        query_embedding = get_query_embedding(query)
        vector_results = vector_search(repo_name, query, top_k=top_k, query_embedding=query_embedding)
        diagnostics["results"]["vector"] = _format_results(vector_results)
    except Exception as e:
        diagnostics["results"]["vector"] = {"error": str(e)}
    diagnostics["timings"]["vector"] = round(time.time() - start, 3)

    # Run BM25 search with timing
    start = time.time()
    try:
        bm25_results = bm25_search(repo_name, query, top_k=top_k)
        diagnostics["results"]["bm25"] = _format_results(bm25_results)
    except Exception as e:
        diagnostics["results"]["bm25"] = {"error": str(e)}
    diagnostics["timings"]["bm25"] = round(time.time() - start, 3)

    # Run hybrid search with timing
    start = time.time()
    try:
        hybrid_results = hybrid_search(repo_name, query, top_k=top_k, use_reranker=False)
        diagnostics["results"]["hybrid_no_rerank"] = _format_results(hybrid_results)
    except Exception as e:
        diagnostics["results"]["hybrid_no_rerank"] = {"error": str(e)}
    diagnostics["timings"]["hybrid_no_rerank"] = round(time.time() - start, 3)

    # Run with reranker if available
    if settings.cohere_api_key:
        start = time.time()
        try:
            reranked_results = hybrid_search(repo_name, query, top_k=top_k, use_reranker=True)
            diagnostics["results"]["hybrid_reranked"] = _format_results(reranked_results)
        except Exception as e:
            diagnostics["results"]["hybrid_reranked"] = {"error": str(e)}
        diagnostics["timings"]["hybrid_reranked"] = round(time.time() - start, 3)

    return diagnostics

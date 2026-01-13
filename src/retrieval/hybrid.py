"""Hybrid search combining vector and BM25 with Reciprocal Rank Fusion."""

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from config.settings import settings
from src.embeddings.openai import get_embedding as get_embedding_openai
from src.embeddings import jina as jina_embeddings
from src.exceptions import SearchError

from .bm25_search import BM25Config, bm25_search
from .centrality import get_centrality_scores
from .file_categorizer import apply_category_boosting
from .reranker import rerank_results
from .vector_search import SearchResult, vector_search

logger = logging.getLogger(__name__)


def apply_centrality_boost(
    results: list[SearchResult],
    repo_name: str,
    weight: float = 1.0,
) -> None:
    """Apply centrality-based boosting to search results in-place.

    Files that are imported by many others (central) get boosted,
    while peripheral files (tests, scripts) get penalized.

    Args:
        results: List of search results to modify in-place
        repo_name: Repository name for fetching centrality scores
        weight: How much to apply the boost (0=disabled, 1=full, >1=amplified)
    """
    if not results or weight == 0:
        return

    filenames = list({r.filename for r in results})
    scores = get_centrality_scores(repo_name, filenames)

    for result in results:
        centrality = scores.get(result.filename, 1.0)
        # Interpolate: weight=0 -> no change, weight=1 -> full effect
        adjusted = 1.0 + (centrality - 1.0) * weight
        result.score *= adjusted


def get_query_embedding(query: str) -> list[float]:
    """Get embedding for search query, using Jina if configured and valid."""
    if not (settings.jina_api_key and settings.use_late_chunking):
        return get_embedding_openai(query, add_query_context=True)

    try:
        return jina_embeddings.get_embedding(query, task="retrieval.query")
    except Exception as e:
        logger.warning(f"Jina query embedding failed: {e}. Falling back to OpenAI.")
        return get_embedding_openai(query, add_query_context=True)


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 40,
    weights: list[float] | None = None,
) -> list[SearchResult]:
    """Combine multiple result lists using Reciprocal Rank Fusion (RRF).

    RRF score = sum(weight * 1 / (k + rank)) for each list

    Args:
        result_lists: List of ranked result lists
        k: Constant to prevent high scores for top results (default 40)
        weights: Optional weights for each result list (default equal weights)

    Returns:
        Combined and re-ranked results
    """
    if not result_lists:
        return []

    # Set up weights (equal by default, normalized otherwise)
    if weights is None:
        weights = [1.0] * len(result_lists)
    else:
        total = sum(weights)
        weights = [w / total for w in weights] if total > 0 else [1.0 / len(result_lists)] * len(result_lists)

    # Track scores by unique identifier (filename + location + content hash)
    scores: dict[str, float] = defaultdict(float)
    results_map: dict[str, SearchResult] = {}

    for weight, results in zip(weights, result_lists):
        for rank, result in enumerate(results):
            # Include content hash in key to deduplicate by content
            content_hash = hash(result.content.strip())
            key = f"{result.filename}:{result.location}:{content_hash}"
            scores[key] += weight * (1.0 / (k + rank + 1))  # +1 because rank is 0-indexed
            # Keep result with highest original score
            if key not in results_map or result.score > results_map[key].score:
                results_map[key] = result

    # Sort by RRF score and return results
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


def hybrid_search(
    repo_name: str,
    query: str,
    top_k: int = 10,
    use_reranker: bool = True,
    rerank_candidates: int | None = None,
    vector_weight: float | None = None,
    bm25_weight: float | None = None,
    parallel: bool = True,
) -> list[SearchResult]:
    """Perform hybrid search combining vector and BM25 with optional reranking.

    Pipeline:
    1. Vector search (semantic similarity) - runs in parallel
    2. BM25 search (keyword matching with tf-idf ranking) - runs in parallel
    3. Reciprocal Rank Fusion to combine results
    4. (Optional) Cohere reranking for final ranking

    Args:
        repo_name: Name of the repository to search
        query: Search query text
        top_k: Number of final results to return
        use_reranker: Whether to use Cohere reranker
        rerank_candidates: Number of candidates to consider for reranking
        vector_weight: Weight for vector search results (default from settings)
        bm25_weight: Weight for BM25 search results (default from settings)
        parallel: Whether to run vector and BM25 searches in parallel

    Returns:
        List of search results
    """
    # Set defaults from settings
    rerank_candidates = rerank_candidates or settings.rerank_candidates
    vector_weight = vector_weight if vector_weight is not None else settings.vector_weight
    bm25_weight = bm25_weight if bm25_weight is not None else settings.bm25_weight

    # Get query embedding with context prefix for better matching
    query_embedding = get_query_embedding(query)
    bm25_config = BM25Config(k1=settings.bm25_k1, b=settings.bm25_b)

    # Define search functions
    def run_vector_search():
        return vector_search(repo_name, query, top_k=rerank_candidates, query_embedding=query_embedding)

    def run_bm25_search():
        return bm25_search(repo_name, query, top_k=rerank_candidates, config=bm25_config)

    # Execute searches (parallel or sequential)
    vector_results = []
    bm25_results = []

    if parallel:
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_vector = executor.submit(run_vector_search)
            future_bm25 = executor.submit(run_bm25_search)

            try:
                vector_results = future_vector.result(timeout=30)
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
            try:
                bm25_results = future_bm25.result(timeout=30)
            except Exception as e:
                logger.warning(f"BM25 search failed: {e}")

        # Ensure at least one search succeeded
        if not vector_results and not bm25_results:
            logger.error("Both vector and BM25 searches failed")
            raise SearchError("All search backends failed")

        if not vector_results:
            logger.warning("Vector search failed, using BM25-only results")
        if not bm25_results:
            logger.warning("BM25 search failed, using vector-only results")
    else:
        # Sequential execution
        logger.debug(f"Running vector search for '{query}' in {repo_name}")
        vector_results = run_vector_search()

        logger.debug(f"Running BM25 search for '{query}' in {repo_name}")
        bm25_results = run_bm25_search()

    logger.info(f"Search results: vector={len(vector_results)}, bm25={len(bm25_results)}")

    # Combine with RRF using configured weights
    fused_results = reciprocal_rank_fusion(
        [vector_results, bm25_results],
        weights=[vector_weight, bm25_weight],
    )

    # Limit to rerank candidates
    candidates = fused_results[:rerank_candidates]

    # Apply centrality boosting (structurally important files ranked higher)
    if settings.centrality_weight > 0:
        apply_centrality_boost(candidates, repo_name, settings.centrality_weight)

    # Apply file category boosting BEFORE reranking (CRITICAL FIX)
    # This ensures implementation files are prioritized before semantic reranking
    # which might favor test files with high semantic similarity
    apply_category_boosting(candidates, sort=True)

    if use_reranker and settings.cohere_api_key and candidates:
        # Rerank with Cohere - now working with category-boosted scores
        # Use more candidates to ensure we don't lose good implementation files
        rerank_input = candidates[:top_k * 2]
        logger.debug(f"Reranking {len(rerank_input)} candidates with Cohere")
        final_results = rerank_results(query, rerank_input, top_n=top_k)
    else:
        # Return top-k from fusion
        final_results = candidates[:top_k]

    return final_results


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
    """Perform search with detailed diagnostics for debugging.

    Returns individual results from each search method along with
    timing information and quality metrics.

    Args:
        repo_name: Name of the repository to search
        query: Search query text
        top_k: Number of results from each method

    Returns:
        Dictionary with detailed search diagnostics
    """
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

    # Get corpus stats
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

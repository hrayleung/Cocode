"""Search quality evaluation metrics for information retrieval.

Implements standard IR metrics: MRR, NDCG@K, Recall@K, Precision@K.
"""

import math
from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """Results from evaluating a single query."""
    query: str
    mrr: float
    ndcg_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]


def reciprocal_rank(ranked_results: list[str], relevant: set[str]) -> float:
    """Calculate Reciprocal Rank (RR).
    
    RR = 1/rank of first relevant result, or 0 if none found.
    """
    for i, result in enumerate(ranked_results, 1):
        if result in relevant:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(queries_results: list[tuple[list[str], set[str]]]) -> float:
    """Calculate Mean Reciprocal Rank (MRR) across multiple queries.
    
    Args:
        queries_results: List of (ranked_results, relevant_set) tuples
    """
    if not queries_results:
        return 0.0
    return sum(reciprocal_rank(r, rel) for r, rel in queries_results) / len(queries_results)


def precision_at_k(ranked_results: list[str], relevant: set[str], k: int) -> float:
    """Calculate Precision@K.
    
    Precision@K = (relevant items in top K) / K
    """
    if k <= 0:
        return 0.0
    top_k = ranked_results[:k]
    relevant_in_top_k = sum(1 for r in top_k if r in relevant)
    return relevant_in_top_k / k


def recall_at_k(ranked_results: list[str], relevant: set[str], k: int) -> float:
    """Calculate Recall@K.
    
    Recall@K = (relevant items in top K) / (total relevant items)
    """
    if not relevant or k <= 0:
        return 0.0
    top_k = ranked_results[:k]
    relevant_in_top_k = sum(1 for r in top_k if r in relevant)
    return relevant_in_top_k / len(relevant)


def dcg_at_k(ranked_results: list[str], relevance_scores: dict[str, float], k: int) -> float:
    """Calculate Discounted Cumulative Gain at K.
    
    DCG@K = sum(rel_i / log2(i+1)) for i in 1..K
    """
    dcg = 0.0
    for i, result in enumerate(ranked_results[:k], 1):
        rel = relevance_scores.get(result, 0.0)
        dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_results: list[str], relevance_scores: dict[str, float], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K.
    
    NDCG@K = DCG@K / IDCG@K (ideal DCG)
    """
    dcg = dcg_at_k(ranked_results, relevance_scores, k)
    
    # Ideal ranking: sort by relevance descending
    ideal_ranking = sorted(relevance_scores.keys(), key=lambda x: relevance_scores[x], reverse=True)
    idcg = dcg_at_k(ideal_ranking, relevance_scores, k)
    
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_query(
    query: str,
    ranked_results: list[str],
    relevant: set[str],
    relevance_scores: dict[str, float] | None = None,
    k_values: list[int] | None = None,
) -> EvaluationResult:
    """Evaluate search results for a single query.
    
    Args:
        query: The search query
        ranked_results: List of result identifiers in ranked order
        relevant: Set of relevant result identifiers
        relevance_scores: Optional graded relevance (default: binary 1.0 for relevant)
        k_values: K values for @K metrics (default: [1, 3, 5, 10])
    """
    k_values = k_values or [1, 3, 5, 10]
    
    # Default to binary relevance if no scores provided
    if relevance_scores is None:
        relevance_scores = {r: 1.0 for r in relevant}
    
    return EvaluationResult(
        query=query,
        mrr=reciprocal_rank(ranked_results, relevant),
        ndcg_at_k={k: ndcg_at_k(ranked_results, relevance_scores, k) for k in k_values},
        recall_at_k={k: recall_at_k(ranked_results, relevant, k) for k in k_values},
        precision_at_k={k: precision_at_k(ranked_results, relevant, k) for k in k_values},
    )


def aggregate_results(results: list[EvaluationResult]) -> dict[str, float]:
    """Aggregate evaluation results across multiple queries.
    
    Returns mean values for all metrics.
    """
    if not results:
        return {}
    
    n = len(results)
    
    # Find common k values across all results
    common_k = set(results[0].ndcg_at_k.keys())
    for r in results[1:]:
        common_k &= set(r.ndcg_at_k.keys())
        common_k &= set(r.recall_at_k.keys())
        common_k &= set(r.precision_at_k.keys())
    
    aggregated = {
        "mrr": sum(r.mrr for r in results) / n,
    }
    
    for k in sorted(common_k):
        aggregated[f"ndcg@{k}"] = sum(r.ndcg_at_k[k] for r in results) / n
        aggregated[f"recall@{k}"] = sum(r.recall_at_k[k] for r in results) / n
        aggregated[f"precision@{k}"] = sum(r.precision_at_k[k] for r in results) / n
    
    return aggregated

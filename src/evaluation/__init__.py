"""Search quality evaluation module."""

from .metrics import (
    EvaluationResult,
    reciprocal_rank,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
    dcg_at_k,
    ndcg_at_k,
    evaluate_query,
    aggregate_results,
)

__all__ = [
    "EvaluationResult",
    "reciprocal_rank",
    "mean_reciprocal_rank",
    "precision_at_k",
    "recall_at_k",
    "dcg_at_k",
    "ndcg_at_k",
    "evaluate_query",
    "aggregate_results",
]

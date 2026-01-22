"""Python wrapper for Rust-accelerated vector operations."""

import numpy as np

from cocode_rust import (
    cosine_similarity as _rust_cosine_similarity,
    cosine_similarity_batch as _rust_cosine_similarity_batch,
    reciprocal_rank_fusion as _rust_rrf,
    reciprocal_rank_fusion_weighted as _rust_rrf_weighted,
)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    v1 = vec1.astype(np.float32) if vec1.dtype != np.float32 else vec1
    v2 = vec2.astype(np.float32) if vec2.dtype != np.float32 else vec2
    return _rust_cosine_similarity(v1, v2)


def cosine_similarity_batch(query: np.ndarray, documents: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query and multiple document vectors."""
    q = query.astype(np.float32) if query.dtype != np.float32 else query
    docs = documents.astype(np.float32) if documents.dtype != np.float32 else documents
    return _rust_cosine_similarity_batch(q, docs)


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: float = 60.0,
) -> list[tuple[str, float]]:
    """Combine multiple ranked lists using RRF. Returns (doc_id, fused_score) sorted by score."""
    return _rust_rrf(ranked_lists, k)


def reciprocal_rank_fusion_weighted(
    ranked_lists: list[list[tuple[str, float]]],
    weights: list[float],
    k: float = 60.0,
) -> list[tuple[str, float]]:
    """Weighted RRF with custom weights per list. Returns (doc_id, fused_score) sorted by score."""
    return _rust_rrf_weighted(ranked_lists, weights, k)

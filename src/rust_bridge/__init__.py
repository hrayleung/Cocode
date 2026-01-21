"""
Rust bridge module for accelerated operations.

Provides Python wrappers for Rust functions.
All modules gracefully fall back to Python implementations if Rust extensions are unavailable.
"""

from .vector_ops import (
    cosine_similarity,
    cosine_similarity_batch,
    reciprocal_rank_fusion,
    reciprocal_rank_fusion_weighted,
)
from .graph_algos import (
    pagerank,
    bfs_expansion,
    strongly_connected_components,
)
from .bm25_engine import BM25Engine
from .tokenizer import (
    extract_code_tokens,
    tokenize_for_search,
    batch_extract_tokens,
    batch_tokenize_queries,
)

__all__ = [
    "cosine_similarity",
    "cosine_similarity_batch",
    "reciprocal_rank_fusion",
    "reciprocal_rank_fusion_weighted",
    "pagerank",
    "bfs_expansion",
    "strongly_connected_components",
    "BM25Engine",
    "extract_code_tokens",
    "tokenize_for_search",
    "batch_extract_tokens",
    "batch_tokenize_queries",
]

"""
Rust bridge module for accelerated operations.

Provides Python wrappers for Rust functions from the cocode_rust extension.
The cocode_rust extension must be installed; ImportError is raised if missing.
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
    bfs_traversal_edges,
    strongly_connected_components,
)
from .bm25_engine import BM25Engine
from .tokenizer import (
    extract_code_tokens,
    tokenize_for_search,
    batch_extract_tokens,
    batch_tokenize_queries,
)
from .utils import (
    compute_file_hash,
    jaccard_similarity,
    jaccard_similarity_batch,
    mmr_select_indices,
    extract_code_by_line_range,
)
from .parser import (
    is_language_supported,
    extract_imports_ast,
    extract_symbols,
    extract_relationships,
    extract_calls,
)

__all__ = [
    "cosine_similarity",
    "cosine_similarity_batch",
    "reciprocal_rank_fusion",
    "reciprocal_rank_fusion_weighted",
    "pagerank",
    "bfs_expansion",
    "bfs_traversal_edges",
    "strongly_connected_components",
    "BM25Engine",
    "extract_code_tokens",
    "tokenize_for_search",
    "batch_extract_tokens",
    "batch_tokenize_queries",
    "compute_file_hash",
    "jaccard_similarity",
    "jaccard_similarity_batch",
    "mmr_select_indices",
    "extract_code_by_line_range",
    "is_language_supported",
    "extract_imports_ast",
    "extract_symbols",
    "extract_relationships",
    "extract_calls",
]

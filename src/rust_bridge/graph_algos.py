"""Python wrapper for Rust-accelerated graph algorithms."""

from cocode_rust import (
    pagerank as _rust_pagerank,
    bfs_expansion as _rust_bfs_expansion,
    bfs_traversal_edges as _rust_bfs_traversal_edges,
    strongly_connected_components as _rust_scc,
)


def pagerank(
    edges: list[tuple[str, str]],
    damping: float = 0.85,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> dict[str, float]:
    """Compute PageRank scores for a directed graph. Returns node -> score mapping."""
    return _rust_pagerank(edges, damping, max_iterations, tolerance)


def bfs_expansion(
    edges: list[tuple[str, str]],
    start_nodes: list[str],
    max_hops: int = 3,
    max_results: int = 30,
    bidirectional: bool = True,
) -> dict[str, int]:
    """Multi-hop BFS graph expansion. Returns node -> hop distance mapping."""
    return _rust_bfs_expansion(edges, start_nodes, max_hops, max_results, bidirectional)


def bfs_traversal_edges(
    edges: list[tuple[str, str]],
    start_nodes: list[str],
    max_hops: int = 3,
    max_results: int = 30,
    bidirectional: bool = True,
) -> list[tuple[str, str, str, int]]:
    """BFS traversal returning predecessor edges in BFS order.

    Returns tuples of (source_file, target_file, relation_type, hop_distance).
    """
    return _rust_bfs_traversal_edges(edges, start_nodes, max_hops, max_results, bidirectional)


def strongly_connected_components(edges: list[tuple[str, str]]) -> list[list[str]]:
    """Detect SCCs using Kosaraju's algorithm. Returns list of component node lists."""
    return _rust_scc(edges)

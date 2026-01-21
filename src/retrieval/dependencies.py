"""Dependency graph helpers.

Provides a lightweight API to compute import edges between a given set of files,
using the existing import-graph cache when available.
"""

from __future__ import annotations

from collections import defaultdict


def get_import_edges(repo_name: str, filenames: list[str]) -> list[dict]:
    """Return direct import edges between the given files.

    Output edges are directed and use relation_type="imports".
    """

    file_set = set(filenames)
    if not file_set:
        return []

    from src.retrieval.graph_cache import get_cached_import_graph
    from src.retrieval.graph_expansion import build_import_graph

    cached = get_cached_import_graph(repo_name)
    if cached:
        import_graph, _ = cached
    else:
        import_graph = build_import_graph(repo_name)

    edges: list[dict] = []

    for src in filenames:
        for tgt in import_graph.get(src, []) or []:
            if tgt in file_set and tgt != src:
                edges.append(
                    {
                        "source_file": src,
                        "target_file": tgt,
                        "relation_type": "imports",
                        "hop_distance": 1,
                    }
                )

    return edges


def get_import_adjacency(repo_name: str, filenames: list[str]) -> dict[str, dict[str, list[str]]]:
    """Return adjacency lists (imports + imported_by) restricted to the given files."""

    file_set = set(filenames)
    if not file_set:
        return {}

    from src.retrieval.graph_cache import get_cached_import_graph
    from src.retrieval.graph_expansion import build_import_graph

    cached = get_cached_import_graph(repo_name)
    if cached:
        import_graph, reverse_graph = cached
    else:
        import_graph = build_import_graph(repo_name)
        reverse_graph: dict[str, list[str]] = defaultdict(list)
        for src, targets in import_graph.items():
            for tgt in targets or []:
                reverse_graph[tgt].append(src)

    out: dict[str, dict[str, list[str]]] = {}

    for f in filenames:
        out[f] = {
            "imports": [t for t in (import_graph.get(f, []) or []) if t in file_set and t != f],
            "imported_by": [t for t in (reverse_graph.get(f, []) or []) if t in file_set and t != f],
        }

    return out

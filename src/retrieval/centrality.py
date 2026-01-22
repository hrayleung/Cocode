"""Graph centrality scoring for code retrieval.

Computes PageRank-style centrality scores based on import relationships.
Files imported by many others are structurally central; files that only
import (tests, scripts) are peripheral.
"""

import logging
from collections import defaultdict

from psycopg import sql

from src.storage.postgres import get_connection
from src.retrieval.vector_search import sanitize_identifier
from src.retrieval.ttl_cache import TTLCache

logger = logging.getLogger(__name__)

DEFAULT_CENTRALITY = 1.0
MIN_CENTRALITY = 0.5
MAX_CENTRALITY = 2.0

_table_exists_cache = TTLCache(ttl_hit=300.0, ttl_miss=30.0)


def compute_pagerank(
    import_graph: dict[str, list[str]],
    damping: float = 0.85,
    iterations: int = 20,
) -> dict[str, float]:
    """Compute PageRank-style centrality scores."""
    if not import_graph:
        return {}

    from src.rust_bridge import pagerank as rust_pagerank

    edges: list[tuple[str, str]] = []
    for source, targets in import_graph.items():
        for target in targets:
            edges.append((source, target))

    return rust_pagerank(edges, damping=damping, max_iterations=iterations)


def normalize_scores(
    scores: dict[str, float],
    min_score: float = MIN_CENTRALITY,
    max_score: float = MAX_CENTRALITY,
) -> dict[str, float]:
    """Normalize scores to a target range."""
    if not scores:
        return {}

    raw_min, raw_max = min(scores.values()), max(scores.values())
    raw_range = raw_max - raw_min

    if raw_range == 0:
        mid = (min_score + max_score) / 2
        return {f: mid for f in scores}

    target_range = max_score - min_score
    return {
        file: min_score + ((score - raw_min) / raw_range) * target_range
        for file, score in scores.items()
    }


def compute_centrality_scores(
    import_graph: dict[str, list[str]],
    damping: float = 0.85,
    iterations: int = 20,
) -> dict[str, float]:
    """Compute normalized centrality scores from import graph."""
    return normalize_scores(compute_pagerank(import_graph, damping, iterations))


def _get_centrality_table_name(repo_name: str) -> str:
    return f"{sanitize_identifier(repo_name)}_centrality"


def store_centrality_scores(repo_name: str, scores: dict[str, float]) -> None:
    """Store centrality scores in the database."""
    table_name = _get_centrality_table_name(repo_name)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {} (
                    filename TEXT PRIMARY KEY,
                    centrality_score FLOAT NOT NULL
                )
            """).format(sql.Identifier(table_name)))

            cur.execute(sql.SQL("DELETE FROM {}").format(sql.Identifier(table_name)))

            if scores:
                batch_size = 500
                items = list(scores.items())
                insert_sql = sql.SQL("INSERT INTO {} (filename, centrality_score) VALUES (%s, %s)").format(
                    sql.Identifier(table_name)
                )
                for i in range(0, len(items), batch_size):
                    cur.executemany(insert_sql, items[i:i + batch_size])

        conn.commit()

    _table_exists_cache.set(table_name, True)
    logger.info(f"Stored {len(scores)} centrality scores for {repo_name}")


def get_centrality_scores(repo_name: str, filenames: list[str]) -> dict[str, float]:
    """Fetch centrality scores for given files."""
    if not filenames:
        return {}

    table_name = _get_centrality_table_name(repo_name)

    try:
        cached_exists = _table_exists_cache.get(table_name)
        if cached_exists is False:
            return {f: DEFAULT_CENTRALITY for f in filenames}

        with get_connection() as conn:
            with conn.cursor() as cur:
                if cached_exists is None:
                    cur.execute(
                        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
                        (table_name,),
                    )
                    exists = bool(cur.fetchone()[0])
                    _table_exists_cache.set(table_name, exists)
                    if not exists:
                        logger.debug(f"Centrality table {table_name} does not exist")
                        return {f: DEFAULT_CENTRALITY for f in filenames}

                placeholders = ",".join(["%s"] * len(filenames))
                cur.execute(sql.SQL("""
                    SELECT filename, centrality_score FROM {} WHERE filename IN ({})
                """).format(sql.Identifier(table_name), sql.SQL(placeholders)), filenames)

                scores = {row[0]: row[1] for row in cur.fetchall()}

        return {f: scores.get(f, DEFAULT_CENTRALITY) for f in filenames}

    except Exception as e:
        logger.warning(f"Failed to fetch centrality scores: {e}")
        return {f: DEFAULT_CENTRALITY for f in filenames}


def compute_and_store_centrality(repo_name: str) -> dict[str, float]:
    """Build import graph and compute/store centrality scores."""
    from src.retrieval.graph_expansion import build_import_graph

    logger.info(f"Computing centrality scores for {repo_name}")

    try:
        import_graph = build_import_graph(repo_name)
        if not import_graph:
            logger.warning(f"Empty import graph for {repo_name}")
            return {}

        scores = compute_centrality_scores(import_graph)
        store_centrality_scores(repo_name, scores)
        logger.debug(f"Computed centrality for {len(scores)} files")
        return scores

    except Exception as e:
        logger.error(f"Failed to compute centrality for {repo_name}: {e}")
        return {}


def delete_centrality_table(repo_name: str) -> None:
    """Delete centrality table for a repo."""
    table_name = _get_centrality_table_name(repo_name)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name)))
            conn.commit()
        _table_exists_cache.set(table_name, False)
        logger.debug(f"Deleted centrality table {table_name}")
    except Exception as e:
        logger.warning(f"Failed to delete centrality table {table_name}: {e}")

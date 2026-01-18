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

logger = logging.getLogger(__name__)

# Default centrality for files not in the graph
DEFAULT_CENTRALITY = 1.0

# Score range for normalization
MIN_CENTRALITY = 0.5
MAX_CENTRALITY = 2.0


def compute_pagerank(
    import_graph: dict[str, list[str]],
    damping: float = 0.85,
    iterations: int = 20,
) -> dict[str, float]:
    """Compute PageRank-style centrality scores.

    Files imported by many others get higher scores. The algorithm
    propagates importance through the graph iteratively.

    Args:
        import_graph: Dict mapping file -> list of files it imports
        damping: Damping factor (probability of following a link)
        iterations: Number of iterations to converge

    Returns:
        Dict mapping file -> raw centrality score
    """
    if not import_graph:
        return {}

    # Collect all files (both importers and imported)
    all_files = set(import_graph.keys())
    for imports in import_graph.values():
        all_files.update(imports)

    n = len(all_files)

    # Initialize uniform scores
    scores = {f: 1.0 / n for f in all_files}

    # Build reverse lookup: file -> files that import it
    imported_by: dict[str, list[str]] = defaultdict(list)
    for source, targets in import_graph.items():
        for target in targets:
            imported_by[target].append(source)

    # Iterative PageRank
    for _ in range(iterations):
        new_scores = {}
        for file in all_files:
            # Base score (random jump)
            score = (1 - damping) / n

            # Score from incoming links
            for importer in imported_by[file]:
                out_degree = len(import_graph.get(importer, []))
                if out_degree > 0:
                    score += damping * scores[importer] / out_degree

            new_scores[file] = score

        scores = new_scores

    return scores


def normalize_scores(
    scores: dict[str, float],
    min_score: float = MIN_CENTRALITY,
    max_score: float = MAX_CENTRALITY,
) -> dict[str, float]:
    """Normalize scores to a target range.

    Args:
        scores: Raw centrality scores
        min_score: Minimum output score
        max_score: Maximum output score

    Returns:
        Dict mapping file -> normalized score in [min_score, max_score]
    """
    if not scores:
        return {}

    raw_min, raw_max = min(scores.values()), max(scores.values())
    raw_range = raw_max - raw_min

    # All scores equal - return midpoint
    if raw_range == 0:
        mid = (min_score + max_score) / 2
        return {f: mid for f in scores}

    # Linear interpolation to target range
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
    """Compute normalized centrality scores from import graph.

    Args:
        import_graph: Dict mapping file -> list of files it imports
        damping: PageRank damping factor
        iterations: PageRank iterations

    Returns:
        Dict mapping file -> centrality score in [0.5, 2.0]
    """
    raw_scores = compute_pagerank(import_graph, damping, iterations)
    return normalize_scores(raw_scores)


def _get_centrality_table_name(repo_name: str) -> str:
    """Get sanitized centrality table name for a repository."""
    return f"{sanitize_identifier(repo_name)}_centrality"


def store_centrality_scores(repo_name: str, scores: dict[str, float]) -> None:
    """Store centrality scores in the database.

    Creates or replaces the centrality table for the repo.

    Args:
        repo_name: Repository name
        scores: Dict mapping filename -> centrality score
    """
    table_name = _get_centrality_table_name(repo_name)

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Create table if not exists
            cur.execute(sql.SQL("""
                CREATE TABLE IF NOT EXISTS {} (
                    filename TEXT PRIMARY KEY,
                    centrality_score FLOAT NOT NULL
                )
            """).format(sql.Identifier(table_name)))

            # Clear existing scores
            cur.execute(sql.SQL("DELETE FROM {}").format(sql.Identifier(table_name)))

            # Insert new scores in batches
            if scores:
                batch_size = 500
                items = list(scores.items())
                insert_sql = sql.SQL("INSERT INTO {} (filename, centrality_score) VALUES (%s, %s)").format(
                    sql.Identifier(table_name)
                )
                for i in range(0, len(items), batch_size):
                    batch = items[i:i + batch_size]
                    cur.executemany(insert_sql, batch)

        conn.commit()

    logger.info(f"Stored {len(scores)} centrality scores for {repo_name}")


def get_centrality_scores(
    repo_name: str,
    filenames: list[str],
) -> dict[str, float]:
    """Fetch centrality scores for given files.

    Args:
        repo_name: Repository name
        filenames: List of filenames to fetch scores for

    Returns:
        Dict mapping filename -> centrality score (default 1.0 if not found)
    """
    if not filenames:
        return {}

    table_name = _get_centrality_table_name(repo_name)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = %s
                    )
                """, (table_name,))

                if not cur.fetchone()[0]:
                    logger.debug(f"Centrality table {table_name} does not exist")
                    return {f: DEFAULT_CENTRALITY for f in filenames}

                # Fetch scores for requested files
                placeholders = ",".join(["%s"] * len(filenames))
                cur.execute(sql.SQL("""
                    SELECT filename, centrality_score
                    FROM {}
                    WHERE filename IN ({})
                """).format(
                    sql.Identifier(table_name),
                    sql.SQL(placeholders)
                ), filenames)

                scores = {row[0]: row[1] for row in cur.fetchall()}

        # Fill in defaults for missing files
        return {f: scores.get(f, DEFAULT_CENTRALITY) for f in filenames}

    except Exception as e:
        logger.warning(f"Failed to fetch centrality scores: {e}")
        return {f: DEFAULT_CENTRALITY for f in filenames}


def compute_and_store_centrality(repo_name: str) -> dict[str, float]:
    """Build import graph and compute/store centrality scores.

    Main entry point for centrality computation during indexing.

    Args:
        repo_name: Repository name

    Returns:
        Dict mapping filename -> centrality score
    """
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
    """Delete centrality table for a repo.

    Args:
        repo_name: Repository name
    """
    table_name = _get_centrality_table_name(repo_name)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name)))
            conn.commit()
        logger.debug(f"Deleted centrality table {table_name}")
    except Exception as e:
        logger.warning(f"Failed to delete centrality table {table_name}: {e}")

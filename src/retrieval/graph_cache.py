"""Graph cache management - store and retrieve pre-computed import graphs.

This module provides caching for import graphs to avoid rebuilding them
on every search operation.
"""

import json
import logging

from psycopg import sql

from src.storage.postgres import get_connection
from src.storage.schema import sanitize_repo_name

logger = logging.getLogger(__name__)


def create_graph_cache_table(repo_name: str) -> None:
    """Create the graph_cache table for a repository if it doesn't exist.

    Args:
        repo_name: Repository name
    """
    from src.storage.schema import get_create_graph_cache_table_sql

    cache_sql = get_create_graph_cache_table_sql(repo_name)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(cache_sql)
            conn.commit()

    logger.debug(f"Created graph_cache table for {repo_name}")


def populate_graph_cache(repo_name: str, import_graph: dict[str, list[str]]) -> int:
    """
    Populate the repository's graph_cache table from a complete forward import graph.
    
    Parameters:
        repo_name (str): Repository name used to scope the cache table.
        import_graph (dict[str, list[str]]): Mapping from filename to list of filenames it imports.
    
    Returns:
        int: Number of cache entries created or updated.
    """
    schema_name = sanitize_repo_name(repo_name)
    cache_table = sql.Identifier(schema_name, "graph_cache")

    # Build reverse graph (imported_by)
    reverse_graph: dict[str, list[str]] = {}
    for source, targets in import_graph.items():
        for target in targets:
            if target not in reverse_graph:
                reverse_graph[target] = []
            reverse_graph[target].append(source)

    # Get all unique files
    all_files = set(import_graph.keys()) | set(reverse_graph.keys())

    entries_created = 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            for filename in all_files:
                imports = import_graph.get(filename, [])
                imported_by = reverse_graph.get(filename, [])

                cur.execute(
                    sql.SQL("""
                        INSERT INTO {table} (filename, imports, imported_by, last_updated)
                        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                        ON CONFLICT (filename)
                        DO UPDATE SET
                            imports = EXCLUDED.imports,
                            imported_by = EXCLUDED.imported_by,
                            last_updated = CURRENT_TIMESTAMP
                    """).format(table=cache_table),
                    (filename, json.dumps(imports), json.dumps(imported_by))
                )
                entries_created += 1

            conn.commit()

    logger.info(f"Populated graph cache with {entries_created} entries for {repo_name}")
    return entries_created


def get_cached_import_graph(repo_name: str) -> tuple[dict[str, list[str]], dict[str, list[str]]] | None:
    """
    Load cached import and reverse-import graphs for a repository.
    
    Parameters:
        repo_name (str): Repository name to locate the graph cache.
    
    Returns:
        tuple[dict[str, list[str]], dict[str, list[str]]] | None: A pair `(import_graph, reverse_graph)` where `import_graph` maps a filename to the list of filenames it imports and `reverse_graph` maps a filename to the list of filenames that import it; `None` if the cache table is missing, the cache is empty, or loading fails.
    """
    schema_name = sanitize_repo_name(repo_name)
    cache_table = sql.Identifier(schema_name, "graph_cache")

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check if cache table exists
                cur.execute(
                    sql.SQL("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_schema = %s AND table_name = 'graph_cache'
                        )
                    """),
                    (schema_name,)
                )

                if not cur.fetchone()[0]:
                    logger.debug(f"Graph cache table doesn't exist for {repo_name}")
                    return None

                # Fetch all cache entries
                cur.execute(
                    sql.SQL("""
                        SELECT filename, imports, imported_by
                        FROM {table}
                    """).format(table=cache_table)
                )

                rows = cur.fetchall()

                if not rows:
                    logger.debug(f"Graph cache is empty for {repo_name}")
                    return None

                # Build graphs from cache
                import_graph = {}
                reverse_graph = {}

                for filename, imports_json, imported_by_json in rows:
                    # psycopg already decodes JSONB to Python objects
                    imports = imports_json if imports_json else []
                    imported_by = imported_by_json if imported_by_json else []

                    if imports:
                        import_graph[filename] = imports
                    if imported_by:
                        reverse_graph[filename] = imported_by

                logger.debug(f"Loaded graph cache with {len(import_graph)} entries for {repo_name}")
                return import_graph, reverse_graph

    except Exception as e:
        logger.warning(f"Failed to load graph cache for {repo_name}: {e}")
        return None


def invalidate_file_cache(repo_name: str, filename: str) -> int:
    """Invalidate cache entry for a file that changed.

    Also invalidates files that import this file (need to rebuild their imported_by).

    Args:
        repo_name: Repository name
        filename: Filename that changed

    Returns:
        Number of cache entries invalidated
    """
    schema_name = sanitize_repo_name(repo_name)
    cache_table = sql.Identifier(schema_name, "graph_cache")

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # First, get files that import this file (need to update their imported_by)
                cur.execute(
                    sql.SQL("""
                        SELECT filename FROM {table}
                        WHERE imports @> %s::jsonb
                    """).format(table=cache_table),
                    (json.dumps([filename]),)
                )

                dependent_files = [row[0] for row in cur.fetchall()]

                # Delete cache entry for the changed file
                cur.execute(
                    sql.SQL("""
                        DELETE FROM {table}
                        WHERE filename = %s
                    """).format(table=cache_table),
                    (filename,)
                )

                deleted = cur.rowcount

                # Delete cache entries for dependent files
                if dependent_files:
                    cur.execute(
                        sql.SQL("""
                            DELETE FROM {table}
                            WHERE filename = ANY(%s)
                        """).format(table=cache_table),
                        (dependent_files,)
                    )
                    deleted += cur.rowcount

                conn.commit()

                logger.debug(f"Invalidated {deleted} cache entries for {filename}")
                return deleted

    except Exception as e:
        logger.warning(f"Failed to invalidate cache for {filename}: {e}")
        return 0


def clear_graph_cache(repo_name: str) -> int:
    """Clear entire graph cache for a repository.

    Args:
        repo_name: Repository name

    Returns:
        Number of cache entries deleted
    """
    schema_name = sanitize_repo_name(repo_name)
    cache_table = sql.Identifier(schema_name, "graph_cache")

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("""
                        DELETE FROM {table}
                    """).format(table=cache_table)
                )

                deleted = cur.rowcount
                conn.commit()

                logger.info(f"Cleared graph cache for {repo_name}: {deleted} entries")
                return deleted

    except Exception as e:
        logger.warning(f"Failed to clear graph cache for {repo_name}: {e}")
        return 0


def update_file_cache_stats(repo_name: str, filename: str, symbol_count: int = 0, edge_count: int = 0) -> None:
    """Update statistics for a file in the cache.

    Args:
        repo_name: Repository name
        filename: Filename
        symbol_count: Number of symbols in the file
        edge_count: Number of call edges from the file
    """
    schema_name = sanitize_repo_name(repo_name)
    cache_table = sql.Identifier(schema_name, "graph_cache")

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("""
                        UPDATE {table}
                        SET symbol_count = %s, edge_count = %s, last_updated = CURRENT_TIMESTAMP
                        WHERE filename = %s
                    """).format(table=cache_table),
                    (symbol_count, edge_count, filename)
                )
                conn.commit()

    except Exception as e:
        logger.debug(f"Failed to update cache stats for {filename}: {e}")
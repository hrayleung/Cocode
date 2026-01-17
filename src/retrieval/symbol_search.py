"""Symbol-level search for functions, classes, and methods.

Searches symbols table using vector similarity and BM25 full-text search,
returning results with precise line number references.
"""

import logging
from psycopg import sql

from config.settings import settings
from src.models import SearchResult
from src.storage.postgres import get_connection
from src.storage.schema import sanitize_repo_name

logger = logging.getLogger(__name__)


def _format_symbol_content(
    signature: str,
    docstring: str | None,
    parent_symbol: str | None,
) -> str:
    """Format symbol signature and docstring into content for search results."""
    parts = []
    if parent_symbol:
        parts.append(f"class {parent_symbol}:")

    parts.append(f"    {signature}" if parent_symbol else signature)

    if docstring:
        doc_preview = docstring[:200] + "..." if len(docstring) > 200 else docstring
        parts.append(f'    """{doc_preview}"""')

    return "\n".join(parts)


def symbol_vector_search(
    repo_name: str,
    query: str,
    top_k: int = 10,
    query_embedding: list[float] | None = None,
) -> list[SearchResult]:
    """Search symbols using vector similarity.

    Args:
        repo_name: Repository name
        query: Search query
        top_k: Number of results to return
        query_embedding: Pre-computed query embedding (optional)

    Returns:
        List of SearchResult objects with symbol matches
    """
    if not settings.enable_symbol_indexing:
        logger.debug("Symbol indexing disabled, skipping symbol search")
        return []

    schema_name = sanitize_repo_name(repo_name)
    symbols_table = sql.Identifier(schema_name, "symbols")

    # If embedding not provided, generate it
    if query_embedding is None:
        from src.retrieval.hybrid import get_query_embedding
        query_embedding = get_query_embedding(query)

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Vector similarity search on symbols
            search_sql = sql.SQL("""
                SELECT
                    filename,
                    symbol_name,
                    symbol_type,
                    line_start,
                    line_end,
                    signature,
                    docstring,
                    parent_symbol,
                    visibility,
                    category,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM {table}
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """).format(table=symbols_table)

            cur.execute(search_sql, (query_embedding, query_embedding, top_k))
            rows = cur.fetchall()

    results = []
    for row in rows:
        filename, symbol_name, symbol_type, line_start, line_end, signature, docstring, parent_symbol, visibility, category, similarity = row

        results.append(
            SearchResult(
                filename=filename,
                location=f"{line_start}:{line_end}",
                content=_format_symbol_content(signature, docstring, parent_symbol),
                score=float(similarity),
            )
        )

    logger.debug(f"Symbol vector search returned {len(results)} results")
    return results


def symbol_bm25_search(
    repo_name: str,
    query: str,
    top_k: int = 10,
) -> list[SearchResult]:
    """Search symbols using BM25 full-text search.

    Args:
        repo_name: Repository name
        query: Search query
        top_k: Number of results to return

    Returns:
        List of SearchResult objects with symbol matches
    """
    if not settings.enable_symbol_indexing:
        logger.debug("Symbol indexing disabled, skipping symbol search")
        return []

    schema_name = sanitize_repo_name(repo_name)
    symbols_table = sql.Identifier(schema_name, "symbols")

    with get_connection() as conn:
        with conn.cursor() as cur:
            # BM25 search using PostgreSQL full-text search
            search_sql = sql.SQL("""
                SELECT
                    filename,
                    symbol_name,
                    symbol_type,
                    line_start,
                    line_end,
                    signature,
                    docstring,
                    parent_symbol,
                    visibility,
                    category,
                    ts_rank_cd(content_tsv, query, 1) AS rank
                FROM {table}, plainto_tsquery('english', %s) query
                WHERE content_tsv @@ query
                ORDER BY rank DESC
                LIMIT %s
            """).format(table=symbols_table)

            cur.execute(search_sql, (query, top_k))
            rows = cur.fetchall()

    results = []
    for row in rows:
        filename, symbol_name, symbol_type, line_start, line_end, signature, docstring, parent_symbol, visibility, category, rank = row

        results.append(
            SearchResult(
                filename=filename,
                location=f"{line_start}:{line_end}",
                content=_format_symbol_content(signature, docstring, parent_symbol),
                score=float(rank),
            )
        )

    logger.debug(f"Symbol BM25 search returned {len(results)} results")
    return results


def symbol_hybrid_search(
    repo_name: str,
    query: str,
    top_k: int = 10,
    query_embedding: list[float] | None = None,
) -> list[SearchResult]:
    """Hybrid search combining symbol vector and BM25 searches.

    Args:
        repo_name: Repository name
        query: Search query
        top_k: Number of results to return
        query_embedding: Pre-computed query embedding (optional)

    Returns:
        Combined and ranked symbol search results
    """
    if not settings.enable_symbol_indexing:
        return []

    from .hybrid import reciprocal_rank_fusion

    # Run both searches
    vector_results = symbol_vector_search(repo_name, query, top_k=top_k * 2, query_embedding=query_embedding)
    bm25_results = symbol_bm25_search(repo_name, query, top_k=top_k * 2)

    # Combine with RRF
    combined = reciprocal_rank_fusion(
        [vector_results, bm25_results],
        weights=[settings.vector_weight, settings.bm25_weight],
    )

    return combined[:top_k]

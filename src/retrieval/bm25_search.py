"""BM25 search using PostgreSQL full-text search with multiple backend support."""

import logging
from dataclasses import dataclass
from enum import Enum

from psycopg import sql as psycopg_sql

from src.exceptions import SearchError
from src.storage.postgres import get_connection

from .tokenizer import tokenize_for_search
from .ttl_cache import TTLCache
from .vector_search import SearchResult, get_chunks_table_name

logger = logging.getLogger(__name__)


class BM25Backend(Enum):
    PG_SEARCH = "pg_search"
    PG_TEXTSEARCH = "pg_textsearch"
    NATIVE_FTS = "native_fts"


@dataclass
class BM25Config:
    k1: float = 1.2
    b: float = 0.75


_available_backend: BM25Backend | None = None
_fts_ready_cache = TTLCache(ttl_hit=300.0, ttl_miss=30.0)
_content_tsv_cache = TTLCache(ttl_hit=300.0, ttl_miss=30.0)


def _check_extension(extension: str, installed: bool = True) -> bool:
    """Check if a PostgreSQL extension is available or installed."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                if installed:
                    cur.execute("SELECT 1 FROM pg_extension WHERE extname = %s", (extension,))
                else:
                    cur.execute("SELECT 1 FROM pg_available_extensions WHERE name = %s", (extension,))
                return cur.fetchone() is not None
    except Exception:
        return False


def detect_backend() -> BM25Backend:
    """Detect the best available BM25 backend."""
    global _available_backend
    if _available_backend:
        return _available_backend

    for ext, backend in [("pg_search", BM25Backend.PG_SEARCH), ("pg_textsearch", BM25Backend.PG_TEXTSEARCH)]:
        if _check_extension(ext):
            logger.info(f"Using {ext} for BM25 search")
            _available_backend = backend
            return backend

    logger.info("Using native PostgreSQL FTS with ts_rank_cd")
    _available_backend = BM25Backend.NATIVE_FTS
    return _available_backend


def _rows_to_results(rows: list[tuple]) -> list[SearchResult]:
    return [
        SearchResult(filename=r[0], location=str(r[1]) if r[1] else "", content=r[2], score=float(r[3]) if r[3] else 0.0)
        for r in rows
    ]


def _has_bm25_index(table_name: str) -> bool:
    """Check if a BM25 index exists for the table."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_indexes WHERE tablename = %s AND indexdef LIKE %s",
                    (table_name, '%USING bm25%')
                )
                return cur.fetchone() is not None
    except Exception:
        return False


def _search_pg_search(table_name: str, tokens: list[str], top_k: int) -> list[SearchResult]:
    """Search using ParadeDB pg_search extension."""
    if not tokens or not _has_bm25_index(table_name):
        return _search_native_fts(table_name, tokens, top_k) if tokens else []

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(psycopg_sql.SQL("""
                SELECT filename, location, content, paradedb.score(id) AS score
                FROM {}
                WHERE content @@@ %s
                ORDER BY score DESC
                LIMIT %s
            """).format(psycopg_sql.Identifier(table_name)), (" ".join(tokens), top_k))
            return _rows_to_results(cur.fetchall())


def _search_pg_textsearch(table_name: str, tokens: list[str], top_k: int) -> list[SearchResult]:
    """Search using pg_textsearch extension."""
    if not tokens or not _has_bm25_index(table_name):
        return _search_native_fts(table_name, tokens, top_k) if tokens else []

    search_query = " ".join(tokens)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(psycopg_sql.SQL("""
                SELECT filename, location, content, -(content <@> %s) AS score
                FROM {}
                ORDER BY content <@> %s
                LIMIT %s
            """).format(psycopg_sql.Identifier(table_name)), (search_query, search_query, top_k))
            return _rows_to_results(cur.fetchall())


def _search_native_fts(table_name: str, tokens: list[str], top_k: int, config: BM25Config | None = None) -> list[SearchResult]:
    """Search using native PostgreSQL full-text search."""
    if not tokens:
        return []

    with get_connection() as conn:
        with conn.cursor() as cur:
            has_tsv = _content_tsv_cache.get(table_name)
            if has_tsv is None:
                cur.execute(
                    "SELECT 1 FROM information_schema.columns WHERE table_name = %s AND column_name = 'content_tsv'",
                    (table_name,),
                )
                has_tsv = cur.fetchone() is not None
                _content_tsv_cache.set(table_name, has_tsv)

            query_str = " OR ".join(tokens)
            if has_tsv:
                cur.execute(psycopg_sql.SQL("""
                    WITH query AS (SELECT websearch_to_tsquery('english', %s) AS q)
                    SELECT filename, location, content, ts_rank_cd(content_tsv, query.q, 37) AS score
                    FROM {}, query
                    WHERE content_tsv @@ query.q
                    ORDER BY score DESC
                    LIMIT %s
                """).format(psycopg_sql.Identifier(table_name)), (query_str, top_k))
            else:
                cur.execute(psycopg_sql.SQL("""
                    WITH query AS (SELECT websearch_to_tsquery('english', %s) AS q)
                    SELECT filename, location, content, ts_rank_cd(to_tsvector('english', content), query.q, 37) AS score
                    FROM {}, query
                    WHERE to_tsvector('english', content) @@ query.q
                    ORDER BY score DESC
                    LIMIT %s
                """).format(psycopg_sql.Identifier(table_name)), (query_str, top_k))

            rows = cur.fetchall()
            if not rows:
                rows = _search_keyword_fallback(cur, table_name, tokens, top_k)

    return _rows_to_results(rows)


def _search_keyword_fallback(cur, table_name: str, tokens: list[str], top_k: int) -> list[tuple]:
    """Fallback keyword search using ILIKE."""
    if not tokens:
        return []

    patterns = [f"%{t}%" for t in tokens]
    score_expr = " + ".join("(CASE WHEN content ILIKE %s THEN 1 ELSE 0 END)" for _ in tokens)
    where_clause = " OR ".join("content ILIKE %s" for _ in tokens)

    cur.execute(psycopg_sql.SQL("""
        SELECT filename, location, content,
               ({score})::float / %s * (1.0 / (1.0 + ln(greatest(length(content), 1)))) AS score
        FROM {table}
        WHERE {where}
        ORDER BY score DESC
        LIMIT %s
    """).format(
        score=psycopg_sql.SQL(score_expr),
        table=psycopg_sql.Identifier(table_name),
        where=psycopg_sql.SQL(where_clause)
    ), patterns + [len(tokens)] + patterns + [top_k])

    return cur.fetchall()


def ensure_fts_index(repo_name: str) -> bool:
    """Ensure full-text search index exists for a repository."""
    cached = _fts_ready_cache.get(repo_name)
    if cached is not None:
        return cached

    try:
        from .fts_setup import setup_fts_for_repo
        ok = setup_fts_for_repo(repo_name)
    except Exception as e:
        logger.warning(f"Could not set up enhanced FTS for {repo_name}: {e}")
        ok = _ensure_basic_fts_index(repo_name)

    _fts_ready_cache.set(repo_name, ok)
    if ok:
        try:
            _content_tsv_cache.set(get_chunks_table_name(repo_name), True)
        except Exception as e:
            logger.debug(f"Failed to prime content_tsv cache for {repo_name}: {e}")
    return ok


def _ensure_basic_fts_index(repo_name: str) -> bool:
    """Basic FTS index setup as fallback."""
    table_name = get_chunks_table_name(repo_name)
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM information_schema.columns WHERE table_name = %s AND column_name = 'content_tsv'",
                    (table_name,)
                )
                if not cur.fetchone():
                    cur.execute(psycopg_sql.SQL("""
                        ALTER TABLE {}
                        ADD COLUMN IF NOT EXISTS content_tsv tsvector
                        GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED
                    """).format(psycopg_sql.Identifier(table_name)))

                index_name = f"idx_{table_name}_content_tsv"
                cur.execute(psycopg_sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING GIN(content_tsv)").format(
                    psycopg_sql.Identifier(index_name),
                    psycopg_sql.Identifier(table_name)
                ))
                conn.commit()
                _content_tsv_cache.set(table_name, True)
                return True
    except Exception as e:
        logger.error(f"Failed to create FTS index for {table_name}: {e}")
        return False


def bm25_search(repo_name: str, query: str, top_k: int = 50, config: BM25Config | None = None) -> list[SearchResult]:
    """Search repository using BM25 ranking."""
    table_name = get_chunks_table_name(repo_name)
    tokens = tokenize_for_search(query) or [w.lower() for w in query.split() if len(w) >= 2]
    if not tokens:
        return []

    backend = detect_backend()
    if backend == BM25Backend.NATIVE_FTS:
        ensure_fts_index(repo_name)

    try:
        if backend == BM25Backend.PG_SEARCH:
            return _search_pg_search(table_name, tokens, top_k)
        if backend == BM25Backend.PG_TEXTSEARCH:
            return _search_pg_textsearch(table_name, tokens, top_k)
        return _search_native_fts(table_name, tokens, top_k, config)
    except Exception as e:
        logger.warning(f"BM25 search failed with {backend.value}, falling back: {e}")
        try:
            return _search_native_fts(table_name, tokens, top_k, config)
        except Exception as e2:
            raise SearchError(f"BM25 search failed: {e2}") from e2


def get_corpus_stats(repo_name: str) -> dict:
    """Get corpus statistics for BM25 scoring."""
    table_name = get_chunks_table_name(repo_name)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(psycopg_sql.SQL("""
                SELECT COUNT(*), AVG(length(content)), SUM(length(content))
                FROM {}
            """).format(psycopg_sql.Identifier(table_name)))
            row = cur.fetchone()
    return {"doc_count": row[0] or 0, "avg_doc_length": float(row[1]) if row[1] else 0.0, "total_length": row[2] or 0}

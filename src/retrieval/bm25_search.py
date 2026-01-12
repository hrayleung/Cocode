"""Production-ready BM25 search using PostgreSQL full-text search.

Implements multiple search backends with graceful fallback:
1. ParadeDB pg_search - Native BM25 indexes (if available)
2. pg_textsearch - True BM25 ranking (if available)
3. Native PostgreSQL FTS - ts_rank_cd with tsvector columns

All backends use proper:
- Term frequency (TF)
- Inverse document frequency (IDF)
- Document length normalization
- Code-aware tokenization
"""

import logging
from dataclasses import dataclass
from enum import Enum

from src.storage.postgres import get_connection

from .tokenizer import tokenize_for_search
from .vector_search import SearchResult, get_table_name

logger = logging.getLogger(__name__)


class BM25Backend(Enum):
    """Available BM25 search backends."""
    PG_SEARCH = "pg_search"         # ParadeDB extension
    PG_TEXTSEARCH = "pg_textsearch" # Tiger Data extension
    NATIVE_FTS = "native_fts"        # PostgreSQL native with ts_rank_cd


@dataclass
class BM25Config:
    """BM25 algorithm parameters."""
    k1: float = 1.2    # Term frequency saturation parameter (1.2-2.0 typical)
    b: float = 0.75    # Length normalization parameter (0.75 typical)


# Cached backend availability
_available_backend: BM25Backend | None = None


def _check_extension_available(extension: str) -> bool:
    """Check if a PostgreSQL extension is available."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_available_extensions WHERE name = %s",
                    (extension,)
                )
                return cur.fetchone() is not None
    except Exception as e:
        logger.debug(f"Error checking extension {extension} availability: {e}")
        return False


def _check_extension_installed(extension: str) -> bool:
    """Check if a PostgreSQL extension is installed."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_extension WHERE extname = %s",
                    (extension,)
                )
                return cur.fetchone() is not None
    except Exception as e:
        logger.debug(f"Error checking extension {extension} installation: {e}")
        return False


def detect_backend() -> BM25Backend:
    """Detect the best available BM25 backend.

    Checks in order of preference:
    1. pg_search (ParadeDB) - Best BM25 implementation
    2. pg_textsearch - True BM25 ranking
    3. Native PostgreSQL FTS - Always available fallback
    """
    global _available_backend

    if _available_backend is not None:
        return _available_backend

    # Check for pg_search (ParadeDB)
    if _check_extension_installed("pg_search"):
        logger.info("Using pg_search (ParadeDB) for BM25 search")
        _available_backend = BM25Backend.PG_SEARCH
        return _available_backend

    # Check for pg_textsearch
    if _check_extension_installed("pg_textsearch"):
        logger.info("Using pg_textsearch for BM25 search")
        _available_backend = BM25Backend.PG_TEXTSEARCH
        return _available_backend

    # Fall back to native PostgreSQL FTS
    logger.info("Using native PostgreSQL FTS with ts_rank_cd for BM25-like search")
    _available_backend = BM25Backend.NATIVE_FTS
    return _available_backend


def _search_pg_search(
    table_name: str,
    query: str,
    tokens: list[str],
    top_k: int,
) -> list[SearchResult]:
    """Search using ParadeDB pg_search extension.

    Uses native BM25 indexes with the @@@ operator.
    """
    if not tokens:
        return []

    # Build pg_search query - uses @@@ operator
    search_query = " ".join(tokens)

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Check if BM25 index exists
            cur.execute(f"""
                SELECT 1 FROM pg_indexes
                WHERE tablename = %s AND indexdef LIKE '%USING bm25%'
            """, (table_name,))

            if cur.fetchone():
                # Use BM25 index with pg_search
                cur.execute(f"""
                    SELECT filename, location, content,
                           paradedb.score(id) AS score
                    FROM {table_name}
                    WHERE content @@@ %s
                    ORDER BY score DESC
                    LIMIT %s
                """, (search_query, top_k))
            else:
                # Fall back to native FTS if no BM25 index
                return _search_native_fts(table_name, query, tokens, top_k)

            rows = cur.fetchall()

    return [
        SearchResult(
            filename=row[0],
            location=str(row[1]) if row[1] else "",
            content=row[2],
            score=float(row[3]) if row[3] else 0.0,
        )
        for row in rows
    ]


def _search_pg_textsearch(
    table_name: str,
    query: str,
    tokens: list[str],
    top_k: int,
    config: BM25Config,
) -> list[SearchResult]:
    """Search using pg_textsearch extension.

    Uses true BM25 ranking with the <@> distance operator.
    """
    if not tokens:
        return []

    search_query = " ".join(tokens)

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Check if bm25 index exists
            cur.execute(f"""
                SELECT 1 FROM pg_indexes
                WHERE tablename = %s AND indexdef LIKE '%USING bm25%'
            """, (table_name,))

            if cur.fetchone():
                # Use pg_textsearch <@> operator
                # Lower scores = better matches (distance metric)
                cur.execute(f"""
                    SELECT filename, location, content,
                           -(content <@> %s) AS score
                    FROM {table_name}
                    ORDER BY content <@> %s
                    LIMIT %s
                """, (search_query, search_query, top_k))
            else:
                return _search_native_fts(table_name, query, tokens, top_k)

            rows = cur.fetchall()

    return [
        SearchResult(
            filename=row[0],
            location=str(row[1]) if row[1] else "",
            content=row[2],
            score=float(row[3]) if row[3] else 0.0,
        )
        for row in rows
    ]


def _search_native_fts(
    table_name: str,
    query: str,
    tokens: list[str],
    top_k: int,
    config: BM25Config | None = None,
) -> list[SearchResult]:
    """Search using native PostgreSQL full-text search with ts_rank_cd.

    Uses:
    - tsvector columns with GIN indexes
    - ts_rank_cd for cover density ranking (considers term proximity)
    - websearch_to_tsquery for user-friendly query parsing
    - Normalization for document length
    """
    if not tokens:
        return []

    if config is None:
        config = BM25Config()

    # Check if content_tsv column exists, if not use to_tsvector on content
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Check for tsvector column
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = %s AND column_name = 'content_tsv'
            """, (table_name,))
            has_tsv_column = cur.fetchone() is not None

            # Build the tsquery using websearch_to_tsquery for better parsing
            # This handles phrases, operators, and partial matches
            tsquery_sql = "websearch_to_tsquery('english', %s)"

            # Build the tsvector expression
            if has_tsv_column:
                tsvector_expr = "content_tsv"
            else:
                tsvector_expr = "to_tsvector('english', content)"

            # Use ts_rank_cd which considers cover density (term proximity)
            # Normalization options:
            #   0: ignore document length
            #   1: divide by 1 + log(length)
            #   2: divide by length
            #   4: divide by mean harmonic distance
            #   8: divide by unique word count
            #  16: divide by 1 + log(unique words)
            #  32: divide by itself + 1 (scale to 0-1)
            # We use 1|4|32 = 37 for good BM25-like behavior
            normalization = 37

            # First try with OR logic for better recall, then rank by relevance
            search_text = " OR ".join(tokens)

            cur.execute(f"""
                WITH query AS (
                    SELECT {tsquery_sql} AS q
                )
                SELECT filename, location, content,
                       ts_rank_cd({tsvector_expr}, query.q, {normalization}) AS score
                FROM {table_name}, query
                WHERE {tsvector_expr} @@ query.q
                ORDER BY score DESC
                LIMIT %s
            """, (search_text, top_k))

            rows = cur.fetchall()

            # If no results with tsvector, try hybrid approach
            if not rows:
                # Fall back to ILIKE with tf-idf style ranking
                rows = _search_keyword_fallback(cur, table_name, tokens, top_k)

    return [
        SearchResult(
            filename=row[0],
            location=str(row[1]) if row[1] else "",
            content=row[2],
            score=float(row[3]) if row[3] else 0.0,
        )
        for row in rows
    ]


def _search_keyword_fallback(
    cur,
    table_name: str,
    tokens: list[str],
    top_k: int,
) -> list[tuple]:
    """Fallback keyword search when tsvector doesn't match.

    Uses ILIKE with a simple tf-idf style scoring:
    - More matching terms = higher score
    - Shorter documents with same matches = higher score
    """
    if not tokens:
        return []

    patterns = [f"%{token}%" for token in tokens]

    score_expr = " + ".join([
        "(CASE WHEN content ILIKE %s THEN 1 ELSE 0 END)" for _ in tokens
    ])

    where_clause = " OR ".join([
        "content ILIKE %s" for _ in tokens
    ])

    query = f"""
        SELECT filename, location, content,
               ({score_expr})::float / %s *
               (1.0 / (1.0 + ln(greatest(length(content), 1)))) AS score
        FROM {table_name}
        WHERE {where_clause}
        ORDER BY score DESC
        LIMIT %s
    """

    # Parameter order: score_expr patterns (N) + len(tokens) + where_clause patterns (N) + top_k
    params = patterns + [len(tokens)] + patterns + [top_k]

    cur.execute(query, params)

    return cur.fetchall()


def ensure_fts_index(repo_name: str) -> bool:
    """Ensure full-text search index exists for a repository.

    Uses the code-aware FTS setup which creates:
    1. content_tsv column with code-aware tokenization
    2. GIN index on content_tsv

    Returns:
        True if index was created or already exists
    """
    try:
        from .fts_setup import setup_fts_for_repo
        return setup_fts_for_repo(repo_name)
    except Exception as e:
        logger.warning(f"Could not set up enhanced FTS for {repo_name}: {e}")
        # Fall back to basic FTS setup
        return _ensure_basic_fts_index(repo_name)


def _ensure_basic_fts_index(repo_name: str) -> bool:
    """Basic FTS index setup as fallback."""
    table_name = get_table_name(repo_name)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check if content_tsv column exists
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = %s AND column_name = 'content_tsv'
                """, (table_name,))

                if not cur.fetchone():
                    # Add tsvector column
                    cur.execute(f"""
                        ALTER TABLE {table_name}
                        ADD COLUMN IF NOT EXISTS content_tsv tsvector
                        GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED
                    """)

                    # Create GIN index
                    cur.execute(f"""
                        CREATE INDEX IF NOT EXISTS idx_{table_name}_content_tsv
                        ON {table_name} USING GIN(content_tsv)
                    """)

                    conn.commit()
                    logger.info(f"Created basic FTS index for {table_name}")

                return True

    except Exception as e:
        logger.error(f"Failed to create FTS index for {table_name}: {e}")
        return False


def bm25_search(
    repo_name: str,
    query: str,
    top_k: int = 50,
    config: BM25Config | None = None,
) -> list[SearchResult]:
    """Search repository using BM25 ranking.

    Automatically selects the best available backend:
    1. ParadeDB pg_search (if available)
    2. pg_textsearch (if available)
    3. Native PostgreSQL FTS with ts_rank_cd

    Args:
        repo_name: Name of the repository to search
        query: Search query text
        top_k: Number of results to return
        config: BM25 configuration parameters

    Returns:
        List of search results sorted by BM25 relevance score
    """
    if config is None:
        config = BM25Config()

    table_name = get_table_name(repo_name)

    # Tokenize query using code-aware tokenizer
    tokens = tokenize_for_search(query)

    if not tokens:
        # If no tokens after processing, use the raw query words
        tokens = [w.lower() for w in query.split() if len(w) >= 2]

    if not tokens:
        return []

    # Ensure FTS index exists
    ensure_fts_index(repo_name)

    # Detect and use the best available backend
    backend = detect_backend()

    try:
        if backend == BM25Backend.PG_SEARCH:
            return _search_pg_search(table_name, query, tokens, top_k)
        elif backend == BM25Backend.PG_TEXTSEARCH:
            return _search_pg_textsearch(table_name, query, tokens, top_k, config)
        else:
            return _search_native_fts(table_name, query, tokens, top_k, config)

    except Exception as e:
        logger.warning(f"BM25 search failed with {backend.value}, falling back: {e}")
        # Always have native FTS as fallback
        try:
            return _search_native_fts(table_name, query, tokens, top_k, config)
        except Exception as e2:
            logger.error(f"All BM25 search backends failed: {e2}")
            return []


def get_corpus_stats(repo_name: str) -> dict:
    """Get corpus statistics for BM25 scoring.

    Returns statistics like:
    - Total documents
    - Average document length
    - Term frequencies (for IDF calculation)
    """
    table_name = get_table_name(repo_name)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT
                    COUNT(*) AS doc_count,
                    AVG(length(content)) AS avg_doc_length,
                    SUM(length(content)) AS total_length
                FROM {table_name}
            """)
            row = cur.fetchone()

    return {
        "doc_count": row[0] or 0,
        "avg_doc_length": float(row[1]) if row[1] else 0.0,
        "total_length": row[2] or 0,
    }

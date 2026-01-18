"""Code-specific PostgreSQL text search configuration.

Creates a text search configuration optimized for source code that:
- Handles camelCase and snake_case identifiers
- Removes code-specific stop words
- Preserves numbers and symbols in identifiers
"""

from psycopg import sql

from src.storage.postgres import get_connection
import logging
import threading

logger = logging.getLogger(__name__)

_code_text_search_ready: bool | None = None
_code_text_search_lock = threading.Lock()


# SQL to create a code-aware text search configuration
CREATE_CODE_TS_CONFIG = """
-- Create a custom text search configuration based on 'simple'
-- 'simple' doesn't do stemming which is better for code identifiers
DO $$
BEGIN
    -- Create the configuration if it doesn't exist
    IF NOT EXISTS (
        SELECT 1 FROM pg_ts_config WHERE cfgname = 'code'
    ) THEN
        -- Create configuration based on simple (no stemming)
        CREATE TEXT SEARCH CONFIGURATION code (COPY = simple);

        -- Use simple dictionary (no stemming, preserves original tokens)
        -- This is important for code where "function" and "functions" are different
    END IF;
END $$;
"""

# SQL to create a function that normalizes code identifiers
CREATE_CODE_NORMALIZE_FUNC = """
CREATE OR REPLACE FUNCTION normalize_code_text(content TEXT)
RETURNS TEXT AS $$
DECLARE
    result TEXT;
BEGIN
    -- Start with original content
    result := content;

    -- Split camelCase: getUserName -> get User Name
    -- Handle lowercase followed by uppercase
    result := regexp_replace(result, '([a-z])([A-Z])', E'\\1 \\2', 'g');

    -- Handle acronyms followed by regular words: XMLParser -> XML Parser
    result := regexp_replace(result, '([A-Z]+)([A-Z][a-z])', E'\\1 \\2', 'g');

    -- Split snake_case and SCREAMING_SNAKE_CASE
    result := regexp_replace(result, '_+', ' ', 'g');

    -- Split kebab-case
    result := regexp_replace(result, '-+', ' ', 'g');

    -- Remove common code noise (but keep the words)
    result := regexp_replace(result, '[\(\)\[\]\{\}\<\>:;,\.]', ' ', 'g');

    -- Collapse multiple spaces
    result := regexp_replace(result, '\s+', ' ', 'g');

    -- Return trimmed result
    RETURN trim(result);
END;
$$ LANGUAGE plpgsql IMMUTABLE;
"""

# SQL to create enhanced tsvector for code
CREATE_CODE_TSVECTOR_FUNC = """
CREATE OR REPLACE FUNCTION code_to_tsvector(content TEXT)
RETURNS tsvector AS $$
BEGIN
    -- Combine original content with normalized version
    -- This allows searching for either "getUserById" or "user"
    RETURN to_tsvector('english', content) ||
           to_tsvector('simple', normalize_code_text(content));
END;
$$ LANGUAGE plpgsql IMMUTABLE;
"""


def create_code_text_search_config() -> bool:
    """Create PostgreSQL text search configuration for code.

    Creates:
    - 'code' text search configuration
    - normalize_code_text function
    - code_to_tsvector function

    Returns:
        True if successful
    """
    global _code_text_search_ready
    if _code_text_search_ready is not None:
        return _code_text_search_ready

    with _code_text_search_lock:
        if _code_text_search_ready is not None:
            return _code_text_search_ready
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    # Create the code text search configuration
                    cur.execute(CREATE_CODE_TS_CONFIG)

                    # Create the normalization function
                    cur.execute(CREATE_CODE_NORMALIZE_FUNC)

                    # Create the enhanced tsvector function
                    cur.execute(CREATE_CODE_TSVECTOR_FUNC)

                conn.commit()
                logger.info("Created code text search configuration")
                _code_text_search_ready = True
                return True

        except Exception as e:
            logger.error(f"Failed to create code text search config: {e}")
            _code_text_search_ready = False
            return False


def add_code_fts_to_table(table_name: str) -> bool:
    """Add code-aware full-text search column to an existing table.

    Creates:
    - content_tsv column with code-aware tsvector
    - GIN index on content_tsv

    Args:
        table_name: Name of the table to modify

    Returns:
        True if successful
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check if content_tsv column already exists
                cur.execute("""
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = %s AND column_name = 'content_tsv'
                """, (table_name,))
                has_content_tsv = cur.fetchone() is not None

                if not has_content_tsv:
                    # Check if normalize_code_text function exists
                    cur.execute("""
                        SELECT 1 FROM pg_proc WHERE proname = 'normalize_code_text'
                    """)
                    has_normalize_func = cur.fetchone() is not None

                    # Build tsvector expression based on available functions
                    if has_normalize_func:
                        tsvector_expr = """
                            to_tsvector('english', coalesce(content, '')) ||
                            to_tsvector('simple', coalesce(normalize_code_text(content), ''))
                        """
                    else:
                        tsvector_expr = "to_tsvector('english', coalesce(content, ''))"

                    cur.execute(sql.SQL("""
                        ALTER TABLE {}
                        ADD COLUMN content_tsv tsvector
                        GENERATED ALWAYS AS ({}) STORED
                    """).format(
                        sql.Identifier(table_name),
                        sql.SQL(tsvector_expr)
                    ))

                # Create GIN index with sanitized names
                index_name = f"idx_{table_name.replace('.', '_')}_content_tsv"
                cur.execute(sql.SQL("""
                    CREATE INDEX IF NOT EXISTS {}
                    ON {} USING GIN(content_tsv)
                """).format(
                    sql.Identifier(index_name),
                    sql.Identifier(table_name)
                ))

            conn.commit()
            logger.info(f"Ensured code FTS on {table_name}")
            return True

    except Exception as e:
        logger.error(f"Failed to add code FTS to {table_name}: {e}")
        return False


def setup_fts_for_repo(repo_name: str) -> bool:
    """Set up full-text search for a repository."""
    from .vector_search import get_chunks_table_name

    create_code_text_search_config()
    table_name = get_chunks_table_name(repo_name)
    return add_code_fts_to_table(table_name)

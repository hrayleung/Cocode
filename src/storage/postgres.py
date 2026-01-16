"""PostgreSQL connection and helper functions."""

import logging
import threading
from contextlib import contextmanager
from typing import Iterator

from psycopg_pool import ConnectionPool

from config.settings import settings

logger = logging.getLogger(__name__)

# Global connection pool
_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> ConnectionPool:
    """Get or create the connection pool."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ConnectionPool(
                    settings.database_url,
                    min_size=2,
                    max_size=10,
                    open=True,
                )
    return _pool


def close_pool() -> None:
    """Close the connection pool."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


@contextmanager
def get_connection() -> Iterator:
    """Get a connection from the pool.

    Transaction behavior:
    - By default, psycopg connections start in autocommit=False mode
    - Changes must be committed explicitly with conn.commit()
    - The connection is automatically returned to the pool on context exit
    - Use conn.transaction() for explicit transaction boundaries
    - Uncommitted changes are rolled back when the connection is returned

    Usage:
        # Explicit transaction management (recommended)
        with get_connection() as conn:
            with conn.transaction():
                cur.execute(...)  # Automatically committed on success

        # Manual commit/rollback
        with get_connection() as conn:
            try:
                cur.execute(...)
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    """
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


def init_db() -> None:
    """Initialize the database with required extensions and tables."""
    from .schema import create_tables

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Ensure required extensions exist
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            # pgcrypto provides gen_random_uuid() for PostgreSQL < 13
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
        conn.commit()
        create_tables(conn)

    # Set up code-specific text search configuration
    try:
        from src.retrieval.fts_setup import create_code_text_search_config
        create_code_text_search_config()
    except Exception as e:
        logger.warning(f"Could not create code text search config: {e}")

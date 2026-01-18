"""PostgreSQL connection pool and database initialization.

This module provides thread-safe connection pooling for PostgreSQL using psycopg3.
The connection pool is lazily initialized and shared across the application.
"""

import logging
import threading
from contextlib import contextmanager
from typing import Iterator

from psycopg_pool import ConnectionPool

from config.settings import settings

logger = logging.getLogger(__name__)

# Global connection pool (thread-safe singleton)
_pool: ConnectionPool | None = None
_pool_lock = threading.Lock()


def get_pool() -> ConnectionPool:
    """Get or create the connection pool (thread-safe singleton).

    Returns:
        ConnectionPool instance configured with database settings
    """
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
    """Close the connection pool and release all connections."""
    global _pool
    with _pool_lock:
        if _pool is not None:
            _pool.close()
            _pool = None


@contextmanager
def get_connection() -> Iterator:
    """Get a database connection from the pool.

    The connection is automatically returned to the pool when the context exits.
    Uncommitted changes are rolled back automatically.

    Transaction patterns:
        # Explicit transaction (recommended)
        with get_connection() as conn:
            with conn.transaction():
                cur.execute(...)  # Auto-committed on success

        # Manual commit
        with get_connection() as conn:
            cur.execute(...)
            conn.commit()

    Yields:
        Database connection from the pool
    """
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


def init_db() -> None:
    """Initialize database with required extensions and tables.

    Creates:
        - pgvector extension for vector similarity search
        - pgcrypto extension for UUID generation
        - Core application tables (repos, etc.)
        - Code-specific text search configuration
    """
    from .schema import create_tables

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
        conn.commit()
        create_tables(conn)

    # Set up code-aware full-text search
    try:
        from src.retrieval.fts_setup import create_code_text_search_config
        create_code_text_search_config()
    except Exception as e:
        logger.warning(f"Could not create code text search config: {e}")

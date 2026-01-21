"""Database schema definitions for cocode.

This module provides SQL schema definitions and helper functions for creating
and managing database tables, schemas, and identifiers safely.
"""

import re
from psycopg import sql

# Core repos table tracks all indexed repositories
REPOS_TABLE = """
CREATE TABLE IF NOT EXISTS repos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    path TEXT NOT NULL,
    status TEXT DEFAULT 'pending',  -- pending, indexing, ready, error
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_indexed TIMESTAMP,
    file_count INT DEFAULT 0,
    chunk_count INT DEFAULT 0,
    error_message TEXT
);
"""


def validate_schema_name(schema_name: str) -> None:
    """Validate that a schema name is safe and follows PostgreSQL rules.

    Note: When using sql.Identifier(), PostgreSQL allows identifiers that start
    with digits or are reserved keywords because they are properly quoted.
    This validation ensures the name is safe to use with sql.Identifier().

    Args:
        schema_name: The schema name to validate

    Raises:
        ValueError: If the schema name is invalid
    """
    if not schema_name:
        raise ValueError("Schema name cannot be empty")

    # Must match pattern: lowercase alphanumeric and underscores only
    # Note: sql.Identifier() properly quotes identifiers, so digit-leading
    # names (e.g., "2024_project") are allowed by PostgreSQL
    if not re.match(r'^[a-z0-9_]+$', schema_name):
        raise ValueError(
            f"Schema name '{schema_name}' contains invalid characters. "
            "Only lowercase letters, numbers, and underscores are allowed."
        )

    # PostgreSQL identifier length limit is 63 bytes
    if len(schema_name.encode('utf-8')) > 63:
        raise ValueError(f"Schema name '{schema_name}' exceeds PostgreSQL 63-byte limit")


def sanitize_repo_name(repo_name: str) -> str:
    """Sanitize and validate a repository name for use as a PostgreSQL schema name.

    Args:
        repo_name: Repository name to sanitize

    Returns:
        Sanitized schema name

    Raises:
        ValueError: If the sanitized name is invalid
    """
    schema_name = repo_name.replace("-", "_").replace(".", "_").lower()
    validate_schema_name(schema_name)
    return schema_name


def get_create_schema_sql(repo_name: str) -> sql.Composed:
    """Generate SQL to create a schema for a repository.

    Args:
        repo_name: Repository name (will be sanitized and validated)

    Returns:
        Composed SQL query

    Raises:
        ValueError: If the sanitized schema name is invalid
    """
    schema_name = sanitize_repo_name(repo_name)

    return sql.SQL("CREATE SCHEMA IF NOT EXISTS {};").format(
        sql.Identifier(schema_name)
    )


def get_create_chunks_table_sql(repo_name: str, dimensions: int = 3072) -> sql.Composed:
    """Generate SQL to create chunks table for a repository.

    Args:
        repo_name: Repository name (will be sanitized and validated)
        dimensions: Vector embedding dimensions (default: 3072 for text-embedding-3-large)

    Returns:
        Composed SQL query

    Raises:
        ValueError: If the sanitized schema name is invalid
    """
    schema_name = sanitize_repo_name(repo_name)
    chunks_table = sql.Identifier(schema_name, "chunks")

    embedding_index = sql.Identifier(f"{schema_name}_chunks_embedding_idx")
    content_tsv_index = sql.Identifier(f"{schema_name}_chunks_content_tsv_idx")
    filename_index = sql.Identifier(f"{schema_name}_chunks_filename_idx")

    return sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS {chunks_table} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            filename TEXT NOT NULL,
            location TEXT,  -- e.g., "10:50" for lines 10-50
            content TEXT NOT NULL,
            embedding vector({dimensions}),
            content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS {embedding_index}
        ON {chunks_table} USING hnsw (embedding vector_cosine_ops);

        CREATE INDEX IF NOT EXISTS {content_tsv_index}
        ON {chunks_table} USING GIN (content_tsv);

        CREATE INDEX IF NOT EXISTS {filename_index}
        ON {chunks_table} (filename);
        """
    ).format(
        chunks_table=chunks_table,
        dimensions=sql.Literal(dimensions),
        embedding_index=embedding_index,
        content_tsv_index=content_tsv_index,
        filename_index=filename_index,
    )


def create_tables(conn) -> None:
    """Create core tables if they don't exist."""
    with conn.cursor() as cur:
        cur.execute(REPOS_TABLE)


def get_create_symbols_table_sql(repo_name: str, dimensions: int = 3072) -> sql.Composed:
    """Generate SQL to create symbols table for a repository.

    The symbols table stores function/class/method-level information for
    symbol-based search and indexing.

    Args:
        repo_name: Repository name (will be sanitized and validated)
        dimensions: Vector embedding dimensions (default: 3072 for text-embedding-3-large)

    Returns:
        Composed SQL query

    Raises:
        ValueError: If the sanitized schema name is invalid
    """
    schema_name = sanitize_repo_name(repo_name)
    symbols_table = sql.Identifier(schema_name, "symbols")

    embedding_index = sql.Identifier(f"{schema_name}_symbols_embedding_idx")
    content_tsv_index = sql.Identifier(f"{schema_name}_symbols_content_tsv_idx")
    filename_index = sql.Identifier(f"{schema_name}_symbols_filename_idx")
    name_index = sql.Identifier(f"{schema_name}_symbols_name_idx")
    type_index = sql.Identifier(f"{schema_name}_symbols_type_idx")

    return sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS {symbols_table} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            filename TEXT NOT NULL,
            symbol_name TEXT NOT NULL,
            symbol_type TEXT NOT NULL,  -- 'function', 'class', 'method', 'interface'
            line_start INT NOT NULL,
            line_end INT NOT NULL,
            signature TEXT,  -- e.g., "def authenticate(user: str, pwd: str) -> bool"
            docstring TEXT,
            parent_symbol TEXT,  -- For methods: parent class name
            visibility TEXT,  -- 'public', 'private', 'internal'
            category TEXT,  -- 'implementation', 'test', 'api', 'config'
            embedding vector({dimensions}),
            content_tsv tsvector,
            batch_id UUID,  -- Identifies indexing batch for deterministic cleanup
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (filename, symbol_name, line_start)
        );

        CREATE INDEX IF NOT EXISTS {embedding_index}
        ON {symbols_table} USING hnsw (embedding vector_cosine_ops);

        CREATE INDEX IF NOT EXISTS {content_tsv_index}
        ON {symbols_table} USING GIN (content_tsv);

        CREATE INDEX IF NOT EXISTS {filename_index}
        ON {symbols_table} (filename);

        CREATE INDEX IF NOT EXISTS {name_index}
        ON {symbols_table} (symbol_name);

        CREATE INDEX IF NOT EXISTS {type_index}
        ON {symbols_table} (symbol_type);
        """
    ).format(
        symbols_table=symbols_table,
        dimensions=sql.Literal(dimensions),
        embedding_index=embedding_index,
        content_tsv_index=content_tsv_index,
        filename_index=filename_index,
        name_index=name_index,
        type_index=type_index,
    )


def get_create_edges_table_sql(repo_name: str) -> sql.Composed:
    """Generate SQL to create edges table for a repository.

    The edges table stores relationships between symbols (function calls,
    inheritance, implementations, etc.) to build a call graph.

    Args:
        repo_name: Repository name (will be sanitized and validated)

    Returns:
        Composed SQL query

    Raises:
        ValueError: If the sanitized schema name is invalid
    """
    schema_name = sanitize_repo_name(repo_name)
    edges_table = sql.Identifier(schema_name, "edges")
    symbols_table = sql.Identifier(schema_name, "symbols")

    source_idx = sql.Identifier(f"{schema_name}_edges_source_idx")
    target_idx = sql.Identifier(f"{schema_name}_edges_target_idx")
    type_idx = sql.Identifier(f"{schema_name}_edges_type_idx")
    files_idx = sql.Identifier(f"{schema_name}_edges_files_idx")

    return sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS {edges_table} (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            source_symbol_id UUID REFERENCES {symbols_table}(id) ON DELETE CASCADE,
            target_symbol_id UUID,  -- NULL if unresolved (external library, dynamic call)
            edge_type TEXT NOT NULL,  -- 'calls', 'implements', 'extends', 'overrides'
            source_file TEXT NOT NULL,
            source_line INT,  -- Line where the call/reference occurs
            target_file TEXT,  -- NULL if unresolved
            target_symbol_name TEXT,  -- Symbol name being called (for unresolved tracking)
            target_line INT,
            confidence FLOAT DEFAULT 1.0,  -- 1.0=exact match, 0.7=partial, 0.5=unresolved
            context TEXT,  -- e.g., "called in loop", "conditional call", "recursive"
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS {source_idx}
        ON {edges_table} (source_symbol_id);

        CREATE INDEX IF NOT EXISTS {target_idx}
        ON {edges_table} (target_symbol_id);

        CREATE INDEX IF NOT EXISTS {type_idx}
        ON {edges_table} (edge_type);

        CREATE INDEX IF NOT EXISTS {files_idx}
        ON {edges_table} (source_file, target_file);
        """
    ).format(
        edges_table=edges_table,
        symbols_table=symbols_table,
        source_idx=source_idx,
        target_idx=target_idx,
        type_idx=type_idx,
        files_idx=files_idx,
    )


def get_create_graph_cache_table_sql(repo_name: str) -> sql.Composed:
    """Generate SQL to create graph_cache table for a repository.

    The graph_cache table stores pre-computed import/dependency graphs
    to avoid rebuilding on every search.

    Args:
        repo_name: Repository name (will be sanitized and validated)

    Returns:
        Composed SQL query

    Raises:
        ValueError: If the sanitized schema name is invalid
    """
    schema_name = sanitize_repo_name(repo_name)
    cache_table = sql.Identifier(schema_name, "graph_cache")

    updated_idx = sql.Identifier(f"{schema_name}_graph_cache_updated_idx")
    filename_idx = sql.Identifier(f"{schema_name}_graph_cache_filename_idx")

    return sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS {cache_table} (
            filename TEXT PRIMARY KEY,
            imports JSONB NOT NULL DEFAULT '[]'::jsonb,  -- Files this file imports
            imported_by JSONB NOT NULL DEFAULT '[]'::jsonb,  -- Files that import this file
            symbol_count INT DEFAULT 0,
            edge_count INT DEFAULT 0,  -- Number of call edges from this file
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS {updated_idx}
        ON {cache_table} (last_updated);

        CREATE INDEX IF NOT EXISTS {filename_idx}
        ON {cache_table} (filename);
        """
    ).format(
        cache_table=cache_table,
        updated_idx=updated_idx,
        filename_idx=filename_idx,
    )


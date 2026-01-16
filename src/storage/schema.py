"""Database schema definitions."""

import re
from psycopg import sql

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

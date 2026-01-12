"""Database schema definitions."""

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


def get_create_schema_sql(repo_name: str) -> sql.Composed:
    """Generate SQL to create a schema for a repository."""
    schema_name = repo_name.replace("-", "_").replace(".", "_").lower()
    return sql.SQL("CREATE SCHEMA IF NOT EXISTS {};").format(
        sql.Identifier(schema_name)
    )


def get_create_chunks_table_sql(repo_name: str, dimensions: int = 3072) -> sql.Composed:
    """Generate SQL to create chunks table for a repository."""
    schema_name = repo_name.replace("-", "_").replace(".", "_").lower()
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

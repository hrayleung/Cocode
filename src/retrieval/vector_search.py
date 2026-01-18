"""Vector search using pgvector."""

import logging
import re

from psycopg import sql

from config.settings import settings
from src.embeddings.openai import get_embedding
from src.models import SearchResult
from src.storage.postgres import get_connection

logger = logging.getLogger(__name__)


def sanitize_identifier(name: str) -> str:
    """Sanitize a name for safe use as a PostgreSQL identifier.

    Uses a whitelist approach - only allows lowercase alphanumeric and underscore.
    Raises ValueError if the name is empty after sanitization.
    """
    sanitized = name.lower()
    sanitized = re.sub(r'[-. ]+', '_', sanitized)
    sanitized = re.sub(r'[^a-z0-9_]', '', sanitized)
    sanitized = sanitized.strip('_')
    if not sanitized:
        raise ValueError(f"Invalid identifier after sanitization: {name!r}")
    return sanitized


def get_chunks_table_name(repo_name: str) -> str:
    """Get the CocoIndex chunks table name for a repository.

    Table name format: codeindex_{repo}__{repo}_chunks
    """
    safe_name = sanitize_identifier(repo_name)
    return f"codeindex_{safe_name}__{safe_name}_chunks"


def vector_search(
    repo_name: str,
    query: str,
    top_k: int = 50,
    query_embedding: list[float] | None = None,
) -> list[SearchResult]:
    """Search repository using vector similarity.

    Args:
        repo_name: Name of the repository to search
        query: Search query text
        top_k: Number of results to return
        query_embedding: Pre-computed query embedding (optional)

    Returns:
        List of search results sorted by similarity
    """
    # Get query embedding if not provided
    if query_embedding is None:
        try:
            query_embedding = get_embedding(query)
            if not query_embedding or len(query_embedding) == 0:
                logger.error(f"Empty embedding generated for query: {query[:50]}...")
                raise ValueError("Empty embedding generated")
        except Exception as e:
            logger.error(f"Failed to generate embedding for query: {e}")
            raise ValueError(f"Embedding generation failed: {e}") from e

    # Validate embedding dimensions match expected
    expected_dim = len(query_embedding)
    if expected_dim != settings.embedding_dimensions:
        logger.warning(
            f"Embedding dimension mismatch: got {expected_dim}, "
            f"expected {settings.embedding_dimensions}"
        )

    table_name = get_chunks_table_name(repo_name)

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Use cosine distance (<=> operator) for similarity search
            cur.execute(
                sql.SQL("""
                SELECT filename, location, content,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """).format(sql.Identifier(table_name)),
                (query_embedding, query_embedding, top_k),
            )
            rows = cur.fetchall()

    return [
        SearchResult(
            filename=row[0],
            location=str(row[1]) if row[1] else "",
            content=row[2],
            score=float(row[3]),
        )
        for row in rows
    ]

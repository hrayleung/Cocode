"""Vector search using pgvector."""

import logging
import re
from dataclasses import dataclass

from config.settings import settings
from src.embeddings.openai import get_embedding
from src.storage.postgres import get_connection

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result with metadata."""

    filename: str
    location: str
    content: str
    score: float


def sanitize_table_name(name: str) -> str:
    """Sanitize a name for safe use as a PostgreSQL identifier.

    Uses a whitelist approach - only allows lowercase alphanumeric and underscore.
    Raises ValueError if the name is empty after sanitization.

    Note: Does NOT add prefixes to preserve backward compatibility with existing
    indexes. PostgreSQL handles numeric-starting identifiers in table names.
    """
    # Lowercase and replace common separators with underscores
    sanitized = name.lower()
    sanitized = re.sub(r'[-. ]+', '_', sanitized)
    # Remove any remaining invalid characters (whitelist approach)
    sanitized = re.sub(r'[^a-z0-9_]', '', sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Validate we have something left
    if not sanitized:
        raise ValueError(f"Invalid table name after sanitization: {name!r}")
    return sanitized


def get_table_name(repo_name: str) -> str:
    """Get the CocoIndex table name for a repository.

    Sanitizes the repo_name to prevent SQL injection.
    """
    # Sanitize the repo name first
    safe_name = sanitize_table_name(repo_name)
    # CocoIndex naming: {flow_name}__{target_name}
    # Flow name: CodeIndex_{repo_name}
    # Target name: {repo_name}_chunks
    flow_name = f"codeindex_{safe_name}"
    target_name = f"{safe_name}_chunks"
    return f"{flow_name}__{target_name}"


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

    table_name = get_table_name(repo_name)

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Use cosine distance (<=> operator) for similarity search
            cur.execute(
                f"""
                SELECT filename, location, content,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM {table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
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

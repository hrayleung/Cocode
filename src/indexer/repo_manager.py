"""Repository manager for tracking indexed codebases."""

import logging
from dataclasses import dataclass
from datetime import datetime

from psycopg import sql as psycopg_sql

from config.settings import settings
from src.retrieval.vector_search import get_chunks_table_name
from src.storage.postgres import get_connection
from src.storage.schema import get_create_chunks_table_sql, get_create_schema_sql

logger = logging.getLogger(__name__)


@dataclass
class RepoInfo:
    """Repository information."""
    id: str
    name: str
    path: str
    status: str
    created_at: datetime
    last_indexed: datetime | None
    file_count: int
    chunk_count: int
    error_message: str | None = None


class RepoManager:
    """Manages repository metadata in the database."""

    def register_repo(self, name: str, path: str) -> RepoInfo:
        """Register or update a repository.

        Args:
            name: Unique name for the repository
            path: Absolute path to the repository

        Returns:
            RepoInfo with registration details
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check if repo exists
                cur.execute("SELECT id FROM repos WHERE name = %s", (name,))
                existing = cur.fetchone()

                if existing:
                    cur.execute(
                        """
                        UPDATE repos SET path = %s, status = 'pending'
                        WHERE name = %s
                        RETURNING id, name, path, status, created_at, last_indexed,
                                  file_count, chunk_count, error_message
                        """,
                        (path, name),
                    )
                else:
                    cur.execute(get_create_schema_sql(name))
                    cur.execute(get_create_chunks_table_sql(name, settings.embedding_dimensions))

                    # Insert new repo
                    cur.execute(
                        """
                        INSERT INTO repos (name, path, status)
                        VALUES (%s, %s, 'pending')
                        RETURNING id, name, path, status, created_at, last_indexed,
                                  file_count, chunk_count, error_message
                        """,
                        (name, path),
                    )

                row = cur.fetchone()
                conn.commit()

        return self._row_to_repo_info(row)

    def get_repo(self, name: str) -> RepoInfo | None:
        """Get repository by name."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, name, path, status, created_at, last_indexed,
                           file_count, chunk_count, error_message
                    FROM repos WHERE name = %s
                    """,
                    (name,),
                )
                row = cur.fetchone()

        return self._row_to_repo_info(row) if row else None

    def update_status(
        self,
        name: str,
        status: str,
        file_count: int | None = None,
        chunk_count: int | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update repository status."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Build update clauses safely using sql.SQL
                update_parts = [psycopg_sql.SQL("status = %s")]
                params = [status]

                if status == "ready":
                    update_parts.append(psycopg_sql.SQL("last_indexed = (NOW() AT TIME ZONE 'UTC')"))

                if file_count is not None:
                    update_parts.append(psycopg_sql.SQL("file_count = %s"))
                    params.append(file_count)

                if chunk_count is not None:
                    update_parts.append(psycopg_sql.SQL("chunk_count = %s"))
                    params.append(chunk_count)

                if error_message is not None:
                    update_parts.append(psycopg_sql.SQL("error_message = %s"))
                    params.append(error_message)
                elif status != "error":
                    update_parts.append(psycopg_sql.SQL("error_message = NULL"))

                params.append(name)

                # Safely construct the query
                query = psycopg_sql.SQL("UPDATE repos SET {updates} WHERE name = %s").format(
                    updates=psycopg_sql.SQL(", ").join(update_parts)
                )
                cur.execute(query, params)
            conn.commit()

    def get_chunk_count(self, name: str) -> int:
        """Get number of indexed chunks for a repository."""
        return self._execute_count_query(name, "SELECT COUNT(*) FROM {table}")

    def get_file_count(self, name: str) -> int:
        """Get number of unique files indexed for a repository."""
        return self._execute_count_query(name, "SELECT COUNT(DISTINCT filename) FROM {table}")

    def _execute_count_query(self, name: str, query_template: str) -> int:
        """Execute a count query on the chunks table."""
        table_name = get_chunks_table_name(name)
        with get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    query = psycopg_sql.SQL(query_template).format(table=psycopg_sql.Identifier(table_name))
                    cur.execute(query)
                    return cur.fetchone()[0]
                except Exception as e:
                    logger.debug(f"Error executing count query for {name}: {e}")
                    return 0

    @staticmethod
    def _row_to_repo_info(row: tuple) -> RepoInfo:
        """Convert database row to RepoInfo."""
        return RepoInfo(
            id=str(row[0]),
            name=row[1],
            path=row[2],
            status=row[3],
            created_at=row[4],
            last_indexed=row[5],
            file_count=row[6],
            chunk_count=row[7],
            error_message=row[8],
        )

"""Repository manager for tracking indexed codebases."""

import logging
from dataclasses import dataclass
from datetime import datetime

from config.settings import settings
from src.storage.postgres import get_connection
from src.storage.schema import get_create_chunks_table_sql, get_create_schema_sql
from src.retrieval.vector_search import get_chunks_table_name

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
                updates = ["status = %s"]
                params = [status]

                if status == "ready":
                    updates.append("last_indexed = (NOW() AT TIME ZONE 'UTC')")

                if file_count is not None:
                    updates.append("file_count = %s")
                    params.append(file_count)

                if chunk_count is not None:
                    updates.append("chunk_count = %s")
                    params.append(chunk_count)

                if error_message is not None:
                    updates.append("error_message = %s")
                    params.append(error_message)
                elif status != "error":
                    updates.append("error_message = NULL")

                params.append(name)

                cur.execute(
                    f"UPDATE repos SET {', '.join(updates)} WHERE name = %s",
                    params,
                )
            conn.commit()

    def get_chunk_count(self, name: str) -> int:
        """Get number of indexed chunks for a repository."""
        table_name = get_chunks_table_name(name)

        with get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                    return cur.fetchone()[0]
                except Exception as e:
                    logger.debug(f"Error counting chunks for {name}: {e}")
                    return 0

    def get_file_count(self, name: str) -> int:
        """Get number of unique files indexed for a repository."""
        table_name = get_chunks_table_name(name)

        with get_connection() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(f"SELECT COUNT(DISTINCT filename) FROM {table_name}")
                    return cur.fetchone()[0]
                except Exception as e:
                    logger.debug(f"Error counting files for {name}: {e}")
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

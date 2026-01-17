"""Indexer service - handles all codebase indexing operations."""

import fnmatch
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cocoindex
from psycopg import sql

from config.settings import settings
from src.exceptions import IndexingError, PathError
from src.indexer.flow import create_indexing_flow
from src.indexer.repo_manager import RepoManager
from src.retrieval.centrality import compute_and_store_centrality, delete_centrality_table
from src.retrieval.vector_search import get_chunks_table_name

logger = logging.getLogger(__name__)


def _matches_any_pattern(name: str, patterns: list[str]) -> bool:
    """Check if a name matches any of the given fnmatch patterns."""
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


@dataclass
class IndexResult:
    """Result of an indexing operation."""
    repo_name: str
    status: str  # 'created', 'updated', 'unchanged', 'error'
    file_count: int = 0
    chunk_count: int = 0
    message: str | None = None


class IndexerService:
    """Service for indexing codebases.

    Handles automatic indexing on first use and incremental updates.
    """

    def __init__(self):
        self._repo_manager = RepoManager()
        self._initialized = False
        # Cache for file change checks on repeated queries.
        # Keys are repo_name; values are (last_indexed_ts, checked_at_monotonic, changed)
        self._change_check_cache: dict[str, tuple[float, float, bool]] = {}

    def _init_cocoindex(self) -> None:
        """Initialize CocoIndex with required environment variables."""
        if self._initialized:
            return

        os.environ["COCOINDEX_DATABASE_URL"] = settings.database_url
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        cocoindex.init()
        self._initialized = True

    @staticmethod
    def path_to_repo_name(path: str) -> str:
        """Convert a path to a consistent repository name.

        Uses a whitelist approach - only allows lowercase alphanumeric and underscore.
        Does NOT add prefixes to preserve backward compatibility with existing indexes.
        """
        import re
        resolved = Path(path).resolve()
        # Use directory name, sanitized for PostgreSQL
        name = resolved.name.lower()
        # Replace common separators with underscores
        name = re.sub(r'[-. ]+', '_', name)
        # Remove any remaining invalid characters (whitelist approach)
        name = re.sub(r'[^a-z0-9_]', '', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        # Ensure we have a valid name
        if not name:
            name = 'repo'
        return name

    def resolve_repo_name(self, path: str | Path) -> str:
        """Resolve a stable repo name for a given path.

        Uses the directory-based name when unambiguous. If that name is already
        taken by a different path, appends a short hash suffix to avoid
        collisions (allowing multiple repos with the same folder name).

        Returns:
            A unique repo name string for this path.
        """
        import hashlib

        resolved = Path(path).resolve()
        resolved_str = str(resolved)
        base = self.path_to_repo_name(resolved_str)

        def get_repo_safe(name: str):
            """Get repo, returning None on any error (including DB issues)."""
            try:
                return self._repo_manager.get_repo(name)
            except Exception as e:
                logger.debug(f"Could not lookup repo '{name}': {e}")
                return None

        # Check if base name is already registered for this exact path
        existing_base = get_repo_safe(base)
        if existing_base is not None and existing_base.path == resolved_str:
            return base

        # Check if a hashed variant is already registered for this path
        digest = hashlib.sha1(resolved_str.encode("utf-8")).hexdigest()
        for prefix_len in (8, 12, 16, 40):
            hashed = f"{base}_{digest[:prefix_len]}"
            existing_hashed = get_repo_safe(hashed)
            if existing_hashed is not None and existing_hashed.path == resolved_str:
                return hashed

        # No existing registration found - use base if available
        if existing_base is None:
            return base

        # Base name is taken by a different path: find an available hashed variant
        for prefix_len in (8, 12, 16, 40):
            hashed = f"{base}_{digest[:prefix_len]}"
            if get_repo_safe(hashed) is None:
                return hashed

        # Fallback: full hash (should never happen in practice)
        return f"{base}_{digest}"

    def _validate_path(self, path: str) -> Path:
        """Validate and resolve a path."""
        resolved = Path(path).resolve()

        if not resolved.exists():
            raise PathError(f"Path does not exist: {path}")
        if not resolved.is_dir():
            raise PathError(f"Path is not a directory: {path}")

        return resolved

    def _get_stats(self, repo_name: str) -> tuple[int, int]:
        """Get current file and chunk counts."""
        file_count = self._repo_manager.get_file_count(repo_name)
        chunk_count = self._repo_manager.get_chunk_count(repo_name)
        return file_count, chunk_count

    def _compute_centrality(self, repo_name: str) -> None:
        """Compute and store centrality scores, logging any failures."""
        try:
            compute_and_store_centrality(repo_name)
        except Exception as e:
            logger.warning(f"Centrality computation failed for {repo_name}: {e}")

    def _get_indexed_files(self, repo_name: str, repo_path: Path) -> tuple[set[str], set[str]]:
        """Get indexed files from DB and current files on disk."""
        from src.storage.postgres import get_connection

        table_name = get_chunks_table_name(repo_name)

        # Get stored files from DB
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Use sql.Identifier to prevent SQL injection
                query = sql.SQL("SELECT DISTINCT filename FROM {}").format(
                    sql.Identifier(table_name)
                )
                cur.execute(query)
                stored_files = {row[0] for row in cur.fetchall()}

        # Get current files on disk
        excluded_patterns = settings.excluded_patterns

        def is_excluded(file_path: Path) -> bool:
            """Check if a file path matches any excluded pattern."""
            return any(
                _matches_any_pattern(part, excluded_patterns)
                for part in file_path.parts
            )

        current_files = set()
        for file_path in repo_path.rglob("*"):
            if not (file_path.is_file() and
                    not file_path.name.startswith('.') and
                    file_path.suffix.lower() in settings.included_extensions):
                continue

            try:
                relative_path = file_path.relative_to(repo_path)
            except ValueError:
                continue

            if not is_excluded(relative_path):
                current_files.add(str(relative_path))

        return stored_files, current_files

    def _has_files_changed(self, repo_name: str, repo_path: str) -> bool:
        """Check if any indexed files have changed.

        Note: This is called on every query via ensure_indexed(). Keep it fast.

        Detects changes by checking whether any relevant file has mtime > last_indexed.

        This avoids expensive DB scans (COUNT(DISTINCT ...) / SELECT DISTINCT ...) and avoids
        `Path.rglob()` traversal into large excluded folders like node_modules by pruning during walk.

        Deleted files are not detected (stale index entries persist until the next update).
        """
        repo = self._repo_manager.get_repo(repo_name)
        if not (repo and repo.last_indexed):
            return False

        last_indexed_ts = repo.last_indexed.timestamp()
        now = time.monotonic()

        cached = self._change_check_cache.get(repo_name)
        if cached and cached[0] == last_indexed_ts and (now - cached[1]) < 5.0:
            return cached[2]

        repo_path_obj = Path(repo_path)

        excluded_patterns = settings.excluded_patterns
        included_extensions = {ext.lower() for ext in settings.included_extensions}

        # Walk and prune excluded directories to avoid scanning huge trees.
        changed = False
        for root, dirnames, filenames in os.walk(repo_path, topdown=True):
            dirnames[:] = [
                d for d in dirnames
                if not d.startswith(".") and not _matches_any_pattern(d, excluded_patterns)
            ]

            for filename in filenames:
                if filename.startswith(".") or _matches_any_pattern(filename, excluded_patterns):
                    continue

                ext = os.path.splitext(filename)[1].lower()
                if ext not in included_extensions:
                    continue

                full_path = Path(root) / filename
                try:
                    relative_path = full_path.relative_to(repo_path_obj)
                except ValueError:
                    continue

                # Match historical behavior: exclude any path containing an excluded part.
                if any(_matches_any_pattern(part, excluded_patterns) for part in relative_path.parts):
                    continue

                try:
                    if full_path.stat().st_mtime > last_indexed_ts:
                        logger.debug(f"Modified file detected: {relative_path}")
                        changed = True
                        break
                except OSError:
                    # File might have been deleted/moved mid-walk; trigger re-index.
                    changed = True
                    break

            if changed:
                break

        self._change_check_cache[repo_name] = (last_indexed_ts, now, changed)
        return changed

    def _verify_index_data_exists(self, repo_name: str) -> bool:
        """Verify that indexed data actually exists in the database."""
        from src.storage.postgres import get_connection

        chunks_table = get_chunks_table_name(repo_name)

        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = %s
                        )
                    """, (chunks_table,))

                    table_exists = cur.fetchone()[0]
                    if not table_exists:
                        logger.warning(f"Chunks table does not exist: {chunks_table}")
                        return False

                    # Use sql.Identifier to prevent SQL injection
                    count_query = sql.SQL("SELECT COUNT(*) FROM {}").format(
                        sql.Identifier(chunks_table)
                    )
                    cur.execute(count_query)
                    chunk_count = cur.fetchone()[0]

                    if chunk_count == 0:
                        logger.warning(f"Chunks table exists but has no data: {chunks_table}")
                        return False

                    logger.debug(f"Verified {chunk_count} chunks in {chunks_table}")
                    return True

        except Exception as e:
            logger.error(f"Error verifying index data for {repo_name}: {e}")
            return False

    def ensure_indexed(self, path: str) -> IndexResult:
        """Ensure a codebase is indexed, creating or updating as needed.

        Args:
            path: Path to the codebase directory

        Returns:
            IndexResult with status and statistics

        Raises:
            PathError: If path is invalid
            IndexingError: If indexing fails
        """
        resolved_path = self._validate_path(path)
        repo_name = self.resolve_repo_name(resolved_path)
        repo_path = str(resolved_path)

        repo = self._repo_manager.get_repo(repo_name)

        # Already indexed - check if files changed before incremental update
        if repo and repo.status == "ready" and repo.path == repo_path:
            if self._has_files_changed(repo_name, repo_path):
                logger.debug(f"Files changed for {repo_name}, running incremental update")
                return self._incremental_update(repo_name, repo_path)
            else:
                logger.debug(f"No files changed for {repo_name}, skipping update")
                return IndexResult(
                    repo_name=repo_name,
                    status="unchanged",
                    file_count=repo.file_count,
                    chunk_count=repo.chunk_count,
                )

        # Path changed - re-index
        if repo and repo.path != repo_path:
            logger.info(f"Path changed for {repo_name}, re-indexing")

        # New or needs re-indexing
        return self._full_index(repo_name, repo_path)

    def _delete_index(self, repo_name: str) -> None:
        """Delete an existing index by dropping database tables.

        Uses explicit transaction to ensure atomic deletion of all related tables.
        """
        from src.storage.postgres import get_connection
        from src.retrieval.vector_search import sanitize_identifier
        from src.indexer.flow import clear_flow_cache
        from src.storage.schema import sanitize_repo_name

        safe_name = sanitize_identifier(repo_name)
        chunks_table = get_chunks_table_name(repo_name)
        tracking_table = f"codeindex_{safe_name}__cocoindex_tracking"

        # Close any cached flow handles to avoid stale state during re-index.
        clear_flow_cache(repo_name)

        try:
            with get_connection() as conn:
                # Use explicit transaction for atomic deletion
                with conn.transaction():
                    with conn.cursor() as cur:
                        # Use sql.Identifier to prevent SQL injection
                        drop_chunks = sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                            sql.Identifier(chunks_table)
                        )
                        drop_tracking = sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
                            sql.Identifier(tracking_table)
                        )
                        cur.execute(drop_chunks)
                        cur.execute(drop_tracking)
                        cur.execute("DELETE FROM repos WHERE name = %s", (repo_name,))

                        # Drop the per-repo schema used for symbols/edges/cache.
                        schema_name = sanitize_repo_name(repo_name)
                        cur.execute(
                            sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                                sql.Identifier(schema_name)
                            )
                        )
                    # Transaction automatically commits on success
            logger.info(f"Deleted index tables for {repo_name}")
        except Exception as e:
            logger.warning(f"Could not delete index for {repo_name}: {e}")

        delete_centrality_table(repo_name)

    def _incremental_update(self, repo_name: str, repo_path: str) -> IndexResult:
        """Perform incremental update on an existing index.

        CocoIndex manages its own database transactions internally.
        If any step fails, the index state is preserved.
        """
        try:
            self._init_cocoindex()

            flow = create_indexing_flow(repo_name, repo_path)

            # CocoIndex manages its own DB connections - no transaction wrapper needed
            flow.setup()
            logger.debug(f"Completed setup() for {repo_name}")

            flow.update()
            logger.debug(f"Completed update() for {repo_name}")

            # Keep symbol index in sync with chunk index.
            try:
                from src.indexer.symbol_indexing import index_repository_symbols

                index_repository_symbols(repo_name, repo_path)
            except Exception as e:
                logger.warning(f"Symbol indexing failed for {repo_name}: {e}")

            file_count, chunk_count = self._get_stats(repo_name)

            if not self._verify_index_data_exists(repo_name):
                logger.warning(f"Index verification failed for {repo_name}: no data found after incremental update")

            self._repo_manager.update_status(
                repo_name, "ready",
                file_count=file_count,
                chunk_count=chunk_count
            )

            self._compute_centrality(repo_name)

            logger.debug(f"Incremental update for {repo_name}: {file_count} files, {chunk_count} chunks")

            return IndexResult(
                repo_name=repo_name,
                status="updated",
                file_count=file_count,
                chunk_count=chunk_count,
            )

        except Exception as e:
            logger.warning(f"Incremental update failed for {repo_name}: {e}")
            file_count, chunk_count = self._get_stats(repo_name)
            return IndexResult(
                repo_name=repo_name,
                status="unchanged",
                file_count=file_count,
                chunk_count=chunk_count,
                message=f"Using existing index (update failed: {e})",
            )

    def _full_index(self, repo_name: str, repo_path: str) -> IndexResult:
        """Perform full indexing of a codebase.

        CocoIndex manages its own database transactions internally.
        If any step fails, the repo status is updated accordingly.
        """
        self._repo_manager.register_repo(repo_name, repo_path)
        self._repo_manager.update_status(repo_name, "indexing")

        try:
            self._init_cocoindex()

            logger.info(f"Indexing {repo_name} from {repo_path}")

            flow = create_indexing_flow(repo_name, repo_path)

            try:
                flow.drop()
            except Exception as e:
                logger.debug(f"No existing flow to drop: {e}")

            # CocoIndex manages its own DB connections - no transaction wrapper needed
            flow.setup()
            logger.debug(f"Completed setup() for {repo_name}")

            flow.update()
            logger.debug(f"Completed update() for {repo_name}")

            # Build/refresh symbol index for symbol-level retrieval and line-level locations.
            try:
                from src.indexer.symbol_indexing import index_repository_symbols

                index_repository_symbols(repo_name, repo_path)
            except Exception as e:
                logger.warning(f"Symbol indexing failed for {repo_name}: {e}")

            file_count, chunk_count = self._get_stats(repo_name)

            if not self._verify_index_data_exists(repo_name):
                logger.warning(f"Index verification failed for {repo_name}: no data found after indexing")

            self._repo_manager.update_status(
                repo_name, "ready",
                file_count=file_count,
                chunk_count=chunk_count
            )

            self._compute_centrality(repo_name)

            logger.info(f"Indexed {repo_name}: {file_count} files, {chunk_count} chunks")

            return IndexResult(
                repo_name=repo_name,
                status="created",
                file_count=file_count,
                chunk_count=chunk_count,
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Indexing failed for {repo_name}: {error_msg}")
            self._repo_manager.update_status(repo_name, "error", error_message=error_msg)
            raise IndexingError(f"Failed to index {repo_path}: {error_msg}") from e


# Singleton instance
_indexer: IndexerService | None = None
_indexer_lock = threading.Lock()


def get_indexer() -> IndexerService:
    """Get the singleton IndexerService instance."""
    global _indexer
    if _indexer is None:
        with _indexer_lock:
            if _indexer is None:
                _indexer = IndexerService()
    return _indexer

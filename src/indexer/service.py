"""Indexer service - central coordinator for codebase indexing operations.

This module provides the IndexerService which handles:
- Full indexing of new codebases
- Incremental updates for changed files
- Repository name resolution (with collision handling)
- Path validation and security checks
- Coordination of chunk, symbol, and centrality indexing
"""

import hashlib
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cocoindex
from pathspec import GitIgnoreSpec
from psycopg import sql

from config.settings import settings
from src.exceptions import IndexingError, PathError
from src.indexer.flow import create_indexing_flow
from src.indexer.repo_manager import RepoManager
from src.retrieval.centrality import compute_and_store_centrality, delete_centrality_table
from src.retrieval.vector_search import get_chunks_table_name

logger = logging.getLogger(__name__)


@dataclass
class IndexResult:
    """Result of an indexing operation."""
    repo_name: str
    status: str
    file_count: int = 0
    chunk_count: int = 0
    message: str | None = None


class IndexerService:
    """Service for indexing codebases."""

    def __init__(self):
        self._repo_manager = RepoManager()
        self._initialized = False
        self._change_check_cache: dict[str, tuple[float, float, bool]] = {}

    def _init_cocoindex(self) -> None:
        """Initialize CocoIndex with database and API configuration."""
        if self._initialized:
            return
        os.environ["COCOINDEX_DATABASE_URL"] = settings.database_url
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        cocoindex.init()
        self._initialized = True

    @staticmethod
    def path_to_repo_name(path: str) -> str:
        """Convert a path to a consistent repository name."""
        name = Path(path).resolve().name.lower()
        name = re.sub(r'[-. ]+', '_', name)
        name = re.sub(r'[^a-z0-9_]', '', name).strip('_')
        return name or 'repo'

    def resolve_repo_name(self, path: str | Path) -> str:
        """Resolve a stable repo name for a given path."""
        resolved = Path(path).resolve()
        resolved_str = str(resolved)
        base = self.path_to_repo_name(resolved_str)
        digest = hashlib.sha1(resolved_str.encode("utf-8")).hexdigest()

        # Check existing registrations
        for name in [base] + [f"{base}_{digest[:n]}" for n in (8, 12, 16, 40)]:
            existing = self._get_repo_safe(name)
            if existing and existing.path == resolved_str:
                return name

        # Find available name
        if self._get_repo_safe(base) is None:
            return base
        for n in (8, 12, 16, 40):
            hashed = f"{base}_{digest[:n]}"
            if self._get_repo_safe(hashed) is None:
                return hashed
        return f"{base}_{digest}"

    def _get_repo_safe(self, name: str):
        """Safely get repo, returning None on error."""
        try:
            return self._repo_manager.get_repo(name)
        except Exception:
            return None

    def _validate_path(self, path: str) -> Path:
        """Validate and sanitize a path for indexing."""
        try:
            resolved = Path(path).resolve(strict=False)

            if not resolved.exists():
                raise PathError(f"Path does not exist: {resolved}")
            if not resolved.is_dir():
                raise PathError(f"Path is not a directory: {resolved}")

            if Path(path).is_symlink():
                logger.warning(f"Path is a symlink: {path} -> {resolved}")

            # Verify read permissions
            try:
                next(resolved.iterdir(), None)
            except PermissionError as e:
                raise PathError(f"Insufficient permissions to read directory: {resolved}") from e

            logger.debug(f"Validated path: {path} -> {resolved}")
            return resolved

        except (OSError, RuntimeError) as e:
            raise PathError(f"Invalid or unsafe path: {e}") from e

    def _get_stats(self, repo_name: str) -> tuple[int, int]:
        return self._repo_manager.get_file_count(repo_name), self._repo_manager.get_chunk_count(repo_name)

    def _compute_centrality(self, repo_name: str) -> None:
        try:
            compute_and_store_centrality(repo_name)
        except Exception as e:
            logger.warning(f"Centrality computation failed for {repo_name}: {e}")

    def _build_ignore_spec(self, repo_path: Path) -> GitIgnoreSpec:
        """Build gitignore spec from .gitignore and default patterns."""
        patterns = list(settings.excluded_patterns)
        gitignore_path = repo_path / ".gitignore"
        if gitignore_path.exists():
            try:
                patterns.extend(gitignore_path.read_text().splitlines())
            except (OSError, UnicodeDecodeError) as e:
                logger.debug(f"Could not read .gitignore: {e}")
        return GitIgnoreSpec.from_lines(patterns)

    def _iter_relevant_files(self, repo_path: str):
        """Yield (full_path, relative_path_str) for index-relevant files."""
        repo_path_obj = Path(repo_path)
        included_ext = {ext.lower() for ext in settings.included_extensions}
        ignore_spec = self._build_ignore_spec(repo_path_obj)

        for root, dirnames, filenames in os.walk(repo_path, topdown=True):
            rel_root = Path(root).relative_to(repo_path_obj)
            
            # Filter directories in-place
            dirnames[:] = [d for d in dirnames if not ignore_spec.match_file(str(rel_root / d) + "/")]

            for filename in filenames:
                if os.path.splitext(filename)[1].lower() not in included_ext:
                    continue

                full_path = Path(root) / filename
                try:
                    relative = full_path.relative_to(repo_path_obj)
                except ValueError:
                    continue

                if not ignore_spec.match_file(str(relative)):
                    yield full_path, str(relative)

    def _has_files_changed(self, repo_name: str, repo_path: str) -> bool:
        """Check if any indexed files have changed since last indexing."""
        repo = self._repo_manager.get_repo(repo_name)
        if not (repo and repo.last_indexed):
            return False

        last_ts = self._datetime_to_timestamp(repo.last_indexed)
        now = time.monotonic()

        # Check cache (5 second TTL)
        cached = self._change_check_cache.get(repo_name)
        if cached and cached[0] == last_ts and (now - cached[1]) < 5.0:
            return cached[2]

        changed = any(
            self._file_modified_since(full_path, last_ts)
            for full_path, _ in self._iter_relevant_files(repo_path)
        )

        self._change_check_cache[repo_name] = (last_ts, now, changed)
        return changed

    @staticmethod
    def _datetime_to_timestamp(value: datetime) -> float:
        """Convert datetime to POSIX timestamp, treating naive as UTC."""
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc).timestamp()
        return value.timestamp()

    def _get_modified_files(self, repo_path: str, last_indexed_ts: float) -> list[str]:
        """Return relative paths of files modified since last_indexed_ts."""
        return [
            rel for full, rel in self._iter_relevant_files(repo_path)
            if self._file_modified_since(full, last_indexed_ts)
        ]

    @staticmethod
    def _file_modified_since(path: Path, ts: float) -> bool:
        try:
            return path.stat().st_mtime > ts
        except OSError:
            return True

    def _verify_index_data_exists(self, repo_name: str) -> bool:
        """Verify that indexed data actually exists in the database."""
        from src.storage.postgres import get_connection

        chunks_table = get_chunks_table_name(repo_name)
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = %s)",
                        (chunks_table,)
                    )
                    if not cur.fetchone()[0]:
                        return False

                    cur.execute(sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(chunks_table)))
                    return cur.fetchone()[0] > 0
        except Exception as e:
            logger.error(f"Error verifying index data for {repo_name}: {e}")
            return False

    def ensure_indexed(self, path: str) -> IndexResult:
        """Ensure a codebase is indexed, creating or updating as needed."""
        resolved_path = self._validate_path(path)
        repo_name = self.resolve_repo_name(resolved_path)
        repo_path = str(resolved_path)

        repo = self._repo_manager.get_repo(repo_name)

        if repo and repo.status == "ready" and repo.path == repo_path:
            if self._has_files_changed(repo_name, repo_path):
                return self._incremental_update(repo_name, repo_path)
            return IndexResult(
                repo_name=repo_name, status="unchanged",
                file_count=repo.file_count, chunk_count=repo.chunk_count
            )

        if repo and repo.path != repo_path:
            logger.info(f"Path changed for {repo_name}, re-indexing")

        return self._full_index(repo_name, repo_path)

    def _delete_index(self, repo_name: str) -> None:
        """Delete an existing index by dropping database tables."""
        from src.storage.postgres import get_connection
        from src.retrieval.vector_search import sanitize_identifier
        from src.indexer.flow import clear_flow_cache
        from src.storage.schema import sanitize_repo_name

        safe_name = sanitize_identifier(repo_name)
        chunks_table = get_chunks_table_name(repo_name)
        tracking_table = f"codeindex_{safe_name}__cocoindex_tracking"

        clear_flow_cache(repo_name)

        try:
            with get_connection() as conn:
                with conn.transaction():
                    with conn.cursor() as cur:
                        for table in [chunks_table, tracking_table]:
                            cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(table)))
                        cur.execute("DELETE FROM repos WHERE name = %s", (repo_name,))
                        cur.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                            sql.Identifier(sanitize_repo_name(repo_name))
                        ))
            logger.info(f"Deleted index tables for {repo_name}")
        except Exception as e:
            logger.warning(f"Could not delete index for {repo_name}: {e}")

        delete_centrality_table(repo_name)

    def _incremental_update(self, repo_name: str, repo_path: str) -> IndexResult:
        """Perform incremental update on an existing index."""
        try:
            self._init_cocoindex()

            repo = self._repo_manager.get_repo(repo_name)
            last_ts = self._datetime_to_timestamp(repo.last_indexed) if repo and repo.last_indexed else 0.0
            modified_files = self._get_modified_files(repo_path, last_ts)

            flow = create_indexing_flow(repo_name, repo_path)
            flow.setup()
            flow.update()

            self._update_symbols_for_files(repo_name, repo_path, modified_files)

            file_count, chunk_count = self._get_stats(repo_name)
            self._verify_index_data_exists(repo_name)
            self._repo_manager.update_status(repo_name, "ready", file_count=file_count, chunk_count=chunk_count)

            from src.retrieval.graph_cache import clear_graph_cache
            clear_graph_cache(repo_name)
            self._compute_centrality(repo_name)

            return IndexResult(repo_name=repo_name, status="updated", file_count=file_count, chunk_count=chunk_count)

        except Exception as e:
            logger.warning(f"Incremental update failed for {repo_name}: {e}")
            file_count, chunk_count = self._get_stats(repo_name)
            return IndexResult(
                repo_name=repo_name, status="unchanged",
                file_count=file_count, chunk_count=chunk_count,
                message=f"Using existing index (update failed: {e})"
            )

    def _update_symbols_for_files(self, repo_name: str, repo_path: str, modified_files: list[str]) -> None:
        """Update symbol index for modified files."""
        if not settings.enable_symbol_indexing or not modified_files:
            return

        try:
            from src.embeddings.provider import get_provider
            from src.indexer.symbol_indexing import create_symbols_table, delete_file_symbols, index_file_symbols

            create_symbols_table(repo_name, settings.embedding_dimensions)
            provider = get_provider()
            repo_root = Path(repo_path)

            for rel_path in sorted(set(modified_files)):
                file_path = repo_root / rel_path
                delete_file_symbols(repo_name, rel_path)
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding="utf-8")
                        index_file_symbols(repo_name, rel_path, content, embedding_provider=provider)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Symbol indexing failed for {repo_name}: {e}")

    def _full_index(self, repo_name: str, repo_path: str) -> IndexResult:
        """Perform full indexing of a codebase."""
        self._repo_manager.register_repo(repo_name, repo_path)
        self._repo_manager.update_status(repo_name, "indexing")

        try:
            self._init_cocoindex()
            logger.info(f"Indexing {repo_name} from {repo_path}")

            flow = create_indexing_flow(repo_name, repo_path)
            try:
                flow.drop()
            except Exception:
                pass

            flow.setup()
            flow.update()

            try:
                from src.indexer.symbol_indexing import index_repository_symbols
                index_repository_symbols(repo_name, repo_path)
            except Exception as e:
                logger.warning(f"Symbol indexing failed for {repo_name}: {e}")

            file_count, chunk_count = self._get_stats(repo_name)
            self._verify_index_data_exists(repo_name)
            self._repo_manager.update_status(repo_name, "ready", file_count=file_count, chunk_count=chunk_count)
            self._compute_centrality(repo_name)

            logger.info(f"Indexed {repo_name}: {file_count} files, {chunk_count} chunks")
            return IndexResult(repo_name=repo_name, status="created", file_count=file_count, chunk_count=chunk_count)

        except Exception as e:
            logger.error(f"Indexing failed for {repo_name}: {e}")
            self._repo_manager.update_status(repo_name, "error", error_message=str(e))
            raise IndexingError(f"Failed to index {repo_path}: {e}") from e


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

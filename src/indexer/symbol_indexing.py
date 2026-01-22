"""Symbol indexing module for extracting and storing symbols.

This module handles extraction and storage of function/class/method symbols
from code files, with embeddings for symbol-level search.
"""

import logging
import threading
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

from pathspec import GitIgnoreSpec
from psycopg import sql

from config.settings import settings
from src.rust_bridge import compute_file_hash as _compute_file_hash
from src.embeddings.provider import get_provider
from src.parser.ast_parser import get_language_from_file
from src.parser.symbol_extractor import Symbol, extract_symbols
from src.storage.postgres import get_connection
from src.storage.schema import get_create_symbols_table_sql, sanitize_repo_name

logger = logging.getLogger(__name__)

# In-memory cache of file hashes for incremental indexing
_file_hash_cache: dict[str, dict[str, str]] = {}  # repo_name -> {filename -> hash}
_file_hash_cache_lock = threading.Lock()


def _get_cached_hash(repo_name: str, filename: str) -> str | None:
    """Get cached file hash if exists."""
    with _file_hash_cache_lock:
        return _file_hash_cache.get(repo_name, {}).get(filename)


def _set_cached_hash(repo_name: str, filename: str, hash_val: str) -> None:
    """Cache file hash."""
    with _file_hash_cache_lock:
        if repo_name not in _file_hash_cache:
            _file_hash_cache[repo_name] = {}
        _file_hash_cache[repo_name][filename] = hash_val


def _clear_cached_hash(repo_name: str, filename: str) -> None:
    """Clear cached hash for a file."""
    with _file_hash_cache_lock:
        if repo_name in _file_hash_cache:
            _file_hash_cache[repo_name].pop(filename, None)


def file_needs_reindex(repo_name: str, filename: str, content: str) -> bool:
    """Check if file needs re-indexing based on content hash.
    
    Returns True if file is new or content has changed.
    """
    current_hash = _compute_file_hash(content)
    cached_hash = _get_cached_hash(repo_name, filename)
    
    if cached_hash is None:
        # Not in cache, check database
        cached_hash = _get_db_file_hash(repo_name, filename)
        if cached_hash:
            _set_cached_hash(repo_name, filename, cached_hash)
    
    return cached_hash != current_hash


def _get_db_file_hash(repo_name: str, filename: str) -> str | None:
    """Get file content hash from database (stored in first symbol's docstring prefix)."""
    schema_name = sanitize_repo_name(repo_name)
    symbols_table = sql.Identifier(schema_name, "symbols")
    
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Content hash stored in first symbol's docstring field prefix
                cur.execute(
                    sql.SQL("""
                        SELECT substring(docstring from '^#hash:([a-f0-9]+)#')
                        FROM {table}
                        WHERE filename = %s AND docstring LIKE '#hash:%'
                        LIMIT 1
                    """).format(table=symbols_table),
                    (filename,)
                )
                row = cur.fetchone()
                return row[0] if row and row[0] else None
    except Exception as e:
        logger.debug(f"Could not get file hash for {filename}: {e}")
        return None


def create_symbols_table(repo_name: str, dimensions: int = 3072) -> None:
    """Create symbols table for a repository if it doesn't exist.

    Args:
        repo_name: Repository name
        dimensions: Vector embedding dimensions
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            create_sql = get_create_symbols_table_sql(repo_name, dimensions)
            cur.execute(create_sql)
        conn.commit()


def create_edges_table(repo_name: str) -> None:
    """Create edges table for a repository if it doesn't exist.

    Args:
        repo_name: Repository name
    """
    from src.storage.schema import get_create_edges_table_sql

    with get_connection() as conn:
        with conn.cursor() as cur:
            create_sql = get_create_edges_table_sql(repo_name)
            cur.execute(create_sql)
        conn.commit()


def generate_symbol_text(symbol: Symbol, filename: str) -> str:
    """Generate text representation of symbol for embedding.

    Combines signature, docstring, and context for better search quality.

    Args:
        symbol: Symbol object
        filename: File path

    Returns:
        Text representation for embedding
    """
    parts = [
        f"# File: {filename}",
        f"# Symbol: {symbol.symbol_name} ({symbol.symbol_type})",
    ]

    if symbol.parent_symbol:
        parts.append(f"# Parent: {symbol.parent_symbol}")

    parts.append(f"\n{symbol.signature}")

    # Strip hash prefix from docstring if present
    docstring = symbol.docstring
    if docstring and docstring.startswith("#hash:"):
        hash_end = docstring.find("#", 6)
        if hash_end > 0:
            docstring = docstring[hash_end + 1:] or None

    if docstring:
        parts.append(f"\n\"\"\"{docstring}\"\"\"")

    return "\n".join(parts)


def index_file_symbols(
    repo_name: str,
    filename: str,
    content: str,
    embedding_provider: Any | None = None,
    content_hash: str | None = None,
    batch_id: str | None = None,
) -> tuple[int, bool]:
    """Extract and index symbols from a single file.

    Args:
        repo_name: Repository name
        filename: File path
        content: File content
        embedding_provider: Optional embedding provider (will create if None)
        content_hash: Optional content hash to store for incremental indexing
        batch_id: Optional batch identifier for this indexing run

    Returns:
        Tuple of (count, success) - count is symbols indexed, success indicates no failure
    """
    # Detect language
    language = get_language_from_file(filename)
    if not language:
        logger.debug(f"Skipping {filename}: unknown language")
        return 0, True  # Not a failure, just unsupported

    # Extract symbols
    symbols = extract_symbols(content, language, filename)
    if not symbols:
        logger.debug(f"No symbols found in {filename}")
        return 0, True  # Success with zero symbols

    # Get embedding provider
    if embedding_provider is None:
        embedding_provider = get_provider()

    # Store content hash in first symbol's docstring (create copy to avoid mutation)
    if content_hash and symbols:
        first_sym = symbols[0]
        hash_prefix = f"#hash:{content_hash}#"
        new_docstring = hash_prefix + (first_sym.docstring or "")
        symbols = [replace(first_sym, docstring=new_docstring)] + list(symbols[1:])

    # Prepare batch embeddings
    symbol_texts = [generate_symbol_text(sym, filename) for sym in symbols]

    try:
        embeddings = embedding_provider.get_embeddings_batch(symbol_texts)
        schema_name = sanitize_repo_name(repo_name)
        symbols_table = sql.Identifier(schema_name, "symbols")

        with get_connection() as conn:
            with conn.cursor() as cur:
                insert_sql = sql.SQL("""
                    INSERT INTO {table} (
                        filename, symbol_name, symbol_type, line_start, line_end,
                        signature, docstring, parent_symbol, visibility, category,
                        embedding, content_tsv, batch_id
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        to_tsvector('english', %s), %s
                    )
                    ON CONFLICT (filename, symbol_name, line_start) DO UPDATE SET
                        symbol_type = EXCLUDED.symbol_type,
                        line_end = EXCLUDED.line_end,
                        signature = EXCLUDED.signature,
                        docstring = EXCLUDED.docstring,
                        parent_symbol = EXCLUDED.parent_symbol,
                        visibility = EXCLUDED.visibility,
                        category = EXCLUDED.category,
                        embedding = EXCLUDED.embedding,
                        content_tsv = EXCLUDED.content_tsv,
                        batch_id = EXCLUDED.batch_id,
                        updated_at = CURRENT_TIMESTAMP
                """).format(table=symbols_table)

                for symbol, embedding in zip(symbols, embeddings):
                    # Strip hash prefix from docstring for search text
                    docstring_for_search = symbol.docstring
                    if docstring_for_search and docstring_for_search.startswith("#hash:"):
                        hash_end = docstring_for_search.find("#", 6)
                        if hash_end > 0:
                            docstring_for_search = docstring_for_search[hash_end + 1:] or None
                    
                    search_text = f"{symbol.signature} {docstring_for_search}" if docstring_for_search else symbol.signature
                    cur.execute(insert_sql, (
                        filename, symbol.symbol_name, symbol.symbol_type,
                        symbol.line_start, symbol.line_end, symbol.signature,
                        symbol.docstring, symbol.parent_symbol, symbol.visibility,
                        symbol.category, embedding, search_text, batch_id,
                    ))
            conn.commit()

        logger.info(f"Indexed {len(symbols)} symbols from {filename}")
        return len(symbols), True

    except Exception as e:
        logger.error(f"Error indexing symbols from {filename}: {e}")
        return 0, False


def _delete_old_symbols(repo_name: str, filename: str, batch_id: str) -> int:
    """Delete old symbols for a file after new batch is indexed.

    Uses batch_id to identify old symbols - only deletes symbols with different batch_id.

    Args:
        repo_name: Repository name
        filename: File path
        batch_id: Current batch identifier

    Returns:
        Number of symbols deleted
    """
    schema_name = sanitize_repo_name(repo_name)
    symbols_table = sql.Identifier(schema_name, "symbols")

    with get_connection() as conn:
        with conn.cursor() as cur:
            delete_sql = sql.SQL("""
                DELETE FROM {table} 
                WHERE filename = %s 
                AND batch_id IS DISTINCT FROM %s
            """).format(table=symbols_table)
            cur.execute(delete_sql, (filename, batch_id))
            deleted_count = cur.rowcount
        conn.commit()

    logger.debug(f"Deleted {deleted_count} old symbols from {filename}")
    return deleted_count


def delete_file_symbols(repo_name: str, filename: str) -> int:
    """Delete all symbols for a file.

    Args:
        repo_name: Repository name
        filename: File path

    Returns:
        Number of symbols deleted
    """
    schema_name = sanitize_repo_name(repo_name)
    symbols_table = sql.Identifier(schema_name, "symbols")

    with get_connection() as conn:
        with conn.cursor() as cur:
            delete_sql = sql.SQL("DELETE FROM {table} WHERE filename = %s").format(table=symbols_table)
            cur.execute(delete_sql, (filename,))
            deleted_count = cur.rowcount
        conn.commit()

    _clear_cached_hash(repo_name, filename)
    logger.debug(f"Deleted {deleted_count} symbols from {filename}")
    return deleted_count


def index_file_symbols_incremental(
    repo_name: str,
    filename: str,
    content: str,
    embedding_provider: Any | None = None,
    force: bool = False,
) -> tuple[int, bool]:
    """Incrementally index symbols - skip if file unchanged.
    
    Args:
        repo_name: Repository name
        filename: File path
        content: File content
        embedding_provider: Optional embedding provider
        force: Force re-index even if unchanged
    
    Returns:
        Tuple of (symbols_indexed, was_updated)
    """
    content_hash = _compute_file_hash(content)
    
    if not force and not file_needs_reindex(repo_name, filename, content):
        logger.debug(f"Skipping {filename}: unchanged")
        return 0, False
    
    # Generate batch_id for this indexing run
    batch_id = str(uuid.uuid4())
    
    # Index new symbols first (before deleting old ones)
    count, success = index_file_symbols(repo_name, filename, content, embedding_provider, content_hash, batch_id)
    
    if success:
        # Success - delete old symbols and update cache (even if count == 0)
        _delete_old_symbols(repo_name, filename, batch_id)
        _set_cached_hash(repo_name, filename, content_hash)
        return count, True
    
    # Indexing failed - don't delete existing symbols or update cache
    return 0, False


def _build_ignore_spec(repo_root: Path, excluded_patterns: list[str]) -> GitIgnoreSpec:
    """Build ignore spec from .gitignore and configured patterns."""

    patterns = list(excluded_patterns)

    # Some unit tests mock Path(); avoid interacting with the filesystem in that case.
    try:
        gitignore_path = repo_root / ".gitignore"
    except TypeError:
        return GitIgnoreSpec.from_lines(patterns)

    try:
        if gitignore_path.exists():
            patterns.extend(gitignore_path.read_text().splitlines())
    except (OSError, UnicodeDecodeError) as e:
        logger.debug(f"Could not read .gitignore: {e}")
    except Exception:
        # Defensive: ignore any other filesystem or mock-related errors.
        return GitIgnoreSpec.from_lines(patterns)

    return GitIgnoreSpec.from_lines(patterns)


def _is_file_excluded(relative_path: str, ignore_spec: GitIgnoreSpec) -> bool:
    """Check if a file path matches any exclusion pattern."""

    normalized = relative_path.replace("\\", "/")
    return bool(ignore_spec.match_file(normalized))


def index_repository_symbols(
    repo_name: str,
    repo_path: str,
    included_patterns: list[str] | None = None,
    excluded_patterns: list[str] | None = None,
) -> dict:
    """Index symbols for all files in a repository.

    Args:
        repo_name: Repository name
        repo_path: Path to repository root
        included_patterns: Glob patterns for files to include
        excluded_patterns: Glob patterns for files to exclude

    Returns:
        Statistics dict with files_processed, symbols_indexed, errors
    """
    if not settings.enable_symbol_indexing:
        logger.info("Symbol indexing is disabled")
        return {"files_processed": 0, "symbols_indexed": 0, "errors": 0}

    # Create symbols table
    create_symbols_table(repo_name, settings.embedding_dimensions)

    # Create edges table for call graph
    create_edges_table(repo_name)

    # Create graph cache table
    from src.retrieval.graph_cache import create_graph_cache_table
    create_graph_cache_table(repo_name)

    if included_patterns is None:
        included_patterns = [f"**/*{ext}" for ext in settings.included_extensions]
    if excluded_patterns is None:
        excluded_patterns = settings.excluded_patterns

    repo_path_obj = Path(repo_path)
    ignore_spec = _build_ignore_spec(repo_path_obj, excluded_patterns)

    files_to_process: list[Path] = []
    for pattern in included_patterns:
        for file_path in repo_path_obj.glob(pattern):
            if not file_path.is_file():
                continue
            rel = str(file_path.relative_to(repo_path_obj)).replace("\\", "/")
            if _is_file_excluded(rel, ignore_spec):
                continue
            files_to_process.append(file_path)

    if not files_to_process:
        logger.warning(f"No files found to process in {repo_path}")
        return {"files_processed": 0, "symbols_indexed": 0, "errors": 0}

    embedding_provider = get_provider()
    stats = {"files_processed": 0, "symbols_indexed": 0, "errors": 0}

    logger.info(f"Indexing symbols for {len(files_to_process)} files in {repo_name}")

    for file_path in files_to_process:
        try:
            content = file_path.read_text(encoding="utf-8")
            relative_path = str(file_path.relative_to(repo_path_obj)).replace("\\", "/")
            count, success = index_file_symbols(repo_name, relative_path, content, embedding_provider)
            stats["files_processed"] += 1
            stats["symbols_indexed"] += count
            if not success:
                stats["errors"] += 1
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            stats["errors"] += 1

    logger.info(f"Symbol indexing complete: {stats['files_processed']} files, "
                f"{stats['symbols_indexed']} symbols, {stats['errors']} errors")

    return stats

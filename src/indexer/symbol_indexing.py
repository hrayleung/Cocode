"""Symbol indexing module for extracting and storing symbols.

This module handles extraction and storage of function/class/method symbols
from code files, with embeddings for symbol-level search.
"""

import logging
from pathlib import Path
from typing import Any

from psycopg import sql

from config.settings import settings
from src.embeddings.provider import get_provider
from src.parser.ast_parser import get_language_from_file
from src.parser.symbol_extractor import Symbol, extract_symbols
from src.storage.postgres import get_connection
from src.storage.schema import get_create_symbols_table_sql, sanitize_repo_name

logger = logging.getLogger(__name__)


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

    if symbol.docstring:
        parts.append(f"\n\"\"\"{symbol.docstring}\"\"\"")

    return "\n".join(parts)


def index_file_symbols(
    repo_name: str,
    filename: str,
    content: str,
    embedding_provider: Any | None = None,
) -> int:
    """Extract and index symbols from a single file.

    Args:
        repo_name: Repository name
        filename: File path
        content: File content
        embedding_provider: Optional embedding provider (will create if None)

    Returns:
        Number of symbols indexed
    """
    # Detect language
    language = get_language_from_file(filename)
    if not language:
        logger.debug(f"Skipping {filename}: unknown language")
        return 0

    # Extract symbols
    symbols = extract_symbols(content, language, filename)
    if not symbols:
        logger.debug(f"No symbols found in {filename}")
        return 0

    # Get embedding provider
    if embedding_provider is None:
        embedding_provider = get_provider()

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
                        embedding, content_tsv
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        to_tsvector('english', %s)
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
                        updated_at = CURRENT_TIMESTAMP
                """).format(table=symbols_table)

                for symbol, embedding in zip(symbols, embeddings):
                    search_text = f"{symbol.signature} {symbol.docstring}" if symbol.docstring else symbol.signature
                    cur.execute(insert_sql, (
                        filename, symbol.symbol_name, symbol.symbol_type,
                        symbol.line_start, symbol.line_end, symbol.signature,
                        symbol.docstring, symbol.parent_symbol, symbol.visibility,
                        symbol.category, embedding, search_text,
                    ))
            conn.commit()

        logger.info(f"Indexed {len(symbols)} symbols from {filename}")
        return len(symbols)

    except Exception as e:
        logger.error(f"Error indexing symbols from {filename}: {e}")
        return 0


def delete_file_symbols(repo_name: str, filename: str) -> int:
    """Delete symbols for a file (for incremental updates).

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

    logger.debug(f"Deleted {deleted_count} symbols from {filename}")
    return deleted_count


def _is_file_excluded(relative_path: Path, excluded_patterns: list[str]) -> bool:
    """Check if a file path matches any exclusion pattern."""
    return any(relative_path.match(pattern) for pattern in excluded_patterns)


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
    files_to_process = [
        file_path
        for pattern in included_patterns
        for file_path in repo_path_obj.glob(pattern)
        if file_path.is_file() and not _is_file_excluded(file_path.relative_to(repo_path_obj), excluded_patterns)
    ]

    if not files_to_process:
        logger.warning(f"No files found to process in {repo_path}")
        return {"files_processed": 0, "symbols_indexed": 0, "errors": 0}

    embedding_provider = get_provider()
    stats = {"files_processed": 0, "symbols_indexed": 0, "errors": 0}

    logger.info(f"Indexing symbols for {len(files_to_process)} files in {repo_name}")

    for file_path in files_to_process:
        try:
            content = file_path.read_text(encoding="utf-8")
            relative_path = str(file_path.relative_to(repo_path_obj))
            count = index_file_symbols(repo_name, relative_path, content, embedding_provider)
            stats["files_processed"] += 1
            stats["symbols_indexed"] += count
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            stats["errors"] += 1

    logger.info(f"Symbol indexing complete: {stats['files_processed']} files, "
                f"{stats['symbols_indexed']} symbols, {stats['errors']} errors")

    return stats

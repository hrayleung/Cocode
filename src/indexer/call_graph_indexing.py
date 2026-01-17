"""Call graph indexing - extract and store function call relationships."""

import logging
from typing import Optional

from src.parser import extract_calls_from_function, Symbol
from src.parser.ast_parser import get_language_from_file
from src.retrieval.call_graph import (
    CallEdge,
    resolve_call_to_symbol,
    store_call_edge,
    delete_symbol_edges,
)
from src.storage.postgres import get_connection
from src.storage.schema import sanitize_repo_name
from psycopg import sql

logger = logging.getLogger(__name__)


def index_symbol_calls(
    repo_name: str,
    symbol: Symbol,
    filename: str,
    file_content: str,
) -> int:
    """Extract and index function calls from a symbol.

    Args:
        repo_name: Repository name
        symbol: Symbol object (function/method/class)
        filename: Filename containing the symbol
        file_content: Full file content

    Returns:
        Number of call edges indexed
    """
    # Only index calls from functions and methods
    if symbol.symbol_type not in ('function', 'method'):
        return 0

    # Detect language
    language = get_language_from_file(filename)
    if not language:
        logger.debug(f"Unknown language for {filename}, skipping call extraction")
        return 0

    # Extract function calls from the symbol's body
    try:
        calls = extract_calls_from_function(
            code=file_content,
            language=language,
            function_name=symbol.symbol_name,
            line_start=symbol.line_start,
            line_end=symbol.line_end,
        )
    except Exception as e:
        logger.warning(f"Failed to extract calls from {symbol.symbol_name} in {filename}: {e}")
        return 0

    if not calls:
        return 0

    # Get symbol ID from database
    symbol_id = get_symbol_id(repo_name, filename, symbol.symbol_name, symbol.line_start)
    if not symbol_id:
        logger.warning(f"Symbol {symbol.symbol_name} not found in database")
        return 0

    # Resolve and store each call
    edges_stored = 0
    for call in calls:
        # Resolve call to target symbol
        target_id, target_file, confidence = resolve_call_to_symbol(
            repo_name=repo_name,
            source_file=filename,
            function_name=call.function_name,
            object_name=call.object_name,
        )

        # Determine target line (approximate for unresolved)
        target_line = None
        if target_id:
            target_line = get_symbol_line(repo_name, target_id)

        # Create edge
        edge = CallEdge(
            source_symbol_id=symbol_id,
            target_symbol_id=target_id,
            edge_type='calls',
            source_file=filename,
            source_line=call.line_number,
            target_file=target_file,
            target_symbol_name=call.function_name,
            target_line=target_line,
            confidence=confidence,
            context=_format_call_context(call),
        )

        # Store edge
        if store_call_edge(repo_name, edge):
            edges_stored += 1

    logger.debug(f"Indexed {edges_stored} call edges from {symbol.symbol_name}")
    return edges_stored


def _format_call_context(call) -> Optional[str]:
    """Format call context for storage.

    Args:
        call: FunctionCall object

    Returns:
        Formatted context string
    """
    contexts = []

    if call.is_recursive:
        contexts.append("recursive")

    if call.context:
        contexts.append(call.context)

    if call.call_type == 'method_call' and call.object_name:
        if call.object_name == '[chained]':
            contexts.append("chained_call")
        else:
            contexts.append(f"object={call.object_name}")

    return ", ".join(contexts) if contexts else None


def get_symbol_id(repo_name: str, filename: str, symbol_name: str, line_start: int) -> Optional[str]:
    """Get symbol ID from database.

    Args:
        repo_name: Repository name
        filename: Filename
        symbol_name: Symbol name
        line_start: Starting line

    Returns:
        Symbol UUID or None if not found
    """
    schema_name = sanitize_repo_name(repo_name)
    symbols_table = sql.Identifier(schema_name, "symbols")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("""
                    SELECT id FROM {table}
                    WHERE filename = %s AND symbol_name = %s AND line_start = %s
                    LIMIT 1
                """).format(table=symbols_table),
                (filename, symbol_name, line_start)
            )
            row = cur.fetchone()
            return row[0] if row else None


def get_symbol_line(repo_name: str, symbol_id: str) -> Optional[int]:
    """Get symbol line_start from database.

    Args:
        repo_name: Repository name
        symbol_id: Symbol UUID

    Returns:
        Line number or None if not found
    """
    schema_name = sanitize_repo_name(repo_name)
    symbols_table = sql.Identifier(schema_name, "symbols")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("""
                    SELECT line_start FROM {table}
                    WHERE id = %s
                    LIMIT 1
                """).format(table=symbols_table),
                (symbol_id,)
            )
            row = cur.fetchone()
            return row[0] if row else None


def index_file_calls(repo_name: str, filename: str, file_content: str, symbols: list[Symbol]) -> int:
    """Index all function calls from a file's symbols.

    Args:
        repo_name: Repository name
        filename: Filename
        file_content: Full file content
        symbols: List of Symbol objects extracted from the file

    Returns:
        Total number of call edges indexed
    """
    total_edges = 0

    for symbol in symbols:
        edges = index_symbol_calls(repo_name, symbol, filename, file_content)
        total_edges += edges

    return total_edges


def delete_file_call_edges(repo_name: str, filename: str) -> int:
    """Delete all call edges originating from symbols in a file.

    This is used during incremental indexing when a file changes.

    Args:
        repo_name: Repository name
        filename: Filename

    Returns:
        Number of edges deleted
    """
    schema_name = sanitize_repo_name(repo_name)
    edges_table = sql.Identifier(schema_name, "edges")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("""
                    DELETE FROM {table}
                    WHERE source_file = %s
                """).format(table=edges_table),
                (filename,)
            )
            deleted = cur.rowcount
            conn.commit()
            logger.debug(f"Deleted {deleted} call edges from {filename}")
            return deleted

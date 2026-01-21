"""Call graph indexing - extract and store function call relationships."""

import logging
from typing import Optional

from src.parser import extract_calls_from_function, Symbol
from src.parser.ast_parser import get_language_from_file
from src.retrieval.call_graph import CallEdge, resolve_call_to_symbol, store_call_edge
from src.storage.postgres import get_connection
from src.storage.schema import sanitize_repo_name
from psycopg import sql

logger = logging.getLogger(__name__)


def index_symbol_calls(repo_name: str, symbol: Symbol, filename: str, file_content: str) -> int:
    """Extract and index function calls from a symbol. Returns number of edges indexed."""
    if symbol.symbol_type not in ('function', 'method'):
        return 0

    language = get_language_from_file(filename)
    if not language:
        logger.debug(f"Unknown language for {filename}, skipping call extraction")
        return 0

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

    symbol_id = get_symbol_id(repo_name, filename, symbol.symbol_name, symbol.line_start)
    if not symbol_id:
        logger.warning(f"Symbol {symbol.symbol_name} not found in database")
        return 0

    edges_stored = 0
    for call in calls:
        target_id, target_file, confidence = resolve_call_to_symbol(
            repo_name=repo_name,
            source_file=filename,
            function_name=call.function_name,
            object_name=call.object_name,
        )

        target_line = get_symbol_line(repo_name, target_id) if target_id else None

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

        if store_call_edge(repo_name, edge):
            edges_stored += 1

    logger.debug(f"Indexed {edges_stored} call edges from {symbol.symbol_name}")
    return edges_stored


def _format_call_context(call) -> Optional[str]:
    """Format call context for storage."""
    contexts = []
    if call.is_recursive:
        contexts.append("recursive")
    if call.context:
        contexts.append(call.context)
    if call.call_type == 'method_call' and call.object_name:
        contexts.append("chained_call" if call.object_name == '[chained]' else f"object={call.object_name}")
    return ", ".join(contexts) if contexts else None


def get_symbol_id(repo_name: str, filename: str, symbol_name: str, line_start: int) -> Optional[str]:
    """Get symbol UUID from database, or None if not found."""
    schema_name = sanitize_repo_name(repo_name)
    symbols_table = sql.Identifier(schema_name, "symbols")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT id FROM {table} WHERE filename = %s AND symbol_name = %s AND line_start = %s LIMIT 1").format(table=symbols_table),
                (filename, symbol_name, line_start)
            )
            row = cur.fetchone()
            return row[0] if row else None


def get_symbol_line(repo_name: str, symbol_id: str) -> Optional[int]:
    """Get symbol line_start from database, or None if not found."""
    schema_name = sanitize_repo_name(repo_name)
    symbols_table = sql.Identifier(schema_name, "symbols")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT line_start FROM {table} WHERE id = %s LIMIT 1").format(table=symbols_table),
                (symbol_id,)
            )
            row = cur.fetchone()
            return row[0] if row else None


def index_file_calls(repo_name: str, filename: str, file_content: str, symbols: list[Symbol]) -> int:
    """Index all function calls from a file's symbols. Returns total edges indexed."""
    return sum(index_symbol_calls(repo_name, symbol, filename, file_content) for symbol in symbols)


def delete_file_call_edges(repo_name: str, filename: str) -> int:
    """Delete all call edges from a file (used during incremental indexing). Returns count deleted."""
    schema_name = sanitize_repo_name(repo_name)
    edges_table = sql.Identifier(schema_name, "edges")

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("DELETE FROM {table} WHERE source_file = %s").format(table=edges_table), (filename,))
            deleted = cur.rowcount
            conn.commit()
            logger.debug(f"Deleted {deleted} call edges from {filename}")
            return deleted

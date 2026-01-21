"""Call graph construction and query APIs.

This module provides functionality for:
- Resolving function calls to symbols with confidence scores
- Storing call relationships in the edges table
- Querying call graphs (callers, callees, call chains)
"""

import logging
from dataclasses import dataclass
from typing import Optional
from psycopg import sql

from src.storage.postgres import get_connection
from src.storage.schema import sanitize_repo_name

logger = logging.getLogger(__name__)


@dataclass
class CallEdge:
    """Represents a call relationship between two symbols."""

    source_symbol_id: str  # UUID of the calling symbol
    target_symbol_id: Optional[str]  # UUID of the called symbol (None if unresolved)
    edge_type: str  # 'calls', 'implements', 'extends', 'overrides'
    source_file: str
    source_line: int
    target_file: Optional[str]
    target_symbol_name: str  # Name of the called function
    target_line: Optional[int]
    confidence: float  # 1.0=exact, 0.7=partial, 0.5=unresolved
    context: Optional[str] = None


@dataclass
class CallChainNode:
    """Node in a call chain."""

    symbol_id: str
    symbol_name: str
    filename: str
    line_start: int
    line_end: int
    depth: int  # How deep in the chain (0=root)
    edge_type: str  # Type of relationship


def resolve_call_to_symbol(
    repo_name: str,
    source_file: str,
    function_name: str,
    object_name: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], float]:
    """Resolve a function call to a symbol ID.

    Resolution strategy:
    1. Exact match: Same file, exact function name → confidence 1.0
    2. Imported symbol: Function name matches imported symbol → confidence 1.0
    3. Partial match: Function name matches but uncertain → confidence 0.7
    4. Unresolved: Not found (external library, dynamic) → confidence 0.5

    Args:
        repo_name: Repository name
        source_file: File where the call occurs
        function_name: Name of the function being called
        object_name: For method calls, the object/class name

    Returns:
        Tuple of (symbol_id, target_file, confidence)
    """
    schema_name = sanitize_repo_name(repo_name)
    symbols_table = sql.Identifier(schema_name, "symbols")

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Strategy 1: Exact match in same file
            cur.execute(
                sql.SQL("""
                    SELECT id, filename FROM {table}
                    WHERE filename = %s AND symbol_name = %s
                    LIMIT 1
                """).format(table=symbols_table),
                (source_file, function_name)
            )
            row = cur.fetchone()
            if row:
                return row[0], row[1], 1.0  # Exact match

            # Strategy 2: Method call with parent class
            if object_name:
                cur.execute(
                    sql.SQL("""
                        SELECT id, filename FROM {table}
                        WHERE symbol_name = %s AND parent_symbol = %s
                        LIMIT 1
                    """).format(table=symbols_table),
                    (function_name, object_name)
                )
                row = cur.fetchone()
                if row:
                    return row[0], row[1], 1.0  # Exact match via parent

            # Strategy 3: Imported symbol (check if function is imported from another file)
            # We'll look for any file that imports source_file and has the function
            # For simplicity, check all files for the function name
            cur.execute(
                sql.SQL("""
                    SELECT id, filename FROM {table}
                    WHERE symbol_name = %s AND visibility = 'public'
                    ORDER BY
                        CASE WHEN filename LIKE %s THEN 0 ELSE 1 END,
                        filename
                    LIMIT 1
                """).format(table=symbols_table),
                (function_name, f"%{function_name.lower()}%")  # Heuristic: file might be named after function
            )
            row = cur.fetchone()
            if row:
                return row[0], row[1], 0.7  # Partial match

            # Strategy 4: Unresolved (external library, dynamic call)
            return None, None, 0.5


def store_call_edge(repo_name: str, edge: CallEdge) -> bool:
    """Store a call edge in the database.

    Args:
        repo_name: Repository name
        edge: CallEdge to store

    Returns:
        True if successful, False otherwise
    """
    schema_name = sanitize_repo_name(repo_name)
    edges_table = sql.Identifier(schema_name, "edges")

    insert_sql = sql.SQL("""
        INSERT INTO {table} (
            source_symbol_id, target_symbol_id, edge_type,
            source_file, source_line, target_file,
            target_symbol_name, target_line, confidence, context
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """).format(table=edges_table)
    params = (
        edge.source_symbol_id,
        edge.target_symbol_id,
        edge.edge_type,
        edge.source_file,
        edge.source_line,
        edge.target_file,
        edge.target_symbol_name,
        edge.target_line,
        edge.confidence,
        edge.context,
    )

    def _insert() -> None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(insert_sql, params)
            conn.commit()

    try:
        _insert()
        return True
    except (UndefinedTable, UndefinedSchema) as e:
        # Fresh schema/table: create then retry once.
        logger.info(f"Edges table missing for {repo_name}, creating: {e}")
        try:
            from src.storage.schema import get_create_edges_table_sql, get_create_schema_sql

            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(get_create_schema_sql(repo_name))
                    cur.execute(get_create_edges_table_sql(repo_name))
                conn.commit()

            _insert()
            return True
        except Exception as e2:
            logger.error(f"Failed to create edges table for {repo_name}: {e2}")
            return False
    except Exception as e:
        logger.error(f"Failed to store call edge: {e}")
        return False


def _row_to_call_edge(row: tuple) -> CallEdge:
    """Convert a database row to a CallEdge object."""
    return CallEdge(
        source_symbol_id=row[0],
        target_symbol_id=row[1],
        edge_type=row[2],
        source_file=row[3],
        source_line=row[4],
        target_file=row[5],
        target_symbol_name=row[6],
        target_line=row[7],
        confidence=row[8],
        context=row[9],
    )


def _query_edges(
    repo_name: str,
    symbol_id: str,
    direction: str,
    min_confidence: float,
) -> list[CallEdge]:
    """Query edges by symbol ID in a given direction.

    Args:
        repo_name: Repository name
        symbol_id: UUID of the symbol
        direction: 'callers' or 'callees'
        min_confidence: Minimum confidence threshold

    Returns:
        List of CallEdge objects
    """
    schema_name = sanitize_repo_name(repo_name)
    edges_table = sql.Identifier(schema_name, "edges")

    if direction == 'callers':
        where_clause = "target_symbol_id = %s"
        order_clause = "source_file, source_line"
    else:
        where_clause = "source_symbol_id = %s"
        order_clause = "target_file, target_line"

    query = sql.SQL("""
        SELECT
            source_symbol_id, target_symbol_id, edge_type,
            source_file, source_line, target_file,
            target_symbol_name, target_line, confidence, context
        FROM {table}
        WHERE """ + where_clause + """ AND confidence >= %s
        ORDER BY confidence DESC, """ + order_clause
    ).format(table=edges_table)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (symbol_id, min_confidence))
            return [_row_to_call_edge(row) for row in cur.fetchall()]


def get_callers(repo_name: str, symbol_id: str, min_confidence: float = 0.7) -> list[CallEdge]:
    """Get all functions that call the given symbol.

    Args:
        repo_name: Repository name
        symbol_id: UUID of the symbol
        min_confidence: Minimum confidence threshold (default: 0.7)

    Returns:
        List of CallEdge objects representing callers
    """
    return _query_edges(repo_name, symbol_id, 'callers', min_confidence)


def get_callees(repo_name: str, symbol_id: str, min_confidence: float = 0.7) -> list[CallEdge]:
    """Get all functions called by the given symbol.

    Args:
        repo_name: Repository name
        symbol_id: UUID of the symbol
        min_confidence: Minimum confidence threshold (default: 0.7)

    Returns:
        List of CallEdge objects representing callees
    """
    return _query_edges(repo_name, symbol_id, 'callees', min_confidence)


def trace_call_chain(
    repo_name: str,
    symbol_id: str,
    max_depth: int = 5,
    direction: str = 'callees',
    min_confidence: float = 0.7,
) -> list[CallChainNode]:
    """Trace the call chain from a symbol.

    Args:
        repo_name: Repository name
        symbol_id: Starting symbol UUID
        max_depth: Maximum depth to traverse
        direction: 'callees' (what this calls) or 'callers' (what calls this)
        min_confidence: Minimum confidence threshold

    Returns:
        List of CallChainNode objects representing the call chain
    """
    schema_name = sanitize_repo_name(repo_name)
    edges_table = sql.Identifier(schema_name, "edges")
    symbols_table = sql.Identifier(schema_name, "symbols")

    chain = []
    visited = set([symbol_id])
    queue = [(symbol_id, 0)]  # (symbol_id, depth)

    with get_connection() as conn:
        with conn.cursor() as cur:
            while queue:
                current_id, depth = queue.pop(0)

                if depth >= max_depth:
                    continue

                # Get symbol info
                cur.execute(
                    sql.SQL("""
                        SELECT symbol_name, filename, line_start, line_end
                        FROM {table}
                        WHERE id = %s
                    """).format(table=symbols_table),
                    (current_id,)
                )
                symbol_row = cur.fetchone()
                if not symbol_row:
                    continue

                symbol_name, filename, line_start, line_end = symbol_row

                # Get edges based on direction
                if direction == 'callees':
                    # What this symbol calls
                    cur.execute(
                        sql.SQL("""
                            SELECT target_symbol_id, edge_type
                            FROM {table}
                            WHERE source_symbol_id = %s AND confidence >= %s AND target_symbol_id IS NOT NULL
                        """).format(table=edges_table),
                        (current_id, min_confidence)
                    )
                else:  # direction == 'callers'
                    # What calls this symbol
                    cur.execute(
                        sql.SQL("""
                            SELECT source_symbol_id, edge_type
                            FROM {table}
                            WHERE target_symbol_id = %s AND confidence >= %s
                        """).format(table=edges_table),
                        (current_id, min_confidence)
                    )

                edges = cur.fetchall()

                # Add current node to chain
                if depth > 0:  # Skip root
                    chain.append(CallChainNode(
                        symbol_id=current_id,
                        symbol_name=symbol_name,
                        filename=filename,
                        line_start=line_start,
                        line_end=line_end,
                        depth=depth,
                        edge_type=edges[0][1] if edges else 'calls',
                    ))

                # Queue next level
                for next_id, edge_type in edges:
                    if next_id not in visited:
                        visited.add(next_id)
                        queue.append((next_id, depth + 1))

    return chain


def delete_symbol_edges(repo_name: str, symbol_id: str) -> int:
    """Delete all edges associated with a symbol.

    Args:
        repo_name: Repository name
        symbol_id: Symbol UUID

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
                    WHERE source_symbol_id = %s OR target_symbol_id = %s
                """).format(table=edges_table),
                (symbol_id, symbol_id)
            )
            deleted = cur.rowcount
            conn.commit()
            return deleted


@dataclass
class SymbolReference:
    """A reference to a symbol (definition or usage)."""
    filename: str
    line: int
    ref_type: str  # 'definition', 'call', 'import'
    context: str | None = None
    confidence: float = 1.0


def find_symbol_references(
    repo_name: str,
    symbol_name: str,
    include_definitions: bool = True,
    include_calls: bool = True,
    min_confidence: float = 0.5,
) -> list[SymbolReference]:
    """Find all references to a symbol across the codebase.
    
    Args:
        repo_name: Repository name
        symbol_name: Name of the symbol to find
        include_definitions: Include symbol definitions
        include_calls: Include call sites
        min_confidence: Minimum confidence for call references
    
    Returns:
        List of SymbolReference objects sorted by filename, line
    """
    schema_name = sanitize_repo_name(repo_name)
    symbols_table = sql.Identifier(schema_name, "symbols")
    edges_table = sql.Identifier(schema_name, "edges")
    
    references: list[SymbolReference] = []
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Find definitions
            if include_definitions:
                cur.execute(
                    sql.SQL("""
                        SELECT filename, line_start, signature
                        FROM {table}
                        WHERE symbol_name = %s
                        ORDER BY filename, line_start
                    """).format(table=symbols_table),
                    (symbol_name,)
                )
                for row in cur.fetchall():
                    references.append(SymbolReference(
                        filename=row[0],
                        line=row[1],
                        ref_type='definition',
                        context=row[2],
                        confidence=1.0,
                    ))
            
            # Find call sites
            if include_calls:
                try:
                    cur.execute(
                        sql.SQL("""
                            SELECT source_file, source_line, context, confidence
                            FROM {table}
                            WHERE target_symbol_name = %s AND confidence >= %s
                            ORDER BY source_file, source_line
                        """).format(table=edges_table),
                        (symbol_name, min_confidence)
                    )
                    for row in cur.fetchall():
                        references.append(SymbolReference(
                            filename=row[0],
                            line=row[1],
                            ref_type='call',
                            context=row[2],
                            confidence=row[3],
                        ))
                except Exception as e:
                    # Edges table may not exist yet - log at debug level
                    logger.debug(f"Could not query edges table: {e}")
    
    # Sort by filename, then line
    references.sort(key=lambda r: (r.filename, r.line))
    return references


def get_symbol_usages_summary(repo_name: str, symbol_name: str) -> dict:
    """Get a summary of symbol usage across the codebase.
    
    Returns:
        Dict with definition_count, call_count, files list
    """
    refs = find_symbol_references(repo_name, symbol_name)
    
    definitions = [r for r in refs if r.ref_type == 'definition']
    calls = [r for r in refs if r.ref_type == 'call']
    files = sorted(set(r.filename for r in refs))
    
    return {
        'symbol_name': symbol_name,
        'definition_count': len(definitions),
        'call_count': len(calls),
        'total_references': len(refs),
        'files': files,
        'definitions': [{'file': d.filename, 'line': d.line} for d in definitions],
        'calls': [{'file': c.filename, 'line': c.line, 'confidence': c.confidence} for c in calls],
    }

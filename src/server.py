"""FastMCP server for precise code symbol retrieval."""

import logging
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastmcp import FastMCP

from config.settings import settings
from src.exceptions import IndexingError, PathError, SearchError
from src.indexer.service import get_indexer
from src.retrieval.service import get_searcher
from src.storage.postgres import close_pool, init_db, get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MAX_SYMBOL_NAME_LENGTH = 500

mcp = FastMCP(
    "cocode-precise",
    instructions="Precise code retrieval. Use get_symbol to get exact function/class code, or search_symbols to find symbols by name.",
)


def _get_symbol_from_db(repo_name: str, symbol_name: str, symbol_type: str | None = None) -> list[dict]:
    """Query symbols table for matching symbols."""
    from src.storage.schema import sanitize_repo_name
    schema = sanitize_repo_name(repo_name)
    
    with get_connection() as conn:
        with conn.cursor() as cur:
            if symbol_type:
                cur.execute(f"""
                    SELECT symbol_name, symbol_type, filename, line_start, line_end, signature, docstring
                    FROM {schema}.symbols
                    WHERE symbol_name ILIKE %s AND symbol_type = %s
                    ORDER BY symbol_name
                    LIMIT 20
                """, (f"%{symbol_name}%", symbol_type))
            else:
                cur.execute(f"""
                    SELECT symbol_name, symbol_type, filename, line_start, line_end, signature, docstring
                    FROM {schema}.symbols
                    WHERE symbol_name ILIKE %s
                    ORDER BY symbol_name
                    LIMIT 20
                """, (f"%{symbol_name}%",))
            
            return [
                {
                    "name": row[0],
                    "type": row[1],
                    "filename": row[2],
                    "line_start": row[3],
                    "line_end": row[4],
                    "signature": row[5],
                    "docstring": row[6],
                }
                for row in cur.fetchall()
            ]


def _read_file_lines(filepath: Path, start: int, end: int) -> str:
    """Read specific lines from a file (1-indexed, inclusive)."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    
    # Convert to 0-indexed
    start_idx = max(0, start - 1)
    end_idx = min(len(lines), end)
    
    return "".join(lines[start_idx:end_idx])


@mcp.tool()
async def get_symbol(
    symbol_name: str,
    path: str | None = None,
    symbol_type: str | None = None,
) -> list[dict]:
    """Get the complete source code of a function, class, or method by name.

    Args:
        symbol_name: Name of the function/class/method to retrieve (supports partial match)
        path: Path to the codebase (defaults to cwd)
        symbol_type: Filter by type: 'function', 'class', or 'method' (optional)

    Returns:
        List of matching symbols with their complete source code
    """
    if not symbol_name or not symbol_name.strip():
        return [{"error": "symbol_name cannot be empty"}]
    if len(symbol_name) > MAX_SYMBOL_NAME_LENGTH:
        return [{"error": f"symbol_name too long (max {MAX_SYMBOL_NAME_LENGTH})"}]
    
    if path is None:
        path = os.getcwd()

    try:
        indexer = get_indexer()
        index_result = indexer.ensure_indexed(path)
        repo_path = Path(path).resolve()

        if index_result.chunk_count == 0:
            return [{"error": f"No code files found in {path}"}]

        # Find matching symbols
        symbols = _get_symbol_from_db(index_result.repo_name, symbol_name.strip(), symbol_type)
        
        if not symbols:
            return [{"message": f"No symbol found matching '{symbol_name}'"}]

        # Read actual source code for each symbol
        results = []
        for sym in symbols:
            filepath = repo_path / sym["filename"]
            if filepath.exists():
                code = _read_file_lines(filepath, sym["line_start"], sym["line_end"])
                results.append({
                    "name": sym["name"],
                    "type": sym["type"],
                    "filename": sym["filename"],
                    "lines": f"L{sym['line_start']}-{sym['line_end']}",
                    "signature": sym["signature"],
                    "docstring": sym["docstring"],
                    "code": code,
                })
            else:
                results.append({
                    "name": sym["name"],
                    "type": sym["type"],
                    "filename": sym["filename"],
                    "lines": f"L{sym['line_start']}-{sym['line_end']}",
                    "error": "File not found",
                })

        return results

    except PathError as e:
        logger.warning(f"Path error: {e}")
        return [{"error": "Invalid path specified"}]
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return [{"error": f"Failed to retrieve symbol: {e}"}]


@mcp.tool()
async def search_symbols(
    query: str,
    path: str | None = None,
    symbol_type: str | None = None,
    top_k: int = 10,
) -> list[dict]:
    """Search for symbols (functions/classes/methods) by semantic similarity.

    Args:
        query: Natural language description of what you're looking for
        path: Path to the codebase (defaults to cwd)
        symbol_type: Filter by type: 'function', 'class', or 'method' (optional)
        top_k: Number of results to return (default: 10)

    Returns:
        List of matching symbols with signatures and locations (use get_symbol to get full code)
    """
    if not query or not query.strip():
        return [{"error": "query cannot be empty"}]
    
    if path is None:
        path = os.getcwd()

    try:
        indexer = get_indexer()
        index_result = indexer.ensure_indexed(path)

        if index_result.chunk_count == 0:
            return [{"error": f"No code files found in {path}"}]

        # Use symbol search from hybrid search
        from src.retrieval.symbol_search import symbol_hybrid_search
        from src.retrieval.hybrid import get_query_embedding
        
        query_embedding = get_query_embedding(query.strip())
        results = symbol_hybrid_search(
            index_result.repo_name,
            query.strip(),
            top_k=top_k,
            query_embedding=query_embedding,
        )

        if not results:
            return [{"message": f"No symbols found matching '{query}'"}]

        # Format results
        return [
            {
                "name": r.filename.split("::")[-1] if "::" in r.filename else Path(r.filename).stem,
                "filename": r.filename.split("::")[0] if "::" in r.filename else r.filename,
                "location": r.location,
                "score": round(r.score, 4),
                "preview": r.content[:200] + "..." if len(r.content) > 200 else r.content,
            }
            for r in results
        ]

    except PathError as e:
        logger.warning(f"Path error: {e}")
        return [{"error": "Invalid path specified"}]
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return [{"error": f"Search failed: {e}"}]


@mcp.tool()
async def list_symbols(
    path: str | None = None,
    filename: str | None = None,
    symbol_type: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """List all symbols in a codebase or specific file.

    Args:
        path: Path to the codebase (defaults to cwd)
        filename: Filter to specific file (optional, relative path)
        symbol_type: Filter by type: 'function', 'class', or 'method' (optional)
        limit: Maximum number of symbols to return (default: 50)

    Returns:
        List of symbols with their signatures and locations
    """
    if path is None:
        path = os.getcwd()

    try:
        indexer = get_indexer()
        index_result = indexer.ensure_indexed(path)

        if index_result.chunk_count == 0:
            return [{"error": f"No code files found in {path}"}]

        from src.storage.schema import sanitize_repo_name
        schema = sanitize_repo_name(index_result.repo_name)
        
        with get_connection() as conn:
            with conn.cursor() as cur:
                conditions = []
                params = []
                
                if filename:
                    conditions.append("filename ILIKE %s")
                    params.append(f"%{filename}%")
                if symbol_type:
                    conditions.append("symbol_type = %s")
                    params.append(symbol_type)
                
                where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
                params.append(min(limit, 100))
                
                cur.execute(f"""
                    SELECT symbol_name, symbol_type, filename, line_start, line_end, signature
                    FROM {schema}.symbols
                    {where}
                    ORDER BY filename, line_start
                    LIMIT %s
                """, params)
                
                return [
                    {
                        "name": row[0],
                        "type": row[1],
                        "filename": row[2],
                        "lines": f"L{row[3]}-{row[4]}",
                        "signature": row[5],
                    }
                    for row in cur.fetchall()
                ]

    except PathError as e:
        logger.warning(f"Path error: {e}")
        return [{"error": "Invalid path specified"}]
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return [{"error": f"Failed to list symbols: {e}"}]


@mcp.tool()
async def read_file(
    filename: str,
    path: str | None = None,
    start_line: int | None = None,
    end_line: int | None = None,
) -> dict:
    """Read a file or specific line range from the codebase.

    Args:
        filename: Relative path to the file
        path: Path to the codebase (defaults to cwd)
        start_line: Starting line number, 1-indexed (optional)
        end_line: Ending line number, 1-indexed, inclusive (optional)

    Returns:
        File content with metadata
    """
    if not filename or not filename.strip():
        return {"error": "filename cannot be empty"}
    
    if path is None:
        path = os.getcwd()

    try:
        repo_path = Path(path).resolve()
        filepath = repo_path / filename.strip()
        
        # Security: ensure file is within repo
        if not filepath.resolve().is_relative_to(repo_path):
            return {"error": "File path outside repository"}
        
        if not filepath.exists():
            return {"error": f"File not found: {filename}"}
        
        if not filepath.is_file():
            return {"error": f"Not a file: {filename}"}

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        
        total_lines = len(lines)
        
        if start_line is not None or end_line is not None:
            start = max(1, start_line or 1)
            end = min(total_lines, end_line or total_lines)
            content = "".join(lines[start-1:end])
            return {
                "filename": filename,
                "lines": f"L{start}-{end}",
                "total_lines": total_lines,
                "content": content,
            }
        else:
            return {
                "filename": filename,
                "total_lines": total_lines,
                "content": "".join(lines),
            }

    except Exception as e:
        logger.exception(f"Failed to read file: {e}")
        return {"error": f"Failed to read file: {e}"}


@mcp.tool()
async def clear_index(path: str | None = None) -> dict:
    """Clear the index for a codebase to force re-indexing.

    Args:
        path: Path to the codebase (defaults to cwd)

    Returns:
        Status message
    """
    if path is None:
        path = os.getcwd()

    try:
        indexer = get_indexer()
        resolved_path = Path(path).resolve()

        if not resolved_path.exists() or not resolved_path.is_dir():
            return {"error": "Invalid path specified"}

        repo_name = indexer.resolve_repo_name(resolved_path)
        indexer._delete_index(repo_name)

        return {"status": "cleared", "message": "Index cleared. Next operation will rebuild."}
    except Exception as e:
        logger.error(f"Failed to clear index: {e}")
        return {"error": "Failed to clear index"}


def main() -> None:
    """Main entry point for MCP server."""
    logger.info("Starting cocode-precise MCP server...")

    if not settings.openai_api_key and not (settings.jina_api_key and settings.use_late_chunking):
        logger.error("OPENAI_API_KEY or JINA_API_KEY (with USE_LATE_CHUNKING=true) required")
        sys.exit(1)

    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)

    def shutdown(sig, frame) -> None:
        logger.info(f"Received signal {sig}, shutting down...")
        close_pool()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    try:
        mcp.run()
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)
    finally:
        close_pool()
        logger.info("Server stopped")


if __name__ == "__main__":
    main()

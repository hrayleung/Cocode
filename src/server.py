"""FastMCP server for semantic code search."""

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
from src.retrieval.dependencies import get_import_edges
from src.retrieval.file_categorizer import categorize_file
from src.retrieval.service import extract_signature, get_searcher
from src.retrieval.symbol_implementation import (
    extract_symbol_code,
    select_top_symbols,
    symbol_hybrid_search_with_metadata,
)
from src.storage.postgres import close_pool, init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MIN_TOP_K = 1
MAX_TOP_K = 100
MAX_QUERY_LENGTH = 50_000

mcp = FastMCP(
    "cocode",
    instructions=(
        "Semantic code search. Call codebase_retrieval_full with your question - indexing is automatic. "
        "It returns key files, file dependencies, and full symbol implementations. "
        "Use clear_index(path) to force a re-index when needed."
    ),
)


def _validate_query(query: str) -> str | None:
    """Validate query and return error message if invalid, None if valid."""
    if not query or not query.strip():
        return "Query cannot be empty"
    if len(query) > MAX_QUERY_LENGTH:
        return f"Query too long (max {MAX_QUERY_LENGTH} characters)"
    return None


def _validate_top_k(top_k: int) -> str | None:
    """Validate top_k and return error message if invalid."""
    if top_k < MIN_TOP_K:
        return f"top_k must be at least {MIN_TOP_K}"
    if top_k > MAX_TOP_K:
        return f"top_k cannot exceed {MAX_TOP_K}"
    return None


def _validate_range(value: int, name: str, min_val: int, max_val: int) -> str | None:
    """Validate a numeric value is within range."""
    if value < min_val:
        return f"{name} must be >= {min_val}"
    if value > max_val:
        return f"{name} cannot exceed {max_val}"
    return None


def _handle_search_error(e: Exception) -> dict:
    """Convert exception to error response dict."""
    if isinstance(e, PathError):
        logger.warning(f"Path error: {e}")
        return {"error": "Invalid path specified"}
    if isinstance(e, IndexingError):
        logger.error(f"Indexing error: {e}")
        return {"error": "Failed to index repository"}
    if isinstance(e, SearchError):
        logger.error(f"Search error: {e}")
        return {"error": "Search operation failed"}
    logger.exception(f"Unexpected error: {e}")
    return {"error": "An unexpected error occurred"}


@mcp.tool()
async def codebase_retrieval_full(
    query: str,
    path: str | None = None,
    top_k: int = 10,
    max_symbols: int = 15,
    max_symbols_per_file: int = 3,
    max_code_chars: int = 50_000,
    include_dependencies: bool = True,
    prefer_concise_symbols: bool = True,
    max_symbol_lines: int = 220,
) -> dict:
    """Search a codebase and return full symbol implementations.

    This tool is optimized for coding agents that want:
    - key files for a query
    - dependency edges between those files
    - full implementations of the most relevant functions/classes/methods

    Args:
        query: Natural language question about the code
        path: Path to the codebase (defaults to cwd)
        top_k: Number of files to return (1-100, default: 10)
        max_symbols: Max number of symbol implementations to return
        max_symbols_per_file: Max symbol implementations per file
        max_code_chars: Max characters of code returned per symbol (line-preserving)
        include_dependencies: Whether to compute import edges between returned files
        prefer_concise_symbols: Prefer function/method symbols over huge class dumps
        max_symbol_lines: When prefer_concise_symbols, exclude symbols longer than this many lines (best-effort)

    Returns:
        Dict with keys: files, dependencies, symbols
    """
    # Validate inputs
    if err := _validate_query(query):
        return {"error": err}
    if err := _validate_top_k(top_k):
        return {"error": err}
    if err := _validate_range(max_symbols, "max_symbols", 1, 100):
        return {"error": err}
    if err := _validate_range(max_symbols_per_file, "max_symbols_per_file", 1, 20):
        return {"error": err}
    if err := _validate_range(max_code_chars, "max_code_chars", 1_000, 500_000):
        return {"error": err}
    if err := _validate_range(max_symbol_lines, "max_symbol_lines", 1, 1000):
        return {"error": err}

    path = path or os.getcwd()
    q = query.strip()

    try:
        indexer = get_indexer()
        index_result = indexer.ensure_indexed(path)

        logger.info(f"Index: {index_result.status} ({index_result.file_count} files, {index_result.chunk_count} chunks)")

        if index_result.chunk_count == 0:
            return {"error": f"No code files found in {path}"}

        # Search for files
        file_snippets = get_searcher().search(
            repo_name=index_result.repo_name,
            query=q,
            top_k=top_k,
        )

        files, file_order, seen_files = _build_file_results(file_snippets)

        # Search for symbols
        symbols, symbol_files = _search_symbols(
            index_result.repo_name, path, q, file_order,
            max_symbols, max_symbols_per_file, max_code_chars,
            prefer_concise_symbols, max_symbol_lines
        )

        # Add files from symbols not already in results
        for f in sorted(symbol_files - seen_files):
            file_order.append(f)
            files.append({
                "filename": f,
                "score": None,
                "category": categorize_file(f),
                "lines": [],
                "preview": "",
                "is_reference_only": True,
                "source": "symbol",
            })

        # Get dependencies
        dependencies = []
        if include_dependencies:
            try:
                dependencies = get_import_edges(index_result.repo_name, file_order)
            except Exception as e:
                logger.warning(f"Dependency graph failed: {e}")

        return {"query": q, "files": files, "dependencies": dependencies, "symbols": symbols}

    except Exception as e:
        return _handle_search_error(e)


def _build_file_results(file_snippets: list) -> tuple[list[dict], list[str], set[str]]:
    """Build file results from search snippets."""
    files: list[dict] = []
    file_order: list[str] = []
    seen_files: set[str] = set()

    for snip in file_snippets:
        if snip.filename in seen_files:
            continue
        seen_files.add(snip.filename)
        file_order.append(snip.filename)

        source = "graph" if snip.is_reference_only and snip.content == "[Related via imports]" else "search"
        files.append({
            "filename": snip.filename,
            "score": round(snip.score, 4),
            "category": categorize_file(snip.filename),
            "lines": snip.locations,
            "preview": extract_signature(snip.content) if snip.content else "",
            "is_reference_only": snip.is_reference_only,
            "source": source,
        })

    return files, file_order, seen_files


def _search_symbols(
    repo_name: str,
    repo_path: str,
    query: str,
    file_order: list[str],
    max_symbols: int,
    max_symbols_per_file: int,
    max_code_chars: int,
    prefer_concise: bool,
    max_symbol_lines: int,
) -> tuple[list[dict], set[str]]:
    """Search and extract symbol implementations."""
    symbols: list[dict] = []
    symbol_files: set[str] = set()

    try:
        # Scope to implementation files if available
        symbol_scope = [f for f in file_order if categorize_file(f) == "implementation"] or None

        candidates = symbol_hybrid_search_with_metadata(
            repo_name=repo_name,
            query=query,
            top_k=min(max_symbols * 5, 200),
            filenames=symbol_scope,
        )

        # Fallback to repo-wide if scoped search finds nothing
        if not candidates and symbol_scope:
            candidates = symbol_hybrid_search_with_metadata(
                repo_name=repo_name,
                query=query,
                top_k=min(max_symbols * 5, 200),
            )

        # Filter for concise symbols if requested
        if prefer_concise and candidates:
            preferred = [c for c in candidates if c.symbol_type in ("function", "method")]
            candidates = preferred or candidates
            limited = [c for c in candidates if (c.line_end - c.line_start + 1) <= max_symbol_lines]
            candidates = limited or candidates

        selected = select_top_symbols(candidates, max_symbols=max_symbols, max_symbols_per_file=max_symbols_per_file)

        for hit in selected:
            symbol_files.add(hit.filename)
            symbols.append(_extract_symbol_info(repo_path, hit, max_code_chars))

    except Exception as e:
        logger.warning(f"Symbol retrieval failed: {e}")

    return symbols, symbol_files


def _extract_symbol_info(repo_path: str, hit, max_code_chars: int) -> dict:
    """Extract symbol information and code."""
    base = {
        "filename": hit.filename,
        "symbol_name": hit.symbol_name,
        "symbol_type": hit.symbol_type,
        "line_start": hit.line_start,
        "line_end": hit.line_end,
        "score": round(hit.score, 6),
    }

    try:
        code_info = extract_symbol_code(
            repo_path=repo_path,
            filename=hit.filename,
            line_start=hit.line_start,
            line_end=hit.line_end,
            max_code_chars=max_code_chars,
        )
        return {
            **base,
            "extracted_line_start": code_info["extracted_line_start"],
            "extracted_line_end": code_info["extracted_line_end"],
            "file_line_count": code_info["file_line_count"],
            "signature": hit.signature,
            "docstring": hit.docstring,
            "parent_symbol": hit.parent_symbol,
            "visibility": hit.visibility,
            "category": hit.category,
            "truncated": bool(code_info["truncated"]),
            "code": code_info["code"],
        }
    except Exception as e:
        return {**base, "error": f"Failed to extract code: {e}"}


@mcp.tool()
async def clear_index(path: str | None = None) -> dict:
    """Clear the index for a codebase to force re-indexing.

    Args:
        path: Path to the codebase (defaults to cwd)

    Returns:
        Status message
    """
    path = path or os.getcwd()

    try:
        indexer = get_indexer()
        resolved_path = Path(path).resolve()

        if not resolved_path.exists() or not resolved_path.is_dir():
            logger.warning(f"Invalid path provided: {path}")
            return {"error": "Invalid path specified"}

        repo_name = indexer.resolve_repo_name(resolved_path)
        indexer._delete_index(repo_name)

        return {"status": "cleared", "message": "Index cleared. Next search will rebuild."}
    except Exception as e:
        logger.error(f"Failed to clear index: {e}")
        return {"error": "Failed to clear index"}


def main() -> None:
    """Main entry point for MCP server."""
    logger.info("Starting cocode MCP server...")

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

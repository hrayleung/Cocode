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
from src.retrieval.curation import curate_code_sections
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
MAX_QUERY_LENGTH = 50_000  # 50KB maximum query length

mcp = FastMCP(
    "cocode",
    instructions=(
        "Semantic code search. Call codebase_retrieval with your question - indexing is automatic. "
        "Call codebase_retrieval_full to get key files, file dependencies, and full symbol implementations."
    ),
)


@mcp.tool()
async def codebase_retrieval(
    query: str,
    path: str | None = None,
    top_k: int = 10,
    max_output_chars: int = 20_000,
    max_files: int = 4,
    max_sections: int = 8,
    include_docs: bool = True,
) -> list[dict]:
    """Search a codebase and return curated code sections.

    This tool is optimized for LLM consumption (similar to Augment's
    codebase-retrieval): it returns a small set of contiguous code sections
    under a global output budget.

    Args:
        query: Natural language question about the code
        path: Path to the codebase (defaults to cwd)
        top_k: Legacy compatibility (upper bound for file candidate pool; 1-100)
        max_output_chars: Total character budget for returned code (default: 20,000)
        max_files: Max distinct files to draw sections from (default: 4)
        max_sections: Max number of sections to return (default: 8)
        include_docs: Whether to include at most one documentation file section

    Returns:
        List of code sections with filename, line range and content
    """
    if not query or not query.strip():
        return [{"error": "Query cannot be empty"}]
    if len(query) > MAX_QUERY_LENGTH:
        return [{"error": f"Query too long (max {MAX_QUERY_LENGTH} characters)"}]
    if top_k < MIN_TOP_K:
        return [{"error": f"top_k must be at least {MIN_TOP_K}"}]
    if top_k > MAX_TOP_K:
        return [{"error": f"top_k cannot exceed {MAX_TOP_K}"}]

    if path is None:
        path = os.getcwd()

    try:
        indexer = get_indexer()
        index_result = indexer.ensure_indexed(path)

        logger.info(f"Index: {index_result.status} ({index_result.file_count} files, {index_result.chunk_count} chunks)")

        if index_result.chunk_count == 0:
            return [{"error": f"No code files found in {path}"}]

        # Curated sections (Augment-style): small number of contiguous ranges,
        # extracted from disk using symbol/chunk evidence and bounded by a budget.
        sections = curate_code_sections(
            repo_name=index_result.repo_name,
            repo_path=path,
            query=query.strip(),
            max_output_chars=max_output_chars,
            max_files=min(max_files, top_k),
            max_sections=max_sections,
            include_docs=include_docs,
        )

        logger.info(f"Curated retrieval returned {len(sections)} sections")

        if not sections:
            return [{"message": "No matching code found", "query": query.strip()}]

        return sections

    except PathError as e:
        logger.warning(f"Path error: {e}")
        return [{"error": "Invalid path specified"}]
    except IndexingError as e:
        logger.error(f"Indexing error: {e}")
        return [{"error": "Failed to index repository"}]
    except SearchError as e:
        logger.error(f"Search error: {e}")
        return [{"error": "Search operation failed"}]
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return [{"error": "An unexpected error occurred"}]


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
    if not query or not query.strip():
        return {"error": "Query cannot be empty"}
    if len(query) > MAX_QUERY_LENGTH:
        return {"error": f"Query too long (max {MAX_QUERY_LENGTH} characters)"}
    if top_k < MIN_TOP_K:
        return {"error": f"top_k must be at least {MIN_TOP_K}"}
    if top_k > MAX_TOP_K:
        return {"error": f"top_k cannot exceed {MAX_TOP_K}"}

    if max_symbols < 1:
        return {"error": "max_symbols must be >= 1"}
    if max_symbols > 100:
        return {"error": "max_symbols cannot exceed 100"}

    if max_symbols_per_file < 1:
        return {"error": "max_symbols_per_file must be >= 1"}
    if max_symbols_per_file > 20:
        return {"error": "max_symbols_per_file cannot exceed 20"}

    if max_code_chars < 1_000:
        return {"error": "max_code_chars must be >= 1000"}
    if max_code_chars > 500_000:
        return {"error": "max_code_chars cannot exceed 500000"}

    if path is None:
        path = os.getcwd()

    q = query.strip()

    try:
        indexer = get_indexer()
        index_result = indexer.ensure_indexed(path)

        logger.info(f"Index: {index_result.status} ({index_result.file_count} files, {index_result.chunk_count} chunks)")

        if index_result.chunk_count == 0:
            return {"error": f"No code files found in {path}"}

        file_snippets = get_searcher().search(
            repo_name=index_result.repo_name,
            query=q,
            top_k=top_k,
        )

        files: list[dict] = []
        file_order: list[str] = []
        seen_files: set[str] = set()

        for snip in file_snippets:
            if snip.filename in seen_files:
                continue
            seen_files.add(snip.filename)
            file_order.append(snip.filename)

            entry = {
                "filename": snip.filename,
                "score": round(snip.score, 4),
                "category": categorize_file(snip.filename),
                "lines": snip.locations,
                "preview": extract_signature(snip.content) if snip.content else "",
                "is_reference_only": snip.is_reference_only,
            }
            # Mark graph-expanded files explicitly
            if snip.is_reference_only and snip.content == "[Related via imports]":
                entry["source"] = "graph"
            else:
                entry["source"] = "search"
            files.append(entry)

        # Symbol search (repo-wide); then extract full code bodies from disk.
        symbols: list[dict] = []
        symbol_files: set[str] = set()

        try:
            symbol_scope = [f for f in file_order if categorize_file(f) == "implementation"]
            if not symbol_scope:
                symbol_scope = None

            candidates = symbol_hybrid_search_with_metadata(
                repo_name=index_result.repo_name,
                query=q,
                top_k=min(max_symbols * 5, 200),
                filenames=symbol_scope,
            )

            # Fallback: if scoped search finds nothing, try repo-wide.
            if not candidates and symbol_scope is not None:
                candidates = symbol_hybrid_search_with_metadata(
                    repo_name=index_result.repo_name,
                    query=q,
                    top_k=min(max_symbols * 5, 200),
                )

            # By default, prefer smaller function/method bodies over huge class dumps.
            if prefer_concise_symbols and candidates:
                preferred = [c for c in candidates if c.symbol_type in ("function", "method")]
                if preferred:
                    candidates = preferred
                limited = [
                    c for c in candidates
                    if (c.line_end - c.line_start + 1) <= max_symbol_lines
                ]
                if limited:
                    candidates = limited

            selected = select_top_symbols(
                candidates,
                max_symbols=max_symbols,
                max_symbols_per_file=max_symbols_per_file,
            )

            for hit in selected:
                symbol_files.add(hit.filename)
                try:
                    code_info = extract_symbol_code(
                        repo_path=path,
                        filename=hit.filename,
                        line_start=hit.line_start,
                        line_end=hit.line_end,
                        max_code_chars=max_code_chars,
                    )
                    symbols.append(
                        {
                            "filename": hit.filename,
                            "symbol_name": hit.symbol_name,
                            "symbol_type": hit.symbol_type,
                            "line_start": hit.line_start,
                            "line_end": hit.line_end,
                            "extracted_line_start": code_info["extracted_line_start"],
                            "extracted_line_end": code_info["extracted_line_end"],
                            "file_line_count": code_info["file_line_count"],
                            "signature": hit.signature,
                            "docstring": hit.docstring,
                            "parent_symbol": hit.parent_symbol,
                            "visibility": hit.visibility,
                            "category": hit.category,
                            "score": round(hit.score, 6),
                            "truncated": bool(code_info["truncated"]),
                            "code": code_info["code"],
                        }
                    )
                except Exception as e:
                    symbols.append(
                        {
                            "filename": hit.filename,
                            "symbol_name": hit.symbol_name,
                            "symbol_type": hit.symbol_type,
                            "line_start": hit.line_start,
                            "line_end": hit.line_end,
                            "score": round(hit.score, 6),
                            "error": f"Failed to extract code: {e}",
                        }
                    )
        except Exception as e:
            logger.warning(f"Symbol retrieval failed: {e}")

        # Merge in files that appear only via selected symbols.
        for f in sorted(symbol_files):
            if f in seen_files:
                continue
            seen_files.add(f)
            file_order.append(f)
            files.append(
                {
                    "filename": f,
                    "score": None,
                    "category": categorize_file(f),
                    "lines": [],
                    "preview": "",
                    "is_reference_only": True,
                    "source": "symbol",
                }
            )

        dependencies = []
        if include_dependencies:
            try:
                dependencies = get_import_edges(index_result.repo_name, file_order)
            except Exception as e:
                logger.warning(f"Dependency graph failed: {e}")

        return {
            "query": q,
            "files": files,
            "dependencies": dependencies,
            "symbols": symbols,
        }

    except PathError as e:
        logger.warning(f"Path error: {e}")
        return {"error": "Invalid path specified"}
    except IndexingError as e:
        logger.error(f"Indexing error: {e}")
        return {"error": "Failed to index repository"}
    except SearchError as e:
        logger.error(f"Search error: {e}")
        return {"error": "Search operation failed"}
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return {"error": "An unexpected error occurred"}


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
            logger.warning(f"Invalid path provided: {path}")
            return {"error": "Invalid path specified"}

        repo_name = indexer.resolve_repo_name(resolved_path)
        indexer._delete_index(repo_name)

        return {
            "status": "cleared",
            "message": "Index cleared. Next search will rebuild.",
        }
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

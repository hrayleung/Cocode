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
from src.retrieval.service import get_searcher
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
    instructions="Semantic code search. Call codebase_retrieval with your question - indexing is automatic.",
)


@mcp.tool()
async def codebase_retrieval(
    query: str,
    path: str | None = None,
    top_k: int = 10,
) -> list[dict]:
    """Search a codebase using semantic and keyword search.

    Args:
        query: Natural language question about the code
        path: Path to the codebase (defaults to cwd)
        top_k: Number of files to return (1-100, default: 10)

    Returns:
        List of relevant code snippets with filename, content, and score
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

        results = get_searcher().search(
            repo_name=index_result.repo_name,
            query=query.strip(),
            top_k=top_k,
        )

        logger.info(f"Search returned {len(results)} results")

        if not results:
            return [{"message": "No matching code found", "query": query.strip()}]

        return [r.to_dict() for r in results]

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

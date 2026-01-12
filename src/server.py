"""FastMCP server for semantic code search.

Provides a single tool for searching codebases with automatic indexing.
"""

import logging
import os
import signal
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastmcp import FastMCP
from config.settings import settings

from src.storage.postgres import close_pool, init_db
from src.exceptions import IndexingError, PathError, SearchError
from src.indexer.service import get_indexer
from src.retrieval.service import get_searcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP(
    "cocode",
    instructions="""Semantic code search for your codebase.

Call codebase_retrieval with your question - indexing happens automatically.
Works with any programming language. Uses hybrid vector + keyword search.""",
)


@mcp.tool()
async def codebase_retrieval(
    query: str,
    path: str | None = None,
    top_k: int = 10,
) -> list[dict]:
    """Search a codebase using semantic and keyword search.

    Indexing happens automatically on first use and updates incrementally
    on subsequent calls to catch any file changes.

    Args:
        query: Natural language question about the code
               Examples: "how does authentication work",
                        "where is the database connection handled",
                        "error handling in API routes"
        path: Path to the codebase (defaults to current working directory)
        top_k: Number of files to return (default: 10)
               - Use 3-5 for focused queries targeting specific functions/classes
               - Use 10-15 for understanding a feature or subsystem
               - Use 20-30 for broad exploration or finding all related code
               Internally searches 3x this number and aggregates by file.

    Returns:
        List of relevant code snippets with filename, content, and score
    """
    # Validate inputs
    if not query or not query.strip():
        return [{"error": "Query cannot be empty"}]

    # Default to current directory
    if path is None:
        path = os.getcwd()

    try:
        # Ensure codebase is indexed
        indexer = get_indexer()
        index_result = indexer.ensure_indexed(path)

        logger.info(
            f"Index status: {index_result.status} "
            f"({index_result.file_count} files, {index_result.chunk_count} chunks)"
        )

        # Check if indexing produced any chunks
        if index_result.chunk_count == 0:
            return [{"error": f"No code files found to index in {path}"}]

        # Search
        searcher = get_searcher()
        results = searcher.search(
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
        return [{"error": str(e)}]

    except IndexingError as e:
        logger.error(f"Indexing error: {e}")
        return [{"error": f"Failed to index codebase: {e}"}]

    except SearchError as e:
        logger.error(f"Search error: {e}")
        return [{"error": f"Search failed: {e}"}]

    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return [{"error": f"Unexpected error: {e}"}]


@mcp.tool()
async def clear_index(path: str | None = None) -> dict:
    """Clear the index for a codebase.

    Use this to delete the existing index and force a fresh re-index
    on the next search. Useful when changing indexing settings.

    Args:
        path: Path to the codebase (defaults to current working directory)

    Returns:
        Status message confirming the index was cleared
    """
    if path is None:
        path = os.getcwd()

    try:
        indexer = get_indexer()
        resolved_path = Path(path).resolve()

        if not resolved_path.exists() or not resolved_path.is_dir():
            return {"error": f"Path does not exist or is not a directory: {path}"}

        repo_name = indexer.path_to_repo_name(str(resolved_path))
        indexer._delete_index(repo_name)

        return {
            "status": "cleared",
            "path": str(resolved_path),
            "message": f"Index cleared. Next search will rebuild the index.",
        }

    except Exception as e:
        logger.error(f"Failed to clear index: {e}")
        return {"error": f"Failed to clear index: {e}"}


def main():
    """Main entry point for MCP server."""
    logger.info("Starting cocode MCP server...")

    if not settings.openai_api_key:
        logger.error(
            "OPENAI_API_KEY is required. "
            "Set it in your environment or .env file."
        )
        sys.exit(1)

    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        sys.exit(1)

    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        close_pool()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f))

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

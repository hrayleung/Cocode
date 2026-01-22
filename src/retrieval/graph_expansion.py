"""Graph-based expansion to find related files.

Parses imports/dependencies from code files using Tree-sitter AST parsing
and uses them to expand search results with contextually related files.
"""

import logging
import json
from dataclasses import dataclass
from pathlib import Path

from psycopg import sql

from src.storage.postgres import get_connection
from src.parser.ast_parser import extract_imports_ast, get_language_from_file
from src.storage.schema import sanitize_repo_name

logger = logging.getLogger(__name__)


@dataclass
class FileRelation:
    """Represents a relationship between two files."""
    source_file: str
    target_file: str
    relation_type: str  # 'imports', 'imported_by'
    hop_distance: int = 1  # How many hops away from the original file


def extract_imports(content: str, language: str) -> list[str]:
    """Extract import statements from code content using Tree-sitter AST parsing.

    Args:
        content: Source code content
        language: Programming language

    Returns:
        List of imported module/file paths
    """
    imports = extract_imports_ast(content, language)
    if imports:
        logger.debug(f"Extracted {len(imports)} imports using AST parsing for {language}")
    return imports


def resolve_import_to_file(
    import_path: str,
    source_file: str,
    repo_files: set[str],
    language: str,
) -> str | None:
    """Resolve an import path to an actual file in the repository.

    Args:
        import_path: The import path from the source code
        source_file: The file containing the import
        repo_files: Set of all files in the repository
        language: Programming language

    Returns:
        Resolved file path or None if not found
    """
    source_dir = str(Path(source_file).parent)

    if language == "python":
        # Convert dot notation to path
        module_path = import_path.replace(".", "/")
        candidates = [
            f"{module_path}.py",
            f"{module_path}/__init__.py",
            f"{source_dir}/{module_path}.py",
            f"{source_dir}/{module_path}/__init__.py",
        ]
    elif language in ("typescript", "javascript"):
        # Handle relative and absolute imports
        if import_path.startswith("."):
            # Relative import
            base = str(Path(source_dir) / import_path)
            candidates = [
                f"{base}.ts",
                f"{base}.tsx",
                f"{base}.js",
                f"{base}.jsx",
                f"{base}/index.ts",
                f"{base}/index.tsx",
                f"{base}/index.js",
            ]
        else:
            # Package import - skip node_modules
            return None
    elif language == "go":
        # Go imports are package paths - harder to resolve without go.mod
        return None
    elif language == "rust":
        # Rust uses crate/module system
        module_path = import_path.replace("::", "/")
        candidates = [
            f"src/{module_path}.rs",
            f"src/{module_path}/mod.rs",
        ]
    else:
        return None

    # Find first matching file
    for candidate in candidates:
        # Normalize path
        normalized = str(Path(candidate).as_posix())
        if normalized in repo_files:
            return normalized
        # Try without leading ./
        if normalized.startswith("./"):
            normalized = normalized[2:]
        if normalized in repo_files:
            return normalized

    return None


def get_repo_files(repo_name: str) -> set[str]:
    """Get all indexed files for a repository."""
    from .vector_search import get_chunks_table_name
    table_name = get_chunks_table_name(repo_name)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SELECT DISTINCT filename FROM {}").format(sql.Identifier(table_name))
            )
            return {row[0] for row in cur.fetchall()}


def build_import_graph(repo_name: str, use_cache: bool = True) -> dict[str, list[str]]:
    """Build a graph of imports for a repository.

    Args:
        repo_name: Repository name
        use_cache: Whether to use cached graph if available (default: True)

    Returns:
        Import graph mapping {filename: [imported_files]}
    """
    # Try to load from cache first
    if use_cache:
        from .graph_cache import get_cached_import_graph

        cached = get_cached_import_graph(repo_name)
        if cached:
            import_graph, _ = cached  # We only need the forward graph here
            logger.debug(f"Using cached import graph with {len(import_graph)} entries")
            return import_graph

    # Cache miss or disabled - build from scratch
    logger.debug(f"Building import graph from scratch for {repo_name}")

    from .vector_search import get_chunks_table_name
    table_name = get_chunks_table_name(repo_name)

    with get_connection() as conn:
        with conn.cursor(name="graph_expansion_cursor") as cur:
            cur.itersize = 100
            # First pass: collect all filenames
            cur.execute(
                sql.SQL("SELECT DISTINCT filename FROM {}").format(sql.Identifier(table_name))
            )
            repo_files = {row[0] for row in cur.fetchall()}

        # Second pass: extract imports with full repo_files set available
        import_graph: dict[str, list[str]] = {}
        with conn.cursor(name="graph_content_cursor") as cur:
            cur.itersize = 100
            cur.execute(
                sql.SQL("SELECT filename, content FROM {}").format(sql.Identifier(table_name))
            )

            # Track which files we've already processed (multiple chunks per file)
            processed_files: dict[str, list[str]] = {}

            for filename, content in cur:
                language = get_language_from_file(filename)

                if not language or not content:
                    continue

                imports = extract_imports(content, language)

                resolved = []
                for imp in imports:
                    resolved_file = resolve_import_to_file(imp, filename, repo_files, language)
                    if resolved_file and resolved_file != filename:
                        resolved.append(resolved_file)

                # Accumulate imports from all chunks of the same file
                if filename not in processed_files:
                    processed_files[filename] = []
                processed_files[filename].extend(resolved)

            # Deduplicate imports per file
            for filename, imports in processed_files.items():
                unique_imports = list(set(imports))
                if unique_imports:
                    import_graph[filename] = unique_imports

    # Populate cache for next time
    if use_cache:
        from .graph_cache import create_graph_cache_table, populate_graph_cache

        try:
            create_graph_cache_table(repo_name)
            populate_graph_cache(repo_name, import_graph)
        except Exception as e:
            logger.warning(f"Failed to populate graph cache: {e}")

    return import_graph


def multi_hop_traversal(
    start_files: list[str],
    import_graph: dict[str, list[str]],
    reverse_graph: dict[str, list[str]],
    max_hops: int = 3,
    max_results: int = 30,
) -> list[tuple[str, str, str, int]]:
    """Perform multi-hop BFS traversal using Rust.

    reverse_graph is accepted for API compatibility but is not used.

    Returns:
        List of tuples: (source_file, target_file, relation_type, hop_distance)
    """

    from src.rust_bridge import bfs_traversal_edges as rust_bfs_edges

    edges: list[tuple[str, str]] = []
    for source, targets in import_graph.items():
        for target in targets:
            edges.append((source, target))

    return rust_bfs_edges(
        edges,
        start_files,
        max_hops=max_hops,
        max_results=max_results,
        bidirectional=True,
    )


def _get_related_files_from_graph_cache(
    repo_name: str,
    filenames: list[str],
    *,
    max_related: int,
    max_hops: int,
) -> list[FileRelation] | None:
    """Fetch related files via the per-repo graph_cache table.

    Returns None when the cache table isn't available or inputs are invalid,
    so callers can fall back to rebuilding the graph from indexed chunks.
    """
    if not filenames or max_related <= 0 or max_hops <= 0:
        return None

    schema_name = sanitize_repo_name(repo_name)
    cache_table = sql.Identifier(schema_name, "graph_cache")

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_schema = %s AND table_name = 'graph_cache'
                    )
                    """,
                    (schema_name,),
                )
                if not cur.fetchone()[0]:
                    return None

                visited = set(filenames)
                frontier = list(filenames)
                relations: list[FileRelation] = []

                for distance in range(max_hops):
                    if not frontier or len(relations) >= max_related:
                        break

                    cur.execute(
                        sql.SQL(
                            """
                            SELECT filename, imports, imported_by
                            FROM {table}
                            WHERE filename = ANY(%s)
                            """
                        ).format(table=cache_table),
                        (frontier,),
                    )
                    rows = cur.fetchall()

                    neighbors: dict[str, tuple[list[str], list[str]]] = {}
                    for filename, imports, imported_by in rows:
                        if isinstance(imports, str):
                            try:
                                imports = json.loads(imports)
                            except Exception:
                                imports = []
                        if isinstance(imported_by, str):
                            try:
                                imported_by = json.loads(imported_by)
                            except Exception:
                                imported_by = []

                        imp_list = imports if isinstance(imports, list) else []
                        by_list = imported_by if isinstance(imported_by, list) else []
                        neighbors[filename] = (imp_list, by_list)

                    next_frontier: list[str] = []
                    hop_distance = distance + 1

                    for source_file in frontier:
                        imports, imported_by = neighbors.get(source_file, ([], []))

                        for target_file in imports:
                            if target_file == source_file or target_file in visited:
                                continue
                            visited.add(target_file)
                            relations.append(
                                FileRelation(
                                    source_file=source_file,
                                    target_file=target_file,
                                    relation_type="imports",
                                    hop_distance=hop_distance,
                                )
                            )
                            if len(relations) >= max_related:
                                break
                            next_frontier.append(target_file)

                        if len(relations) >= max_related:
                            break

                        for target_file in imported_by:
                            if target_file == source_file or target_file in visited:
                                continue
                            visited.add(target_file)
                            relations.append(
                                FileRelation(
                                    source_file=source_file,
                                    target_file=target_file,
                                    relation_type="imported_by",
                                    hop_distance=hop_distance,
                                )
                            )
                            if len(relations) >= max_related:
                                break
                            next_frontier.append(target_file)

                        if len(relations) >= max_related:
                            break

                    frontier = next_frontier

                return relations[:max_related]

    except Exception as e:
        logger.debug(f"Graph cache traversal failed for {repo_name}: {e}")
        return None


def get_related_files(
    repo_name: str,
    filenames: list[str],
    max_related: int = None,
    max_hops: int = None,
) -> list[FileRelation]:
    """Get files related to the given files through imports using multi-hop BFS.

    Args:
        repo_name: Repository name
        filenames: List of files to find relations for
        max_related: Maximum number of related files to return (default: from settings)
        max_hops: Maximum hop distance to traverse (default: from settings)

    Returns:
        List of file relations with hop distance tracking
    """
    from config.settings import settings

    # Use configured values if not explicitly provided
    if max_hops is None:
        max_hops = settings.max_graph_hops
    if max_related is None:
        max_related = settings.max_graph_results

    cached = _get_related_files_from_graph_cache(
        repo_name,
        filenames,
        max_related=max_related,
        max_hops=max_hops,
    )
    if cached is not None:
        return cached

    try:
        import_graph = build_import_graph(repo_name)
    except Exception as e:
        logger.warning(f"Failed to build import graph: {e}")
        return []

    # Perform multi-hop BFS traversal
    traversal_results = multi_hop_traversal(
        start_files=filenames,
        import_graph=import_graph,
        reverse_graph={},
        max_hops=max_hops,
        max_results=max_related,
    )

    # Convert traversal results to FileRelation objects
    relations = []
    for source_file, target_file, relation_type, hop_distance in traversal_results:
        relations.append(FileRelation(
            source_file=source_file,
            target_file=target_file,
            relation_type=relation_type,
            hop_distance=hop_distance,
        ))

    logger.debug(f"Found {len(relations)} related files across {max_hops} hops")
    return relations


def expand_results_with_related(
    repo_name: str,
    result_filenames: list[str],
    max_expansion: int = 3,
) -> list[str]:
    """Expand search results with related files.

    Args:
        repo_name: Repository name
        result_filenames: Original result filenames
        max_expansion: Maximum number of related files to add

    Returns:
        List of additional related filenames
    """
    from src.retrieval.file_categorizer import categorize_file

    relations = get_related_files(repo_name, result_filenames, max_related=max_expansion)
    result_set = set(result_filenames)
    related_files = []

    for rel in relations:
        candidate = rel.target_file
        if candidate not in result_set and categorize_file(candidate) == "implementation":
            related_files.append(candidate)

    return related_files[:max_expansion]

"""Graph-based expansion to find related files.

Parses imports/dependencies from code files and uses them to
expand search results with contextually related files.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from src.storage.postgres import get_connection

logger = logging.getLogger(__name__)


@dataclass
class FileRelation:
    """Represents a relationship between two files."""
    source_file: str
    target_file: str
    relation_type: str  # 'imports', 'imported_by'


# Import patterns for different languages
IMPORT_PATTERNS = {
    "python": [
        re.compile(r'^from\s+([\w.]+)\s+import', re.MULTILINE),
        re.compile(r'^import\s+([\w.]+)', re.MULTILINE),
        # Handle imports inside try blocks (common for optional dependencies)
        re.compile(r'^\s+from\s+([\w.]+)\s+import', re.MULTILINE),
        re.compile(r'^\s+import\s+([\w.]+)', re.MULTILINE),
    ],
    "typescript": [
        re.compile(r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]", re.MULTILINE),
        re.compile(r"import\s+['\"]([^'\"]+)['\"]", re.MULTILINE),
        re.compile(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", re.MULTILINE),
        re.compile(r"import\s+type\s+\{[^}]+\}\s+from\s+['\"]([^'\"]+)['\"]", re.MULTILINE),
        re.compile(r"export\s+\*\s+from\s+['\"]([^'\"]+)['\"]", re.MULTILINE),
    ],
    "javascript": [
        re.compile(r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]", re.MULTILINE),
        re.compile(r"import\s+['\"]([^'\"]+)['\"]", re.MULTILINE),
        re.compile(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)", re.MULTILINE),
    ],
    "go": [
        re.compile(r'import\s+["\']([^"\']+)["\']', re.MULTILINE),
        re.compile(r'import\s*\(\s*[^)]*?["\']([^"\']+)["\']', re.MULTILINE | re.DOTALL),
    ],
    "rust": [
        re.compile(r'^use\s+([\w:]+)', re.MULTILINE),
        re.compile(r'^mod\s+(\w+)', re.MULTILINE),
        re.compile(r'^extern\s+crate\s+\w+', re.MULTILINE),
    ],
}

# Map file extensions to language
EXT_TO_LANG = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
}


def extract_imports(content: str, language: str) -> list[str]:
    """Extract import statements from code content.

    Args:
        content: Source code content
        language: Programming language

    Returns:
        List of imported module/file paths
    """
    patterns = IMPORT_PATTERNS.get(language, [])
    imports = []

    for pattern in patterns:
        matches = pattern.findall(content)
        imports.extend(matches)

    return list(set(imports))


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
            cur.execute(f"""
                SELECT DISTINCT filename
                FROM {table_name}
            """)
            return {row[0] for row in cur.fetchall()}


def build_import_graph(repo_name: str) -> dict[str, list[str]]:
    """Build a graph of imports for a repository."""
    from .vector_search import get_chunks_table_name
    table_name = get_chunks_table_name(repo_name)

    with get_connection() as conn:
        with conn.cursor(name="graph_expansion_cursor") as cur:
            cur.itersize = 100
            # First pass: collect all filenames
            cur.execute(f"""
                SELECT DISTINCT filename
                FROM {table_name}
            """)
            repo_files = {row[0] for row in cur.fetchall()}

        # Second pass: extract imports with full repo_files set available
        import_graph: dict[str, list[str]] = {}
        with conn.cursor(name="graph_content_cursor") as cur:
            cur.itersize = 100
            cur.execute(f"""
                SELECT filename, content
                FROM {table_name}
            """)

            # Track which files we've already processed (multiple chunks per file)
            processed_files: dict[str, list[str]] = {}

            for filename, content in cur:
                ext = Path(filename).suffix.lower()
                language = EXT_TO_LANG.get(ext)

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

    return import_graph


def get_related_files(
    repo_name: str,
    filenames: list[str],
    max_related: int = 5,
) -> list[FileRelation]:
    """Get files related to the given files through imports.

    Args:
        repo_name: Repository name
        filenames: List of files to find relations for
        max_related: Maximum number of related files to return

    Returns:
        List of file relations
    """
    try:
        import_graph = build_import_graph(repo_name)
    except Exception as e:
        logger.warning(f"Failed to build import graph: {e}")
        return []

    # Build reverse graph (imported_by)
    reverse_graph: dict[str, list[str]] = {}
    for source, targets in import_graph.items():
        for target in targets:
            if target not in reverse_graph:
                reverse_graph[target] = []
            reverse_graph[target].append(source)

    relations = []
    seen_files = set(filenames)

    for filename in filenames:
        # Files this file imports
        for imported in import_graph.get(filename, []):
            if imported not in seen_files:
                relations.append(FileRelation(
                    source_file=filename,
                    target_file=imported,
                    relation_type="imports",
                ))
                seen_files.add(imported)

        # Files that import this file
        for importer in reverse_graph.get(filename, []):
            if importer not in seen_files:
                relations.append(FileRelation(
                    source_file=importer,
                    target_file=filename,
                    relation_type="imported_by",
                ))
                seen_files.add(importer)

        if len(relations) >= max_related:
            break

    return relations[:max_related]


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
        candidate = rel.target_file if rel.relation_type == "imports" else rel.source_file
        if candidate not in result_set and categorize_file(candidate) == "implementation":
            related_files.append(candidate)

    return related_files[:max_expansion]

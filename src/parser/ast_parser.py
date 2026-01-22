"""AST-based code parsing delegated to the Rust extension.

This module intentionally has no Python fallback implementation. If the Rust
extension (cocode_rust) is missing or errors, callers should treat it as a hard
failure.
"""

from pathlib import Path

from src.rust_bridge import (
    extract_imports_ast as _rust_extract_imports_ast,
    is_language_supported as _rust_is_language_supported,
)


def is_language_supported(language: str) -> bool:
    """Return True if Rust has a parser for the given language."""
    return bool(_rust_is_language_supported(language))


def extract_imports_ast(content: str, language: str) -> list[str]:
    """Extract imports using Rust Tree-sitter parsing."""
    return list(_rust_extract_imports_ast(content, language))


# Extension to language mapping
EXT_TO_AST_LANG = {
    ".py": "python",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".hh": "cpp",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
}


def get_language_from_file(filename: str) -> str | None:
    """Get language name from filename extension."""
    return EXT_TO_AST_LANG.get(Path(filename).suffix.lower())

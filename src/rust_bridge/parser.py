"""Python wrapper for Rust Tree-sitter parsing functions."""

from cocode_rust import (
    is_language_supported as _rust_is_language_supported,
    extract_imports_ast as _rust_extract_imports_ast,
    extract_symbols as _rust_extract_symbols,
    extract_relationships as _rust_extract_relationships,
    extract_calls as _rust_extract_calls,
)


def is_language_supported(language: str) -> bool:
    """Return True if Rust has a parser for the given language."""
    return bool(_rust_is_language_supported(language))


def extract_imports_ast(content: str, language: str) -> list[str]:
    """Extract import/module references using Rust Tree-sitter parsing."""
    return list(_rust_extract_imports_ast(content, language))


def extract_symbols(content: str, language: str, filename: str) -> list[tuple]:
    """Extract symbols using Rust Tree-sitter parsing.

    Returns tuples compatible with src.parser.symbol_extractor.Symbol fields.
    """
    return list(_rust_extract_symbols(content, language, filename))


def extract_relationships(content: str, language: str) -> list[tuple]:
    """Extract inheritance/implementation relationships."""
    return list(_rust_extract_relationships(content, language))


def extract_calls(
    content: str,
    language: str,
    current_function_name: str | None = None,
) -> list[tuple]:
    """Extract function calls from code."""
    return list(_rust_extract_calls(content, language, current_function_name))

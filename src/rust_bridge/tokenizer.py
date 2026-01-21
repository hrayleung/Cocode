"""Python wrapper for Rust tokenizer."""

from cocode_rust import (
    extract_code_tokens as _rust_extract,
    tokenize_for_search as _rust_tokenize,
    batch_extract_tokens as _rust_batch_extract,
    batch_tokenize_queries as _rust_batch_tokenize,
)


def extract_code_tokens(text: str) -> list[str]:
    """Extract code tokens (handles camelCase, snake_case, etc.)."""
    return _rust_extract(text)


def tokenize_for_search(query: str) -> list[str]:
    """Tokenize a search query."""
    return _rust_tokenize(query)


def batch_extract_tokens(texts: list[str]) -> list[list[str]]:
    """Extract tokens from multiple texts in parallel."""
    return _rust_batch_extract(texts)


def batch_tokenize_queries(queries: list[str]) -> list[list[str]]:
    """Tokenize multiple queries in parallel."""
    return _rust_batch_tokenize(queries)

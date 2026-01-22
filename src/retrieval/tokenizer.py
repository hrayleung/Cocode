"""Code-aware tokenization for BM25 search.

Uses Rust for performance.
"""

import re

from src.rust_bridge import (
    extract_code_tokens,
    tokenize_for_search,
)

__all__ = ["tokenize_for_search", "extract_code_tokens", "build_tsquery", "normalize_content_for_fts"]


def split_camel_case(text: str) -> list[str]:
    """Split camelCase/PascalCase."""
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', text)
    text = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', text)
    return text.split('_')


def split_snake_case(text: str) -> list[str]:
    """Split snake_case."""
    return [p for p in text.split('_') if p]


def split_kebab_case(text: str) -> list[str]:
    """Split kebab-case."""
    return [p for p in text.split('-') if p]


def build_tsquery(tokens: list[str], mode: str = "and") -> str:
    """Build a PostgreSQL tsquery string from tokens."""
    if not tokens:
        return ""

    operator = " & " if mode == "and" else " | "
    escaped = []
    for t in tokens:
        safe = t.replace("'", "''").replace("\\", "\\\\")
        escaped.append(f"{safe}:*")
    return operator.join(escaped)


def normalize_content_for_fts(content: str) -> str:
    """Normalize code content for full-text search indexing."""
    tokens = extract_code_tokens(content)
    return content + " " + " ".join(tokens)

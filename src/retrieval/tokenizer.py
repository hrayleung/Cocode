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
    """
    Split a CamelCase or PascalCase identifier into its component words.
    
    Returns:
        components (list[str]): The identifier's components in order. Empty strings may appear if the input contains underscores.
    """
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', text)
    text = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', text)
    return text.split('_')


def split_snake_case(text: str) -> list[str]:
    """
    Split a snake_case string into its non-empty components.
    
    Parameters:
        text (str): The snake_case input string.
    
    Returns:
        list[str]: Segments between underscores, excluding any empty segments.
    """
    return [p for p in text.split('_') if p]


def split_kebab_case(text: str) -> list[str]:
    """
    Split a kebab-case string into its non-empty components.
    
    Parameters:
        text (str): Input string potentially containing kebab-case segments separated by hyphens.
    
    Returns:
        list[str]: List of segments from `text` with empty segments removed.
    """
    return [p for p in text.split('-') if p]


def build_tsquery(tokens: list[str], mode: str = "and") -> str:
    """
    Build a PostgreSQL tsquery expression from a list of token strings.
    
    Parameters:
    	tokens (list[str]): Tokens to include in the tsquery. If empty, returns an empty string.
    	mode (str): "and" to join terms with `&`, any other value to join with `|`.
    
    Returns:
    	tsquery (str): A tsquery string where each token is escaped (single quotes doubled, backslashes escaped), suffixed with `:*`, and joined by the selected operator.
    """
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
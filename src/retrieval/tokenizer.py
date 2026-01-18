"""Code-aware tokenization for BM25 search.

Handles code-specific patterns like camelCase, snake_case, SCREAMING_CASE,
and other programming language conventions.
"""

import re

# Programming language keywords and common terms that should be excluded
CODE_STOP_WORDS: set[str] = {
    # Common programming keywords
    "def", "class", "function", "const", "let", "var", "if", "else", "elif",
    "for", "while", "do", "switch", "case", "break", "continue", "return",
    "try", "catch", "except", "finally", "throw", "raise", "import", "from",
    "export", "default", "async", "await", "yield", "lambda", "new", "this",
    "self", "super", "extends", "implements", "interface", "type", "enum",
    "struct", "trait", "impl", "pub", "private", "public", "protected",
    "static", "final", "abstract", "virtual", "override", "const", "mut",
    "fn", "mod", "use", "crate", "extern", "unsafe", "where", "dyn", "ref",
    "match", "loop", "in", "not", "and", "or", "is", "as", "with", "pass",
    "none", "null", "nil", "true", "false", "void", "int", "str", "bool",
    "float", "double", "char", "byte", "long", "short", "unsigned", "signed",
    # Common single letters and articles
    "a", "an", "the", "of", "to", "in", "on", "at", "by", "it", "is", "be",
    # Common method/variable prefixes that are too generic
    "get", "set", "has", "can", "should", "will", "did", "was", "are",
}

# Minimum token length to consider
MIN_TOKEN_LENGTH = 2


def split_camel_case(text: str) -> list[str]:
    """Split camelCase and PascalCase into separate words.

    Examples:
        "getUserName" -> ["get", "User", "Name"]
        "XMLParser" -> ["XML", "Parser"]
        "getHTTPResponse" -> ["get", "HTTP", "Response"]
    """
    # Handle acronyms followed by lowercase (e.g., XMLParser -> XML, Parser)
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', text)
    # Handle lowercase followed by uppercase (e.g., getUser -> get, User)
    text = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', text)
    return text.split('_')


def split_snake_case(text: str) -> list[str]:
    """Split snake_case and SCREAMING_SNAKE_CASE into words."""
    return [part for part in text.split('_') if part]


def split_kebab_case(text: str) -> list[str]:
    """Split kebab-case into words."""
    return [part for part in text.split('-') if part]


def extract_code_tokens(text: str) -> list[str]:
    """Extract meaningful tokens from code text.

    This handles:
    - camelCase, PascalCase
    - snake_case, SCREAMING_SNAKE_CASE
    - kebab-case
    - Number-suffixed identifiers (e.g., model_v2, configV3)
    - Common code patterns

    Args:
        text: Source code or code-related text

    Returns:
        List of normalized tokens
    """
    # First, extract potential identifiers (alphanumeric with underscores/dashes)
    identifier_pattern = r'[a-zA-Z_][a-zA-Z0-9_-]*'
    identifiers = re.findall(identifier_pattern, text)

    tokens = []

    for identifier in identifiers:
        # Skip very short identifiers
        if len(identifier) < MIN_TOKEN_LENGTH:
            continue

        # Split by multiple conventions
        parts = []

        # First split by underscores and dashes
        for segment in re.split(r'[_-]+', identifier):
            if not segment:
                continue
            # Then split camelCase within each segment
            camel_parts = split_camel_case(segment)
            parts.extend(camel_parts)

        # Normalize and filter
        for part in parts:
            normalized = part.lower()
            # Skip stop words and very short tokens
            if len(normalized) >= MIN_TOKEN_LENGTH and normalized not in CODE_STOP_WORDS:
                tokens.append(normalized)

    return tokens


def tokenize_for_search(query: str) -> list[str]:
    """Tokenize a search query for BM25 search.

    More lenient than code extraction - keeps some common terms
    that might be meaningful in search context.

    Args:
        query: User's search query

    Returns:
        List of search tokens
    """
    # Extract code-style tokens
    code_tokens = extract_code_tokens(query)

    # Also extract quoted phrases as single tokens
    quoted = re.findall(r'"([^"]+)"', query)

    # Add word tokens for natural language queries
    words = re.findall(r'\b[a-zA-Z]{2,}\b', query.lower())

    # Combine, deduplicate, and filter
    all_tokens = set(code_tokens)
    all_tokens.update(word for word in words if word not in CODE_STOP_WORDS)

    return list(all_tokens)


def build_tsquery(tokens: list[str], mode: str = "and") -> str:
    """Build a PostgreSQL tsquery string from tokens.

    Args:
        tokens: List of search tokens
        mode: "and" for all tokens required, "or" for any token

    Returns:
        PostgreSQL tsquery string
    """
    if not tokens:
        return ""

    operator = " & " if mode == "and" else " | "

    # Escape special characters and handle prefix matching
    escaped_tokens = []
    for token in tokens:
        # Add prefix matching for partial word search
        escaped = token.replace("'", "''").replace("\\", "\\\\")
        escaped_tokens.append(f"{escaped}:*")

    return operator.join(escaped_tokens)


def normalize_content_for_fts(content: str) -> str:
    """Normalize code content for full-text search indexing.

    Expands identifiers so they can be found by individual words.
    For example: "getUserById" -> "getUserById get user by id"

    Args:
        content: Source code content

    Returns:
        Normalized text suitable for FTS indexing
    """
    tokens = extract_code_tokens(content)

    # Create expanded version with original content + extracted tokens
    # This allows searching for either "getUserById" or "user" to match
    expanded = content + " " + " ".join(tokens)

    return expanded

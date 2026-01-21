"""Code-aware tokenization for BM25 search.

Uses Rust for performance when available, falls back to Python.
"""

import re

# Try to use Rust tokenizer
try:
    from src.rust_bridge.tokenizer import (
        extract_code_tokens as _rust_extract,
        tokenize_for_search as _rust_tokenize,
    )
    _USE_RUST = True
except ImportError:
    _USE_RUST = False

# Python fallback implementation
CODE_STOP_WORDS: set[str] = {
    # Language keywords (control flow)
    "if", "else", "elif", "then", "endif", "switch", "case", "default",
    "for", "while", "do", "loop", "break", "continue", "goto", "return",
    "try", "catch", "except", "finally", "throw", "raise", "throws",
    "yield", "await", "async",
    # Declarations
    "def", "func", "fn", "function", "proc", "sub", "method",
    "class", "struct", "enum", "interface", "trait", "type", "typedef",
    "var", "let", "const", "val", "mut", "static", "final", "readonly",
    "import", "from", "export", "require", "include", "using", "use",
    "package", "module", "mod", "namespace", "crate", "extern",
    # OOP keywords
    "new", "this", "self", "super", "extends", "implements", "override",
    "public", "private", "protected", "internal", "abstract", "virtual",
    "sealed", "partial",
    # Type keywords
    "void", "null", "nil", "none", "undefined", "true", "false",
    "int", "integer", "float", "double", "bool", "boolean", "string", "str",
    "char", "byte", "short", "long", "unsigned", "signed", "size",
    "object", "any", "dynamic", "auto",
    # Rust specific
    "impl", "pub", "where", "dyn", "ref", "box", "move", "unsafe", "match",
    # Common short identifiers
    "i", "j", "k", "n", "m", "x", "y", "z", "a", "b", "c", "e", "f", "p", "q", "r", "s", "t", "v", "w",
    "id", "ok", "err", "io", "os", "fs", "db", "ui", "tx", "rx",
    # Common generic variable names
    "tmp", "temp", "arg", "args", "param", "params",
    "ret", "res", "result", "out", "output", "in", "input",
    "buf", "buffer", "ptr", "ctx", "cfg", "opt", "opts",
    "idx", "len", "num", "cnt", "pos", "key",
    # Common method prefixes
    "get", "set", "has", "is", "can", "should", "will", "did", "was", "are",
    "add", "remove", "delete", "update", "create", "init", "load", "save",
    "read", "write", "open", "close", "start", "stop", "run", "exec",
    "on", "to", "with", "by", "at", "of", "as", "or", "and", "not",
    # English articles
    "an", "the", "be", "it", "its",
    # Common but meaningless alone
    "data", "info", "item", "items", "list", "array", "map",
    "node", "elem", "element", "entry", "record", "row", "col",
    "src", "dst", "source", "dest", "target", "origin",
    "old", "prev", "next", "cur", "current", "last", "first",
    "min", "max", "sum", "avg", "total", "count",
    "name", "value", "kind", "mode", "state", "status",
    "msg", "message", "text", "content", "body", "payload",
    "path", "file", "dir", "url", "uri",
    "cb", "callback", "handler", "listener", "observer",
}

MIN_TOKEN_LENGTH = 2


def _split_camel_case_py(text: str) -> list[str]:
    """Split camelCase/PascalCase (Python fallback)."""
    text = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', text)
    text = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', text)
    return text.split('_')


def _extract_code_tokens_py(text: str) -> list[str]:
    """Extract code tokens (Python fallback)."""
    identifiers = re.findall(r'[a-zA-Z_][a-zA-Z0-9_-]*', text)
    tokens = []

    for identifier in identifiers:
        if len(identifier) < MIN_TOKEN_LENGTH:
            continue

        for segment in re.split(r'[_-]+', identifier):
            if not segment:
                continue
            for part in _split_camel_case_py(segment):
                normalized = part.lower()
                if len(normalized) >= MIN_TOKEN_LENGTH and normalized not in CODE_STOP_WORDS:
                    tokens.append(normalized)

    return tokens


def _tokenize_for_search_py(query: str) -> list[str]:
    """Tokenize search query (Python fallback)."""
    code_tokens = _extract_code_tokens_py(query)
    words = re.findall(r'\b[a-zA-Z]{2,}\b', query.lower())
    
    all_tokens = set(code_tokens)
    all_tokens.update(word for word in words if word not in CODE_STOP_WORDS)
    return list(all_tokens)


# Public API - uses Rust when available
def extract_code_tokens(text: str) -> list[str]:
    """Extract meaningful tokens from code text."""
    if _USE_RUST:
        return _rust_extract(text)
    return _extract_code_tokens_py(text)


def tokenize_for_search(query: str) -> list[str]:
    """Tokenize a search query for BM25 search."""
    if _USE_RUST:
        return _rust_tokenize(query)
    return _tokenize_for_search_py(query)


# Keep these for compatibility
split_camel_case = _split_camel_case_py
split_snake_case = lambda text: [p for p in text.split('_') if p]
split_kebab_case = lambda text: [p for p in text.split('-') if p]


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

"""Parser module for AST-based code analysis."""

from .ast_parser import extract_imports_ast, is_language_supported, get_language_from_file
from .symbol_extractor import extract_symbols, Symbol
from .call_extractor import extract_calls, extract_calls_from_function, FunctionCall

__all__ = [
    "extract_imports_ast",
    "is_language_supported",
    "get_language_from_file",
    "extract_symbols",
    "Symbol",
    "extract_calls",
    "extract_calls_from_function",
    "FunctionCall",
]

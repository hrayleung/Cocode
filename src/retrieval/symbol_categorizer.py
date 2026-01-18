"""Enhanced symbol-level categorization for improved ranking.

This module extends file categorization to individual symbols,
allowing more nuanced ranking (e.g., a test file might contain
utility functions that should rank higher).
"""

import logging
import re
from typing import Optional

from src.parser.symbol_extractor import Symbol
from src.parser.ast_parser import get_language_from_file

logger = logging.getLogger(__name__)


# Symbol category weights (applied on top of file category weights)
SYMBOL_CATEGORY_WEIGHTS = {
    "api": 1.2,          # Public API exports
    "implementation": 1.0,  # Regular implementation
    "internal": 0.8,     # Internal/private helpers
    "test": 0.3,         # Test functions
}


def categorize_symbol(symbol: Symbol, filename: str) -> str:
    """Categorize a symbol for ranking purposes.

    Categories:
    - 'api': Public API (exported, public visibility)
    - 'implementation': Regular implementation code
    - 'internal': Private/internal helpers
    - 'test': Test functions/classes

    Args:
        symbol: Symbol to categorize
        filename: Filename containing the symbol

    Returns:
        Category string
    """
    # Already categorized during extraction?
    if symbol.category and symbol.category != "implementation":
        return symbol.category

    # Test detection (highest priority)
    if is_test_symbol(symbol, filename):
        return "test"

    # API detection
    if is_api_symbol(symbol):
        return "api"

    # Internal/private detection
    if is_internal_symbol(symbol):
        return "internal"

    # Default to implementation
    return "implementation"


def is_test_symbol(symbol: Symbol, filename: str) -> bool:
    """Check if symbol is test-related.

    Args:
        symbol: Symbol to check
        filename: Filename

    Returns:
        True if test symbol
    """
    # Already categorized as test?
    if symbol.category == "test":
        return True

    # Test function names
    if symbol.symbol_name.startswith("test_") or symbol.symbol_name.startswith("Test"):
        return True

    # Test class names
    if symbol.symbol_type == "class" and symbol.symbol_name.endswith("Test"):
        return True

    # In test directory or file
    if any(part in filename for part in ["test/", "tests/", "__tests__/", "spec/"]):
        return True

    if filename.endswith("_test.py") or filename.endswith("_test.go") or filename.endswith("_test.rs"):
        return True

    return False


def is_api_symbol(symbol: Symbol) -> bool:
    """Check if symbol is part of public API.

    Args:
        symbol: Symbol to check

    Returns:
        True if API symbol
    """
    # Explicitly marked as API
    if symbol.category == "api":
        return True

    # Public visibility and not in internal location
    if symbol.visibility == "public":
        # Top-level functions/classes in main modules often constitute API
        if symbol.parent_symbol is None:  # Not a method
            return True

    return False


def is_internal_symbol(symbol: Symbol) -> bool:
    """Check if symbol is internal/private.

    Args:
        symbol: Symbol to check

    Returns:
        True if internal symbol
    """
    # Private visibility
    if symbol.visibility in ("private", "internal"):
        return True

    # Python convention: leading underscore
    if symbol.symbol_name.startswith("_") and not symbol.symbol_name.startswith("__"):
        return True

    # Double underscore (name mangling in Python)
    if symbol.symbol_name.startswith("__") and not symbol.symbol_name.endswith("__"):
        return True

    return False


def get_symbol_category_weight(category: str) -> float:
    """Get ranking weight for a symbol category.

    Args:
        category: Symbol category

    Returns:
        Weight multiplier
    """
    return SYMBOL_CATEGORY_WEIGHTS.get(category, 1.0)


def apply_symbol_category_boost(symbols: list, symbol_categories: dict[str, str]) -> None:
    """Apply category-based boosting to symbol search results.

    Modifies scores in-place.

    Args:
        symbols: List of symbol search results (must have .symbol_id and .score)
        symbol_categories: Mapping {symbol_id: category}
    """
    for symbol in symbols:
        category = symbol_categories.get(symbol.symbol_id, "implementation")
        weight = get_symbol_category_weight(category)
        symbol.score *= weight


def detect_exported_symbols(code: str, language: str) -> set[str]:
    """Detect explicitly exported symbols (for API detection).

    This is a heuristic check for export statements.

    Args:
        code: Source code
        language: Programming language

    Returns:
        Set of exported symbol names
    """
    exports = set()

    if language == "python":
        # Check for __all__ definition
        if "__all__" in code:
            match = re.search(r'__all__\s*=\s*\[(.*?)\]', code, re.DOTALL)
            if match:
                items = match.group(1)
                names = re.findall(r'["\']([^"\']+)["\']', items)
                exports.update(names)

    elif language in ("typescript", "javascript"):
        # export function foo(), export class Bar, export const baz
        pattern = r'export\s+(?:function|class|const|let|var)\s+(\w+)'
        exports.update(re.findall(pattern, code))

        # export { foo, bar }
        for match in re.findall(r'export\s+\{([^}]+)\}', code):
            names = [name.strip().split()[0] for name in match.split(',')]
            exports.update(names)

    elif language == "rust":
        pattern = r'pub\s+(?:fn|struct|enum|trait)\s+(\w+)'
        exports.update(re.findall(pattern, code))

    return exports


def enhance_symbol_with_category(symbol: Symbol, filename: str, code: Optional[str] = None) -> Symbol:
    """Enhance a symbol with improved category detection.

    Args:
        symbol: Symbol to enhance
        filename: Filename
        code: Optional source code (for export detection)

    Returns:
        Enhanced symbol (modified in-place)
    """
    if code:
        language = get_language_from_file(filename)
        if language:
            exports = detect_exported_symbols(code, language)
            if symbol.symbol_name in exports:
                symbol.category = "api"
                return symbol

    symbol.category = categorize_symbol(symbol, filename)
    return symbol

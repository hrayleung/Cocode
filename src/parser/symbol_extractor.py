"""Symbol extraction via the Rust Tree-sitter parser.

This module intentionally stays thin: it only defines Python dataclasses and
converts Rust results into those dataclasses. Parsing must be provided by the
cocode_rust extension.
"""

import logging
from dataclasses import dataclass

from src.rust_bridge import (
    extract_relationships as _rust_extract_relationships,
    extract_symbols as _rust_extract_symbols,
)

logger = logging.getLogger(__name__)


@dataclass
class Symbol:
    """Represents a code symbol (function, class, method, etc.)."""

    symbol_name: str
    symbol_type: str
    line_start: int
    line_end: int
    signature: str
    docstring: str | None = None
    parent_symbol: str | None = None
    visibility: str = "public"
    category: str = "implementation"


@dataclass
class SymbolRelationship:
    """Represents a relationship between symbols."""

    source_name: str
    target_name: str
    relationship_type: str
    source_line: int
    confidence: float = 1.0


def extract_symbols(content: str, language: str, filename: str) -> list[Symbol]:
    """Extract symbols using Rust Tree-sitter parsing.

    This function intentionally does not fall back to Python.
    """

    raw = _rust_extract_symbols(content, language, filename)
    out: list[Symbol] = []

    for (
        symbol_name,
        symbol_type,
        line_start,
        line_end,
        signature,
        docstring,
        parent_symbol,
        visibility,
        category,
    ) in raw:
        out.append(
            Symbol(
                symbol_name=symbol_name,
                symbol_type=symbol_type,
                line_start=int(line_start),
                line_end=int(line_end),
                signature=signature,
                docstring=docstring,
                parent_symbol=parent_symbol,
                visibility=visibility,
                category=category,
            )
        )

    return out


def extract_relationships(content: str, language: str) -> list[SymbolRelationship]:
    """Extract relationships using Rust Tree-sitter parsing.

    This function intentionally does not fall back to Python.
    """

    raw = _rust_extract_relationships(content, language)
    out: list[SymbolRelationship] = []

    for source_name, target_name, relationship_type, source_line, confidence in raw:
        out.append(
            SymbolRelationship(
                source_name=source_name,
                target_name=target_name,
                relationship_type=relationship_type,
                source_line=int(source_line),
                confidence=float(confidence),
            )
        )

    return out

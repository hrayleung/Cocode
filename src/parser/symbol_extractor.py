"""Symbol extraction from source code using AST parsing.

This module extracts function, class, and method definitions from code files
using Tree-sitter for accurate AST parsing. Extracted symbols include:
- Name, type, and signature
- Line numbers (1-indexed, inclusive)
- Docstrings and parent relationships
- Visibility and category classification
"""

import logging
from dataclasses import dataclass
from typing import Optional

try:
    from tree_sitter import Node
    from src.parser.ast_parser import get_parser, TREE_SITTER_AVAILABLE
    HAS_TREE_SITTER = TREE_SITTER_AVAILABLE
except ImportError:
    HAS_TREE_SITTER = False

logger = logging.getLogger(__name__)


@dataclass
class Symbol:
    """Represents a code symbol (function, class, method, etc.).

    Attributes:
        symbol_name: Name of the symbol
        symbol_type: Type: 'function', 'class', 'method', or 'interface'
        line_start: Starting line number (1-indexed)
        line_end: Ending line number (1-indexed, inclusive)
        signature: Full signature line (e.g., "def foo(x: int) -> str:")
        docstring: Extracted docstring if present
        parent_symbol: Parent class name for methods
        visibility: 'public', 'private', or 'internal'
        category: 'implementation', 'test', 'api', or 'config'
    """

    symbol_name: str
    symbol_type: str
    line_start: int
    line_end: int
    signature: str
    docstring: Optional[str] = None
    parent_symbol: Optional[str] = None
    visibility: str = "public"
    category: str = "implementation"


def extract_python_symbols(tree_root: Node, source_bytes: bytes, filename: str) -> list[Symbol]:
    """Extract symbols from Python AST."""
    symbols = []

    def get_docstring(node: Node) -> Optional[str]:
        body_node = node.child_by_field_name("body")
        if not body_node:
            return None
        for child in body_node.children:
            if child.type == "expression_statement":
                expr = child.child_by_field_name("expression") or (child.children[0] if child.children else None)
                if expr and expr.type == "string":
                    docstring = source_bytes[expr.start_byte:expr.end_byte].decode("utf-8")
                    for quote in ['"""', "'''", '"', "'"]:
                        if docstring.startswith(quote) and docstring.endswith(quote):
                            return docstring[len(quote):-len(quote)].strip()
                    return docstring.strip()
        return None

    def get_signature(node: Node) -> str:
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8").split("\n")[0].strip()

    def get_visibility(name: str, decorators: list[str]) -> str:
        if name.startswith("__") and not name.endswith("__"):
            return "private"
        if name.startswith("_"):
            return "internal"
        if "@private" in decorators or "private" in " ".join(decorators).lower():
            return "private"
        return "public"

    def get_decorators(node: Node) -> list[str]:
        decorators = []
        parent = node.parent
        if parent and parent.type == "decorated_definition":
            for child in parent.children:
                if child.type == "decorator":
                    decorators.append(source_bytes[child.start_byte:child.end_byte].decode("utf-8"))
        return decorators

    def visit_node(node: Node, parent_class: Optional[str] = None):
        if node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")
                decorators = get_decorators(node)
                symbol_type = "method" if parent_class else "function"
                is_test = name.startswith("test_") or "@pytest" in " ".join(decorators)

                symbols.append(Symbol(
                    symbol_name=name,
                    symbol_type=symbol_type,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=get_signature(node),
                    docstring=get_docstring(node),
                    parent_symbol=parent_class,
                    visibility=get_visibility(name, decorators),
                    category="test" if is_test else "implementation",
                ))

        elif node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")
                decorators = get_decorators(node)
                is_test = name.startswith("Test") or "@pytest" in " ".join(decorators)

                symbols.append(Symbol(
                    symbol_name=name,
                    symbol_type="class",
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=get_signature(node),
                    docstring=get_docstring(node),
                    parent_symbol=None,
                    visibility=get_visibility(name, decorators),
                    category="test" if is_test else "implementation",
                ))

                body_node = node.child_by_field_name("body")
                if body_node:
                    for child in body_node.children:
                        visit_node(child, parent_class=name)
                return

        for child in node.children:
            visit_node(child, parent_class)

    visit_node(tree_root)
    return symbols


def extract_go_symbols(tree_root: Node, source_bytes: bytes, filename: str) -> list[Symbol]:
    """Extract symbols from Go AST."""
    symbols = []

    def get_signature(node: Node) -> str:
        return source_bytes[node.start_byte:node.end_byte].decode("utf-8").split("\n")[0].strip()

    def get_comment_above(node: Node) -> Optional[str]:
        return None  # Simplified - full implementation would track comments

    def visit_node(node: Node):
        if node.type == "function_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")
                is_test = name.startswith("Test") or name.startswith("Benchmark")

                symbols.append(Symbol(
                    symbol_name=name,
                    symbol_type="function",
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=get_signature(node),
                    docstring=get_comment_above(node),
                    visibility="public" if name[0].isupper() else "internal",
                    category="test" if is_test else "implementation",
                ))

        elif node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
            receiver_node = node.child_by_field_name("receiver")

            if name_node:
                name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")
                parent_type = None

                if receiver_node:
                    import re
                    receiver_text = source_bytes[receiver_node.start_byte:receiver_node.end_byte].decode("utf-8")
                    match = re.search(r'\*?(\w+)\s*\)', receiver_text)
                    if match:
                        parent_type = match.group(1)

                is_test = name.startswith("Test")

                symbols.append(Symbol(
                    symbol_name=name,
                    symbol_type="method",
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=get_signature(node),
                    docstring=get_comment_above(node),
                    parent_symbol=parent_type,
                    visibility="public" if name[0].isupper() else "internal",
                    category="test" if is_test else "implementation",
                ))

        elif node.type == "type_declaration":
            for spec in node.children:
                if spec.type == "type_spec":
                    name_node = spec.child_by_field_name("name")
                    type_node = spec.child_by_field_name("type")

                    if name_node and type_node:
                        name = source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8")
                        type_kind = "interface" if type_node.type == "interface_type" else "class"

                        symbols.append(Symbol(
                            symbol_name=name,
                            symbol_type=type_kind,
                            line_start=spec.start_point[0] + 1,
                            line_end=spec.end_point[0] + 1,
                            signature=source_bytes[spec.start_byte:spec.end_byte].decode("utf-8").split("\n")[0],
                            docstring=get_comment_above(spec),
                            visibility="public" if name[0].isupper() else "internal",
                            category="implementation",
                        ))

        for child in node.children:
            visit_node(child)

    visit_node(tree_root)
    return symbols


def extract_symbols(content: str, language: str, filename: str) -> list[Symbol]:
    """Extract symbols from source code using AST parsing."""
    if not HAS_TREE_SITTER:
        logger.debug("Tree-sitter not available")
        return []

    parser = get_parser(language)
    if not parser:
        logger.debug(f"No parser for language: {language}")
        return []

    try:
        source_bytes = content.encode("utf-8")
        tree = parser.parse(source_bytes)

        if not tree or not tree.root_node:
            logger.warning(f"Failed to parse {language} code")
            return []

        if language == "python":
            return extract_python_symbols(tree.root_node, source_bytes, filename)
        if language == "go":
            return extract_go_symbols(tree.root_node, source_bytes, filename)

        logger.debug(f"Symbol extraction not implemented for {language}")
        return []

    except Exception as e:
        logger.error(f"Error extracting symbols from {language}: {e}")
        return []

"""Symbol extraction from source code using Tree-sitter AST parsing.

Uses a generic, language-agnostic approach based on tree-sitter node types.
Each language defines which node types represent definitions (functions, classes, etc.)
in a declarative config, avoiding per-language extraction functions.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Node
from src.parser.ast_parser import get_parser

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


# Language-agnostic symbol extraction config
SYMBOL_NODE_TYPES: dict[str, dict[str, tuple[str, str]]] = {
    "python": {
        "function_definition": ("function", "name"),
        "class_definition": ("class", "name"),
    },
    "go": {
        "function_declaration": ("function", "name"),
        "method_declaration": ("method", "name"),
        "type_declaration": ("class", None),
    },
    "rust": {
        "function_item": ("function", "name"),
        "struct_item": ("class", "name"),
        "enum_item": ("class", "name"),
        "trait_item": ("interface", "name"),
        "impl_item": ("impl", "type"),
    },
    "typescript": {
        "function_declaration": ("function", "name"),
        "class_declaration": ("class", "name"),
        "method_definition": ("method", "name"),
        "interface_declaration": ("interface", "name"),
        "type_alias_declaration": ("type", "name"),
        "enum_declaration": ("enum", "name"),
        "lexical_declaration": ("variable", None),
    },
    "javascript": {
        "function_declaration": ("function", "name"),
        "class_declaration": ("class", "name"),
        "method_definition": ("method", "name"),
        "lexical_declaration": ("variable", None),
    },
    "tsx": {
        "function_declaration": ("function", "name"),
        "class_declaration": ("class", "name"),
        "method_definition": ("method", "name"),
        "interface_declaration": ("interface", "name"),
        "type_alias_declaration": ("type", "name"),
        "lexical_declaration": ("variable", None),
    },
    "c": {
        "function_definition": ("function", "declarator"),
        "struct_specifier": ("class", "name"),
    },
    "cpp": {
        "function_definition": ("function", "declarator"),
        "class_specifier": ("class", "name"),
        "struct_specifier": ("class", "name"),
    },
}

PARENT_CONTEXT_NODES: dict[str, set[str]] = {
    "python": {"class_definition"},
    "go": {"type_declaration"},
    "rust": {"impl_item"},
    "typescript": {"class_declaration", "class_body"},
    "javascript": {"class_declaration", "class_body"},
    "tsx": {"class_declaration", "class_body"},
    "c": set(),
    "cpp": {"class_specifier"},
}

INHERITANCE_NODE_TYPES: dict[str, dict[str, str]] = {
    "python": {"argument_list": "extends"},
    "typescript": {"extends_clause": "extends", "implements_clause": "implements"},
    "javascript": {"extends_clause": "extends"},
    "tsx": {"extends_clause": "extends", "implements_clause": "implements"},
    "go": {},
    "rust": {},
}


class _SymbolVisitor:
    """AST visitor for extracting symbols from source code."""

    def __init__(self, source_bytes: bytes, language: str, filename: str):
        self.source_bytes = source_bytes
        self.language = language
        self.filename = filename
        self.node_types = SYMBOL_NODE_TYPES.get(language, {})
        self.parent_nodes = PARENT_CONTEXT_NODES.get(language, set())
        self.is_test_file = _is_test_file(filename)
        self.symbols: list[Symbol] = []

    def get_text(self, node: Node) -> str:
        return self.source_bytes[node.start_byte:node.end_byte].decode("utf-8")

    def get_name(self, node: Node, name_field: str | None) -> str | None:
        if name_field is None:
            return None
        name_node = node.child_by_field_name(name_field)
        if not name_node:
            return None
        
        # For C/C++ declarators, try to find identifier child node first
        if name_field == "declarator":
            return self._extract_declarator_name(name_node)
        
        text = self.get_text(name_node)
        if "(" in text:
            parts = text.split("(")[0].strip().split()
            if parts:
                return parts[-1]
            return None
        return text

    def _extract_declarator_name(self, node: Node) -> str | None:
        """Extract function name from C/C++ declarator, handling function pointers."""
        # Try to find identifier in declarator tree
        def find_identifier(n: Node) -> str | None:
            if n.type == "identifier":
                return self.get_text(n)
            for child in n.children:
                result = find_identifier(child)
                if result:
                    return result
            return None
        
        name = find_identifier(node)
        if name:
            return name
        
        # Fallback to text parsing
        text = self.get_text(node)
        if "(" in text:
            parts = text.split("(")[0].strip().split()
            if parts:
                # Handle (*foo) pattern
                last = parts[-1].lstrip("*")
                return last if last else None
        return text if text else None

    def get_signature(self, node: Node) -> str:
        text = self.get_text(node)
        first_line = text.split("\n")[0].strip()
        if self.language == "python" and ":" in first_line:
            idx = first_line.rfind(":")
            first_line = first_line[:idx + 1]
        elif "{" in first_line:
            first_line = first_line.split("{")[0].strip()
        return first_line[:200]

    def get_docstring(self, node: Node) -> str | None:
        """Extract docstring/doc comment above or inside node."""
        comments = []
        prev = node.prev_sibling
        while prev:
            if prev.type in ("comment", "line_comment", "block_comment"):
                text = self.get_text(prev).strip()
                if text.startswith("///") or text.startswith("//!"):
                    comments.insert(0, text.lstrip("/!").strip())
                elif text.startswith("#") and not text.startswith("#!"):
                    comments.insert(0, text.lstrip("#").strip())
                elif text.startswith("/*") or text.startswith("/**"):
                    comments.insert(0, text.strip("/* \n"))
                prev = prev.prev_sibling
            else:
                break
        if comments:
            return "\n".join(comments)

        if self.language == "python":
            body = node.child_by_field_name("body")
            if body and body.children:
                first = body.children[0]
                if first.type == "expression_statement" and first.children:
                    expr = first.children[0]
                    if expr.type == "string":
                        return self.get_text(expr).strip("\"'")
        return None

    def get_visibility(self, node: Node, name: str) -> str:
        """Determine symbol visibility."""
        for child in node.children:
            if child.type == "visibility_modifier":
                return "public"
            if child.type in ("private", "protected", "public"):
                return self.get_text(child)

        if self.language == "python":
            if name.startswith("__") and not name.endswith("__"):
                return "private"
            if name.startswith("_"):
                return "internal"
        elif self.language == "go":
            if name and name[0].islower():
                return "internal"
        elif self.language == "rust":
            for child in node.children:
                if child.type == "visibility_modifier":
                    return "public"
            return "private"

        return "public"

    def _handle_go_type_declaration(self, node: Node) -> tuple[str | None, str]:
        """Handle Go type_declaration to extract name and determine if interface."""
        symbol_type = "class"
        name = None
        for child in node.children:
            if child.type == "type_spec":
                name_node = child.child_by_field_name("name")
                if name_node:
                    name = self.get_text(name_node)
                    type_node = child.child_by_field_name("type")
                    if type_node and type_node.type == "interface_type":
                        symbol_type = "interface"
                break
        return name, symbol_type

    def _handle_lexical_declaration(self, node: Node) -> list[tuple[str, str]]:
        """Handle JS/TS lexical_declaration for React components/hooks.
        
        Returns list of (name, symbol_type) tuples for all declarators.
        """
        results: list[tuple[str, str]] = []
        for child in node.children:
            if child.type != "variable_declarator":
                continue
            name_node = child.child_by_field_name("name")
            value_node = child.child_by_field_name("value")
            if not name_node:
                continue

            name = self.get_text(name_node)
            if name.startswith("[") or name.startswith("{"):
                continue  # Skip destructuring

            if value_node and value_node.type in ("arrow_function", "function"):
                # React 19 `use` hook or useXxx pattern
                if name == "use" or (name.startswith("use") and len(name) > 3 and name[3].isupper()):
                    results.append((name, "hook"))
                elif name[0].isupper():
                    results.append((name, "component"))
                else:
                    results.append((name, "function"))
        return results

    def visit(self, node: Node, parent_name: str | None = None) -> None:
        node_type = node.type

        if node_type in self.node_types:
            symbol_type, name_field = self.node_types[node_type]

            # Handle impl blocks (Rust)
            if symbol_type == "impl":
                impl_type = self.get_name(node, name_field)
                for child in node.children:
                    self.visit(child, parent_name=impl_type)
                return

            name = self.get_name(node, name_field)

            # Special handling for Go type_declaration
            if node_type == "type_declaration" and not name:
                name, symbol_type = self._handle_go_type_declaration(node)

            # Special handling for JS/TS lexical_declaration - returns list of declarators
            if node_type == "lexical_declaration" and not name:
                declarators = self._handle_lexical_declaration(node)
                for decl_name, decl_type in declarators:
                    is_test = self.is_test_file or _is_test_symbol(decl_name, self.language)
                    self.symbols.append(Symbol(
                        symbol_name=decl_name,
                        symbol_type=decl_type,
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        signature=self.get_signature(node),
                        docstring=self.get_docstring(node),
                        parent_symbol=parent_name,
                        visibility=self.get_visibility(node, decl_name),
                        category="test" if is_test else "implementation",
                    ))
                return  # Already handled all declarators

            if name:
                actual_type = "method" if parent_name and symbol_type == "function" else symbol_type
                is_test = self.is_test_file or _is_test_symbol(name, self.language)

                self.symbols.append(Symbol(
                    symbol_name=name,
                    symbol_type=actual_type,
                    line_start=node.start_point[0] + 1,
                    line_end=node.end_point[0] + 1,
                    signature=self.get_signature(node),
                    docstring=self.get_docstring(node),
                    parent_symbol=parent_name,
                    visibility=self.get_visibility(node, name),
                    category="test" if is_test else "implementation",
                ))

                # Visit children with this as parent for classes
                if symbol_type == "class" and node_type in self.parent_nodes:
                    for child in node.children:
                        self.visit(child, parent_name=name)
                    return

        # Check if this provides parent context
        if node_type in self.parent_nodes:
            ctx_name = self.get_name(node, self.node_types.get(node_type, (None, None))[1])
            for child in node.children:
                self.visit(child, parent_name=ctx_name or parent_name)
            return

        # Recurse
        for child in node.children:
            self.visit(child, parent_name)


def _extract_symbols_python(content: str, language: str, filename: str) -> list[Symbol]:
    """Extract symbols using Python Tree-sitter parsing (fallback)."""
    parser = get_parser(language)
    if not parser:
        logger.debug(f"No parser for language: {language}")
        return []

    if language not in SYMBOL_NODE_TYPES:
        logger.debug(f"No symbol config for language: {language}")
        return []

    try:
        source_bytes = content.encode("utf-8")
        tree = parser.parse(source_bytes)
        if not tree or not tree.root_node:
            return []

        visitor = _SymbolVisitor(source_bytes, language, filename)
        visitor.visit(tree.root_node)
        return visitor.symbols

    except Exception:
        logger.exception("Error extracting symbols from %s", language)
        return []


def extract_symbols(content: str, language: str, filename: str) -> list[Symbol]:
    """Extract symbols using Rust Tree-sitter parsing.

    Falls back to the Python Tree-sitter implementation when Rust parsing fails.
    """
    try:
        from src.rust_bridge import extract_symbols as rust_extract_symbols

        raw = rust_extract_symbols(content, language, filename)
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
    except Exception as e:
        logger.debug(f"Rust symbol extraction failed, falling back to Python: {e}")
        return _extract_symbols_python(content, language, filename)


def _is_test_file(filename: str) -> bool:
    """Check if file is a test file based on naming conventions."""
    name = Path(filename).stem.lower()
    return any(p in name for p in ("test_", "_test", "tests", "spec")) or "/test" in filename.lower()


def _is_test_symbol(name: str, language: str) -> bool:
    """Check if symbol name indicates a test."""
    name_lower = name.lower()
    if language == "python":
        return name_lower.startswith("test")
    if language == "go":
        return name.startswith("Test") or name.startswith("Benchmark")
    if language in ("typescript", "javascript"):
        return name_lower in ("it", "describe", "test") or name_lower.startswith("test")
    return name_lower.startswith("test")


class _RelationshipVisitor:
    """AST visitor for extracting inheritance relationships."""

    def __init__(self, source_bytes: bytes, language: str):
        self.source_bytes = source_bytes
        self.language = language
        self.inheritance_types = INHERITANCE_NODE_TYPES.get(language, {})
        self.relationships: list[SymbolRelationship] = []

    def get_text(self, node: Node) -> str:
        return self.source_bytes[node.start_byte:node.end_byte].decode("utf-8")

    def visit(self, node: Node, class_name: str | None = None) -> None:
        # Track current class context
        if node.type in ("class_definition", "class_declaration"):
            name_node = node.child_by_field_name("name")
            if name_node:
                class_name = self.get_text(name_node)

                # Python: check argument_list for base classes
                if self.language == "python":
                    self._extract_python_bases(node, class_name)
                    return

        # TypeScript/JavaScript: extends_clause, implements_clause
        if node.type in self.inheritance_types and class_name:
            rel_type = self.inheritance_types[node.type]
            for child in node.children:
                if child.type in ("type_identifier", "identifier"):
                    self.relationships.append(SymbolRelationship(
                        source_name=class_name,
                        target_name=self.get_text(child),
                        relationship_type=rel_type,
                        source_line=node.start_point[0] + 1,
                    ))

        for child in node.children:
            self.visit(child, class_name)

    def _extract_python_bases(self, node: Node, class_name: str) -> None:
        """Extract Python base classes from argument_list."""
        for child in node.children:
            if child.type == "argument_list":
                for arg in child.children:
                    if arg.type == "identifier":
                        base = self.get_text(arg)
                        if base not in ("object", "ABC"):
                            self.relationships.append(SymbolRelationship(
                                source_name=class_name,
                                target_name=base,
                                relationship_type="extends",
                                source_line=node.start_point[0] + 1,
                            ))
            else:
                self.visit(child, class_name)


def _extract_relationships_python(content: str, language: str) -> list[SymbolRelationship]:
    """Extract relationships using Python Tree-sitter parsing (fallback)."""
    parser = get_parser(language)
    if not parser:
        return []

    inheritance_types = INHERITANCE_NODE_TYPES.get(language, {})
    if not inheritance_types and language not in ("go", "rust"):
        return []

    try:
        source_bytes = content.encode("utf-8")
        tree = parser.parse(source_bytes)
        if not tree or not tree.root_node:
            return []

        visitor = _RelationshipVisitor(source_bytes, language)
        visitor.visit(tree.root_node)
        return visitor.relationships

    except Exception:
        logger.exception("Error extracting relationships from %s", language)
        return []


def extract_relationships(content: str, language: str) -> list[SymbolRelationship]:
    """Extract relationships using Rust Tree-sitter parsing.

    Falls back to the Python Tree-sitter implementation when Rust parsing fails.
    """
    try:
        from src.rust_bridge import extract_relationships as rust_extract_relationships

        raw = rust_extract_relationships(content, language)
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
    except Exception as e:
        logger.debug(f"Rust relationship extraction failed, falling back to Python: {e}")
        return _extract_relationships_python(content, language)

"""AST-based code parsing using Tree-sitter.

This module provides language-aware parsing for import extraction using
Tree-sitter grammars. It supports Python, Go, Rust, C/C++, JavaScript,
TypeScript, and TSX.

Falls back gracefully when Tree-sitter is unavailable.
"""

import logging
from pathlib import Path
from typing import Optional

try:
    from tree_sitter import Language, Parser, Node
    import tree_sitter_python as ts_python
    import tree_sitter_go as ts_go
    import tree_sitter_rust as ts_rust
    import tree_sitter_c as ts_c
    import tree_sitter_cpp as ts_cpp
    import tree_sitter_javascript as ts_javascript
    import tree_sitter_typescript as ts_typescript
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

logger = logging.getLogger(__name__)


class LanguageParsers:
    """Singleton manager for Tree-sitter language parsers.

    Parsers are lazily initialized on first use and reused across calls.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized or not TREE_SITTER_AVAILABLE:
            return

        self.languages = {}
        self.parsers = {}
        try:
            self.languages = {
                "python": Language(ts_python.language()),
                "go": Language(ts_go.language()),
                "rust": Language(ts_rust.language()),
                "c": Language(ts_c.language()),
                "cpp": Language(ts_cpp.language()),
                "javascript": Language(ts_javascript.language()),
                "typescript": Language(ts_typescript.language_typescript()),
                "tsx": Language(ts_typescript.language_tsx()),
            }
            self.parsers = {
                name: Parser(lang)
                for name, lang in self.languages.items()
            }
            self._initialized = True
            logger.info(f"Initialized Tree-sitter parsers for {len(self.parsers)} languages")
        except Exception as e:
            logger.error(f"Failed to initialize Tree-sitter parsers: {e}")
            self._initialized = False


def get_parser(language: str) -> Optional[Parser]:
    """Get Tree-sitter parser for a language.

    Args:
        language: Language name (e.g., "python", "go")

    Returns:
        Parser instance or None if unavailable
    """
    if not TREE_SITTER_AVAILABLE:
        return None
    return LanguageParsers().parsers.get(language)


def is_language_supported(language: str) -> bool:
    """Check if a language is supported for AST parsing.

    Args:
        language: Language name to check

    Returns:
        True if the language can be parsed
    """
    return TREE_SITTER_AVAILABLE and language in LanguageParsers().parsers


def extract_python_imports(tree_root: Node, source_bytes: bytes) -> list[str]:
    """Extract import statements from Python AST."""
    imports = []

    def visit_node(node: Node):
        if node.type == "import_statement":
            for child in node.children:
                if child.type == "dotted_name":
                    imports.append(source_bytes[child.start_byte:child.end_byte].decode("utf-8"))
                elif child.type == "aliased_import":
                    dotted = child.child_by_field_name("name")
                    if dotted:
                        imports.append(source_bytes[dotted.start_byte:dotted.end_byte].decode("utf-8"))
        elif node.type == "import_from_statement":
            module_name_node = node.child_by_field_name("module_name")
            if module_name_node:
                imports.append(source_bytes[module_name_node.start_byte:module_name_node.end_byte].decode("utf-8"))

        for child in node.children:
            visit_node(child)

    visit_node(tree_root)
    return list(set(imports))


def extract_go_imports(tree_root: Node, source_bytes: bytes) -> list[str]:
    """Extract import statements from Go AST."""
    imports = []

    def visit_node(node: Node):
        if node.type == "import_spec":
            for child in node.children:
                if child.type == "interpreted_string_literal":
                    import_path = source_bytes[child.start_byte:child.end_byte].decode("utf-8").strip('"')
                    imports.append(import_path)

        for child in node.children:
            visit_node(child)

    visit_node(tree_root)
    return list(set(imports))


def extract_rust_imports(tree_root: Node, source_bytes: bytes) -> list[str]:
    """Extract import statements from Rust AST."""
    imports = []

    def visit_node(node: Node):
        if node.type == "use_declaration":
            for child in node.children:
                if child.type in ("scoped_identifier", "identifier"):
                    path_text = source_bytes[child.start_byte:child.end_byte].decode("utf-8")
                    base_path = path_text.split("::")[0].strip()
                    if base_path and not base_path.startswith("crate"):
                        imports.append(base_path)
                    break
                elif child.type == "scoped_use_list":
                    for subchild in child.children:
                        if subchild.type in ("scoped_identifier", "identifier"):
                            path_text = source_bytes[subchild.start_byte:subchild.end_byte].decode("utf-8")
                            base_path = path_text.split("::")[0].strip()
                            if base_path and not base_path.startswith("crate"):
                                imports.append(base_path)
                            break
                    break
                elif child.type == "use_as_clause":
                    for subchild in child.children:
                        if subchild.type == "scoped_identifier":
                            path_text = source_bytes[subchild.start_byte:subchild.end_byte].decode("utf-8")
                            base_path = path_text.split("::")[0].strip()
                            if base_path and not base_path.startswith("crate"):
                                imports.append(base_path)
                            break
                    break
        elif node.type == "mod_item":
            name_node = node.child_by_field_name("name")
            if name_node:
                imports.append(source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8"))
        elif node.type == "extern_crate_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                imports.append(source_bytes[name_node.start_byte:name_node.end_byte].decode("utf-8"))

        for child in node.children:
            visit_node(child)

    visit_node(tree_root)
    return list(set(imports))


def extract_c_cpp_includes(tree_root: Node, source_bytes: bytes) -> list[str]:
    """Extract #include directives from C/C++ AST."""
    imports = []

    def visit_node(node: Node):
        if node.type == "preproc_include":
            for child in node.children:
                if child.type in ("system_lib_string", "string_literal"):
                    header = source_bytes[child.start_byte:child.end_byte].decode("utf-8")
                    imports.append(header.strip('<>').strip('"'))

        for child in node.children:
            visit_node(child)

    visit_node(tree_root)
    return list(set(imports))


def extract_javascript_imports(tree_root: Node, source_bytes: bytes) -> list[str]:
    """Extract import statements from JavaScript/TypeScript AST."""
    imports = []

    def visit_node(node: Node):
        if node.type == "import_statement":
            source_node = node.child_by_field_name("source")
            if source_node and source_node.type == "string":
                import_path = source_bytes[source_node.start_byte:source_node.end_byte].decode("utf-8")
                imports.append(import_path.strip("'\""))
        elif node.type == "call_expression":
            func_node = node.child_by_field_name("function")
            if func_node and source_bytes[func_node.start_byte:func_node.end_byte].decode("utf-8") == "require":
                args_node = node.child_by_field_name("arguments")
                if args_node:
                    for child in args_node.children:
                        if child.type == "string":
                            import_path = source_bytes[child.start_byte:child.end_byte].decode("utf-8")
                            imports.append(import_path.strip("'\""))
        elif node.type == "export_statement":
            source_node = node.child_by_field_name("source")
            if source_node and source_node.type == "string":
                import_path = source_bytes[source_node.start_byte:source_node.end_byte].decode("utf-8")
                imports.append(import_path.strip("'\""))

        for child in node.children:
            visit_node(child)

    visit_node(tree_root)
    return list(set(imports))


def extract_imports_ast(content: str, language: str) -> list[str]:
    """Extract imports using AST parsing."""
    if not TREE_SITTER_AVAILABLE:
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

        extractors = {
            "python": extract_python_imports,
            "go": extract_go_imports,
            "rust": extract_rust_imports,
            "c": extract_c_cpp_includes,
            "cpp": extract_c_cpp_includes,
            "javascript": extract_javascript_imports,
            "typescript": extract_javascript_imports,
            "tsx": extract_javascript_imports,
        }

        extractor = extractors.get(language)
        if extractor:
            return extractor(tree.root_node, source_bytes)

        logger.warning(f"No import extractor for: {language}")
        return []

    except Exception as e:
        logger.error(f"Error extracting imports from {language}: {e}")
        return []


# Extension to language mapping
EXT_TO_AST_LANG = {
    ".py": "python",
    ".go": "go",
    ".rs": "rust",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cxx": "cpp", ".cc": "cpp", ".hpp": "cpp", ".hxx": "cpp", ".hh": "cpp",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".mts": "typescript", ".cts": "typescript",
    ".tsx": "tsx",
}


def get_language_from_file(filename: str) -> Optional[str]:
    """Get language name from filename extension."""
    return EXT_TO_AST_LANG.get(Path(filename).suffix.lower())

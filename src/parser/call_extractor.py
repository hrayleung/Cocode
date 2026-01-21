"""Extract function calls from code using AST parsing."""

import logging
from dataclasses import dataclass
from typing import Optional

from tree_sitter import Node

from .ast_parser import get_parser

logger = logging.getLogger(__name__)


@dataclass
class FunctionCall:
    """Represents a function call found in code."""
    function_name: str
    line_number: int
    call_type: str  # 'function_call', 'method_call', 'constructor'
    context: Optional[str] = None  # 'loop', 'conditional', 'try_block', etc.
    object_name: Optional[str] = None  # For method calls: object.method()
    is_recursive: bool = False


def get_call_context(node: Node, source_bytes: bytes) -> Optional[str]:
    """Determine the context where a call occurs."""
    parent = node.parent
    contexts = []

    while parent:
        if parent.type in ('for_statement', 'while_statement', 'for_in_clause'):
            contexts.append('loop')
        elif parent.type in ('if_statement', 'elif_clause', 'else_clause', 'conditional_expression'):
            contexts.append('conditional')
        elif parent.type in ('try_statement', 'except_clause'):
            contexts.append('try_block')
        elif parent.type == 'with_statement':
            contexts.append('with_block')
        elif parent.type == 'lambda':
            contexts.append('lambda')
        parent = parent.parent

    return ', '.join(contexts) if contexts else None


def extract_python_calls(tree_root: Node, source_bytes: bytes, current_function_name: Optional[str] = None) -> list[FunctionCall]:
    """Extract function calls from Python AST."""
    calls = []

    def traverse(node: Node):
        if node.type == 'call':
            function_node = node.child_by_field_name('function')
            if function_node:
                function_name = None
                object_name = None
                call_type = 'function_call'

                if function_node.type == 'identifier':
                    function_name = source_bytes[function_node.start_byte:function_node.end_byte].decode('utf-8')
                elif function_node.type == 'attribute':
                    object_node = function_node.child_by_field_name('object')
                    attr_node = function_node.child_by_field_name('attribute')

                    if attr_node:
                        function_name = source_bytes[attr_node.start_byte:attr_node.end_byte].decode('utf-8')
                        call_type = 'method_call'
                        if object_node:
                            if object_node.type == 'identifier':
                                object_name = source_bytes[object_node.start_byte:object_node.end_byte].decode('utf-8')
                            elif object_node.type == 'call':
                                object_name = '[chained]'

                if function_name:
                    calls.append(FunctionCall(
                        function_name=function_name,
                        line_number=node.start_point[0] + 1,
                        call_type=call_type,
                        context=get_call_context(node, source_bytes),
                        object_name=object_name,
                        is_recursive=(current_function_name == function_name),
                    ))

        for child in node.children:
            traverse(child)

    traverse(tree_root)
    return calls


def extract_go_calls(tree_root: Node, source_bytes: bytes, current_function_name: Optional[str] = None) -> list[FunctionCall]:
    """Extract function calls from Go AST."""
    calls = []

    def traverse(node: Node):
        if node.type == 'call_expression':
            function_node = node.child_by_field_name('function')
            if function_node:
                function_name = None
                object_name = None
                call_type = 'function_call'

                if function_node.type == 'identifier':
                    function_name = source_bytes[function_node.start_byte:function_node.end_byte].decode('utf-8')
                elif function_node.type == 'selector_expression':
                    operand = function_node.child_by_field_name('operand')
                    field = function_node.child_by_field_name('field')

                    if field:
                        function_name = source_bytes[field.start_byte:field.end_byte].decode('utf-8')
                        call_type = 'method_call'
                        if operand and operand.type == 'identifier':
                            object_name = source_bytes[operand.start_byte:operand.end_byte].decode('utf-8')

                if function_name:
                    calls.append(FunctionCall(
                        function_name=function_name,
                        line_number=node.start_point[0] + 1,
                        call_type=call_type,
                        context=get_call_context(node, source_bytes),
                        object_name=object_name,
                        is_recursive=(current_function_name == function_name),
                    ))

        for child in node.children:
            traverse(child)

    traverse(tree_root)
    return calls


def _extract_calls_python(code: str, language: str, current_function_name: Optional[str] = None) -> list[FunctionCall]:
    """Extract function calls using Python Tree-sitter parsing (fallback)."""
    if not code or not code.strip():
        return []

    parser = get_parser(language)
    if not parser:
        logger.warning(f"No parser for language: {language}")
        return []

    try:
        source_bytes = code.encode('utf-8')
        tree = parser.parse(source_bytes)

        if language == 'python':
            return extract_python_calls(tree.root_node, source_bytes, current_function_name)
        if language == 'go':
            return extract_go_calls(tree.root_node, source_bytes, current_function_name)

        logger.debug(f"Call extraction not implemented for {language}")
        return []

    except Exception as e:
        logger.warning(f"Failed to extract calls from {language}: {e}")
        return []


def extract_calls(code: str, language: str, current_function_name: Optional[str] = None) -> list[FunctionCall]:
    """Extract function calls using Rust Tree-sitter parsing.

    Falls back to the Python Tree-sitter implementation when Rust parsing fails.
    """
    if not code or not code.strip():
        return []

    try:
        from src.rust_bridge import extract_calls as rust_extract_calls

        raw = rust_extract_calls(code, language, current_function_name)
        out: list[FunctionCall] = []
        for function_name, line_number, call_type, context, object_name, is_recursive in raw:
            out.append(
                FunctionCall(
                    function_name=function_name,
                    line_number=int(line_number),
                    call_type=call_type,
                    context=context,
                    object_name=object_name,
                    is_recursive=bool(is_recursive),
                )
            )
        return out
    except Exception as e:
        logger.debug(f"Rust call extraction failed, falling back to Python: {e}")
        return _extract_calls_python(code, language, current_function_name)


def extract_calls_from_function(code: str, language: str, function_name: str,
                                line_start: int, line_end: int) -> list[FunctionCall]:
    """Extract calls from a specific function's body."""
    lines = code.split('\n')
    function_code = '\n'.join(lines[line_start - 1:line_end])
    calls = extract_calls(function_code, language, function_name)

    for call in calls:
        call.line_number += (line_start - 1)

    return calls

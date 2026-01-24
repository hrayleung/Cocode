"""Extract function calls using the Rust Tree-sitter parser.

This module intentionally does not implement a Python fallback. Call extraction
must be provided by the cocode_rust extension.
"""

from dataclasses import dataclass, replace

from src.rust_bridge import extract_calls as _rust_extract_calls


@dataclass
class FunctionCall:
    """Represents a function call found in code."""

    function_name: str
    line_number: int
    call_type: str  # 'function_call', 'method_call', 'constructor'
    context: str | None = None  # 'loop', 'conditional', 'try_block', etc.
    object_name: str | None = None  # For method calls: object.method()
    is_recursive: bool = False


def extract_calls(
    code: str,
    language: str,
    current_function_name: str | None = None,
) -> list[FunctionCall]:
    """Extract function calls using Rust Tree-sitter parsing."""

    if not code or not code.strip():
        return []

    raw = _rust_extract_calls(code, language, current_function_name)
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


def extract_calls_from_function(
    code: str,
    language: str,
    function_name: str,
    line_start: int,
    line_end: int,
) -> list[FunctionCall]:
    """Extract calls from a specific function's body."""

    lines = code.split("\n")
    function_code = "\n".join(lines[line_start - 1 : line_end])
    calls = extract_calls(function_code, language, function_name)

    # Return new objects with adjusted line numbers (immutable pattern)
    offset = line_start - 1
    return [replace(call, line_number=call.line_number + offset) for call in calls]

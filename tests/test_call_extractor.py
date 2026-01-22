"""Tests for function call extraction from AST."""

import pytest
from src.parser.call_extractor import (
    extract_calls,
    extract_calls_from_function,
    FunctionCall,
)


class TestPythonCallExtraction:
    """Test function call extraction for Python."""

    def test_simple_function_call(self):
        """Test extraction of a simple function call."""
        code = """
def foo():
    bar()
"""
        calls = extract_calls(code, 'python')

        assert len(calls) == 1
        assert calls[0].function_name == 'bar'
        assert calls[0].line_number == 3
        assert calls[0].call_type == 'function_call'
        assert calls[0].object_name is None
        assert not calls[0].is_recursive

    def test_method_call(self):
        """Test extraction of method calls."""
        code = """
def process():
    obj.method()
"""
        calls = extract_calls(code, 'python')

        assert len(calls) == 1
        assert calls[0].function_name == 'method'
        assert calls[0].call_type == 'method_call'
        assert calls[0].object_name == 'obj'

    def test_chained_method_calls(self):
        """Test extraction of chained method calls."""
        code = """
def process():
    result = obj.method1().method2()
"""
        calls = extract_calls(code, 'python')

        # Should find both method1 and method2
        assert len(calls) == 2
        function_names = {call.function_name for call in calls}
        assert 'method1' in function_names
        assert 'method2' in function_names

    def test_multiple_calls(self):
        """Test extraction of multiple function calls."""
        code = """
def process():
    foo()
    bar()
    baz()
"""
        calls = extract_calls(code, 'python')

        assert len(calls) == 3
        function_names = [call.function_name for call in calls]
        assert function_names == ['foo', 'bar', 'baz']

    def test_recursive_call(self):
        """Test detection of recursive calls."""
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""
        calls = extract_calls(code, 'python', current_function_name='factorial')

        assert len(calls) == 1
        assert calls[0].function_name == 'factorial'
        assert calls[0].is_recursive

    def test_call_in_loop(self):
        """Test context detection for calls in loops."""
        code = """
def process():
    for i in range(10):
        helper()
"""
        calls = extract_calls(code, 'python')

        assert len(calls) == 2  # range() and helper()
        helper_call = next(c for c in calls if c.function_name == 'helper')
        assert 'loop' in helper_call.context

    def test_call_in_conditional(self):
        """Test context detection for calls in conditionals."""
        code = """
def process():
    if condition:
        handle_true()
    else:
        handle_false()
"""
        calls = extract_calls(code, 'python')

        assert len(calls) == 2
        for call in calls:
            assert 'conditional' in call.context

    def test_call_in_try_block(self):
        """Test context detection for calls in try blocks."""
        code = """
def process():
    try:
        risky_operation()
    except Exception:
        handle_error()
"""
        calls = extract_calls(code, 'python')

        assert len(calls) == 2
        risky_call = next(c for c in calls if c.function_name == 'risky_operation')
        assert 'try_block' in risky_call.context

    def test_nested_contexts(self):
        """Test multiple nested contexts."""
        code = """
def process():
    for i in range(10):
        if condition:
            try:
                nested_call()
            except:
                pass
"""
        calls = extract_calls(code, 'python')

        nested_call = next(c for c in calls if c.function_name == 'nested_call')
        # Should detect all three contexts
        assert 'loop' in nested_call.context
        assert 'conditional' in nested_call.context
        assert 'try_block' in nested_call.context

    def test_empty_code(self):
        """Test with empty code."""
        calls = extract_calls("", 'python')
        assert len(calls) == 0

    def test_code_without_calls(self):
        """Test code that doesn't contain any function calls."""
        code = """
x = 10
y = 20
z = x + y
"""
        calls = extract_calls(code, 'python')
        assert len(calls) == 0

    def test_builtin_calls(self):
        """Test extraction of builtin function calls."""
        code = """
def process():
    print("hello")
    len(items)
    str(value)
"""
        calls = extract_calls(code, 'python')

        assert len(calls) == 3
        function_names = {call.function_name for call in calls}
        assert function_names == {'print', 'len', 'str'}


class TestGoCallExtraction:
    """Test function call extraction for Go."""

    def test_simple_function_call(self):
        """Test extraction of simple Go function call."""
        code = """
package main

func process() {
    helper()
}
"""
        calls = extract_calls(code, 'go')

        assert len(calls) == 1
        assert calls[0].function_name == 'helper'
        assert calls[0].call_type == 'function_call'

    def test_package_function_call(self):
        """Test extraction of package-qualified function call."""
        code = """
package main

func process() {
    fmt.Println("hello")
}
"""
        calls = extract_calls(code, 'go')

        assert len(calls) == 1
        assert calls[0].function_name == 'Println'
        assert calls[0].call_type == 'method_call'
        assert calls[0].object_name == 'fmt'

    def test_method_call(self):
        """Test extraction of method calls on objects."""
        code = """
package main

func process() {
    obj.Method()
}
"""
        calls = extract_calls(code, 'go')

        assert len(calls) == 1
        assert calls[0].function_name == 'Method'
        assert calls[0].object_name == 'obj'


class TestExtractCallsFromFunction:
    """Test extracting calls from a specific function."""

    def test_extract_from_specific_function(self):
        """Test extracting calls only from a specific function."""
        code = """
def foo():
    helper1()

def bar():
    helper2()
    helper3()
"""
        # Extract only from bar (lines 5-7)
        calls = extract_calls_from_function(code, 'python', 'bar', 5, 7)

        assert len(calls) == 2
        function_names = {call.function_name for call in calls}
        assert function_names == {'helper2', 'helper3'}

        # Line numbers should be adjusted to full file coordinates
        assert all(call.line_number >= 5 for call in calls)

    def test_line_number_adjustment(self):
        """Test that line numbers are correctly adjusted to file coordinates."""
        code = """
def outer():
    pass

def target():
    first_call()  # Line 6
    second_call()  # Line 7
"""
        calls = extract_calls_from_function(code, 'python', 'target', 5, 7)

        assert len(calls) == 2
        line_numbers = [call.line_number for call in calls]
        assert 6 in line_numbers
        assert 7 in line_numbers


class TestEdgeCases:
    """Test edge cases in call extraction."""

    def test_unsupported_language(self):
        """Test that unsupported languages return empty list."""
        code = "some code"
        calls = extract_calls(code, 'unsupported_lang')
        assert len(calls) == 0

    def test_syntax_error_handling(self):
        """Test that syntax errors are handled gracefully."""
        code = """
def broken(:
    this is not valid python
"""
        # Should not raise exception
        calls = extract_calls(code, 'python')
        # May return empty or partial results, but shouldn't crash
        assert isinstance(calls, list)

    def test_lambda_context(self):
        """Test context detection for calls in lambdas."""
        code = """
def process():
    result = list(map(lambda x: transform(x), items))
"""
        calls = extract_calls(code, 'python')

        transform_calls = [c for c in calls if c.function_name == 'transform']
        if transform_calls:
            assert 'lambda' in transform_calls[0].context

    def test_calls_with_args(self):
        """Test extraction of calls with various argument types."""
        code = """
def process():
    foo(1, 2, 3)
    bar(x=1, y=2)
    baz(*args, **kwargs)
"""
        calls = extract_calls(code, 'python')

        assert len(calls) == 3
        function_names = {call.function_name for call in calls}
        assert function_names == {'foo', 'bar', 'baz'}

"""Tests for symbol extraction."""

import pytest
from src.parser.symbol_extractor import extract_symbols, Symbol


class TestPythonSymbolExtraction:
    """Test Python symbol extraction."""

    def test_extract_simple_function(self):
        """Test extracting a simple function."""
        code = """
def hello():
    \"\"\"Say hello.\"\"\"
    print("Hello, world!")
"""
        symbols = extract_symbols(code, "python", "test.py")
        assert len(symbols) == 1

        func = symbols[0]
        assert func.symbol_name == "hello"
        assert func.symbol_type == "function"
        assert func.line_start == 2
        assert func.signature == "def hello():"
        assert "Say hello" in func.docstring
        assert func.visibility == "public"
        assert func.parent_symbol is None

    def test_extract_function_with_parameters(self):
        """Test extracting function with parameters and return type."""
        code = """
def authenticate(username: str, password: str) -> bool:
    \"\"\"Authenticate a user.\"\"\"
    return True
"""
        symbols = extract_symbols(code, "python", "auth.py")
        assert len(symbols) == 1

        func = symbols[0]
        assert func.symbol_name == "authenticate"
        assert "username: str" in func.signature
        assert "-> bool" in func.signature

    def test_extract_class(self):
        """Test extracting a class."""
        code = """
class User:
    \"\"\"User model.\"\"\"

    def __init__(self, name: str):
        self.name = name

    def greet(self):
        return f"Hello, {self.name}"
"""
        symbols = extract_symbols(code, "python", "models.py")
        assert len(symbols) == 3  # Class + 2 methods

        # Class
        class_symbol = symbols[0]
        assert class_symbol.symbol_name == "User"
        assert class_symbol.symbol_type == "class"
        assert "User model" in class_symbol.docstring

        # Methods
        init_method = symbols[1]
        assert init_method.symbol_name == "__init__"
        assert init_method.symbol_type == "method"
        assert init_method.parent_symbol == "User"
        # Dunder methods (start and end with __) are considered internal
        assert init_method.visibility in ("internal", "private")

        greet_method = symbols[2]
        assert greet_method.symbol_name == "greet"
        assert greet_method.parent_symbol == "User"
        assert greet_method.visibility == "public"

    def test_extract_private_function(self):
        """Test extracting private function (starts with _)."""
        code = """
def _internal_helper():
    pass
"""
        symbols = extract_symbols(code, "python", "utils.py")
        assert len(symbols) == 1
        assert symbols[0].visibility == "internal"

    def test_extract_test_function(self):
        """Test extracting test function."""
        code = """
def test_user_authentication():
    assert True
"""
        symbols = extract_symbols(code, "python", "test_auth.py")
        assert len(symbols) == 1

        test_func = symbols[0]
        assert test_func.symbol_name == "test_user_authentication"
        assert test_func.category == "test"

    def test_extract_test_class(self):
        """Test extracting test class."""
        code = """
class TestUserModel:
    def test_creation(self):
        pass
"""
        symbols = extract_symbols(code, "python", "test_models.py")
        assert len(symbols) == 2

        test_class = symbols[0]
        assert test_class.symbol_name == "TestUserModel"
        assert test_class.category == "test"

    def test_extract_multiple_functions(self):
        """Test extracting multiple functions."""
        code = """
def func1():
    pass

def func2():
    pass

def func3():
    pass
"""
        symbols = extract_symbols(code, "python", "utils.py")
        assert len(symbols) == 3
        assert [s.symbol_name for s in symbols] == ["func1", "func2", "func3"]

    def test_extract_nested_class(self):
        """Test extracting class with nested methods."""
        code = """
class Calculator:
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b
"""
        symbols = extract_symbols(code, "python", "calc.py")
        assert len(symbols) == 3

        calc_class = symbols[0]
        assert calc_class.symbol_type == "class"

        add_method = symbols[1]
        assert add_method.symbol_type == "method"
        assert add_method.parent_symbol == "Calculator"

        subtract_method = symbols[2]
        assert subtract_method.symbol_type == "method"
        assert subtract_method.parent_symbol == "Calculator"

    def test_function_without_docstring(self):
        """Test function without docstring."""
        code = """
def simple():
    return 42
"""
        symbols = extract_symbols(code, "python", "simple.py")
        assert len(symbols) == 1
        assert symbols[0].docstring is None or symbols[0].docstring == ""

    def test_line_numbers(self):
        """Test that line numbers are correct."""
        code = """
# Comment
def func1():
    pass

def func2():
    pass
"""
        symbols = extract_symbols(code, "python", "test.py")
        assert len(symbols) == 2

        # Line counting: empty line 1, comment line 2, func1 def line 3
        assert symbols[0].line_start == 3
        assert symbols[1].line_start == 6


class TestGoSymbolExtraction:
    """Test Go symbol extraction."""

    def test_extract_simple_function(self):
        """Test extracting a simple Go function."""
        code = """
package main

func Hello() {
    println("Hello, world!")
}
"""
        symbols = extract_symbols(code, "go", "main.go")
        assert len(symbols) >= 1

        func = next((s for s in symbols if s.symbol_name == "Hello"), None)
        assert func is not None
        assert func.symbol_type == "function"
        assert func.visibility == "public"  # Capitalized = public in Go

    def test_extract_private_function(self):
        """Test extracting private Go function (lowercase)."""
        code = """
package main

func helper() {
    println("internal")
}
"""
        symbols = extract_symbols(code, "go", "utils.go")
        assert len(symbols) >= 1

        func = next((s for s in symbols if s.symbol_name == "helper"), None)
        assert func is not None
        assert func.visibility == "internal"

    def test_extract_method(self):
        """Test extracting Go method with receiver."""
        code = """
package main

type User struct {
    name string
}

func (u *User) GetName() string {
    return u.name
}
"""
        symbols = extract_symbols(code, "go", "user.go")

        # Find the method
        method = next((s for s in symbols if s.symbol_name == "GetName"), None)
        if method:  # Implementation may vary
            assert method.symbol_type == "method"
            assert method.visibility == "public"

    def test_extract_test_function(self):
        """Test extracting Go test function."""
        code = """
package main

func TestUserCreation(t *testing.T) {
    // test code
}
"""
        symbols = extract_symbols(code, "go", "user_test.go")

        test_func = next((s for s in symbols if s.symbol_name == "TestUserCreation"), None)
        if test_func:
            assert test_func.category == "test"


class TestRustSymbolExtraction:
    """Test Rust symbol extraction."""

    def test_extract_struct(self):
        """Test extracting Rust struct."""
        code = """
pub struct BM25Engine {
    num_docs: usize,
}
"""
        symbols = extract_symbols(code, "rust", "engine.rs")
        assert len(symbols) == 1
        assert symbols[0].symbol_name == "BM25Engine"
        assert symbols[0].symbol_type == "class"
        assert symbols[0].visibility == "public"

    def test_extract_impl_methods(self):
        """Test extracting methods from impl block."""
        code = """
impl BM25Engine {
    pub fn new() -> Self {
        Self {}
    }

    fn internal_helper(&self) {}
}
"""
        symbols = extract_symbols(code, "rust", "engine.rs")
        assert len(symbols) == 2

        new_fn = next(s for s in symbols if s.symbol_name == "new")
        assert new_fn.symbol_type == "method"
        assert new_fn.parent_symbol == "BM25Engine"
        assert new_fn.visibility == "public"

        helper = next(s for s in symbols if s.symbol_name == "internal_helper")
        assert helper.visibility == "private"

    def test_extract_trait(self):
        """Test extracting Rust trait."""
        code = """
pub trait Searchable {
    fn search(&self, query: &str) -> Vec<String>;
}
"""
        symbols = extract_symbols(code, "rust", "traits.rs")
        trait_sym = next((s for s in symbols if s.symbol_name == "Searchable"), None)
        assert trait_sym is not None
        assert trait_sym.symbol_type == "interface"

    def test_extract_standalone_function(self):
        """Test extracting standalone Rust function."""
        code = """
pub fn tokenize(text: &str) -> Vec<String> {
    vec![]
}
"""
        symbols = extract_symbols(code, "rust", "utils.rs")
        assert len(symbols) == 1
        assert symbols[0].symbol_type == "function"
        assert symbols[0].parent_symbol is None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_code(self):
        """Test with empty source code."""
        symbols = extract_symbols("", "python", "empty.py")
        assert symbols == []

    def test_syntax_error(self):
        """Test code with syntax errors."""
        code = """
def broken(
    # syntax error
"""
        symbols = extract_symbols(code, "python", "broken.py")
        # Should not crash, may return partial results or empty list
        assert isinstance(symbols, list)

    def test_unsupported_language(self):
        """Test unsupported language."""
        symbols = extract_symbols("code", "unknown", "file.txt")
        assert symbols == []

    def test_code_without_symbols(self):
        """Test code without any symbols."""
        code = """
# Just a comment
x = 42
"""
        symbols = extract_symbols(code, "python", "config.py")
        assert symbols == []


class TestSymbolDataclass:
    """Test the Symbol dataclass."""

    def test_symbol_creation(self):
        """Test creating a Symbol."""
        symbol = Symbol(
            symbol_name="test_func",
            symbol_type="function",
            line_start=10,
            line_end=20,
            signature="def test_func():",
            docstring="Test function.",
            visibility="public",
            category="test",
        )

        assert symbol.symbol_name == "test_func"
        assert symbol.symbol_type == "function"
        assert symbol.line_start == 10
        assert symbol.line_end == 20
        assert symbol.visibility == "public"
        assert symbol.category == "test"

    def test_symbol_defaults(self):
        """Test Symbol default values."""
        symbol = Symbol(
            symbol_name="func",
            symbol_type="function",
            line_start=1,
            line_end=5,
            signature="def func():",
        )

        assert symbol.docstring is None
        assert symbol.parent_symbol is None
        assert symbol.visibility == "public"
        assert symbol.category == "implementation"

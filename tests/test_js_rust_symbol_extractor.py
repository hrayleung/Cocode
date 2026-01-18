"""Tests for JavaScript/TypeScript symbol extraction."""

import pytest
from src.parser.symbol_extractor import extract_symbols


class TestJavaScriptSymbolExtraction:
    """Test JavaScript symbol extraction."""

    def test_extract_function_declaration(self):
        code = """
function greet(name) {
    return "Hello, " + name;
}
"""
        symbols = extract_symbols(code, "javascript", "test.js")
        assert len(symbols) == 1
        assert symbols[0].symbol_name == "greet"
        assert symbols[0].symbol_type == "function"

    def test_extract_arrow_function(self):
        code = """
const add = (a, b) => {
    return a + b;
};
"""
        symbols = extract_symbols(code, "javascript", "test.js")
        assert len(symbols) == 1
        assert symbols[0].symbol_name == "add"
        assert symbols[0].symbol_type == "function"

    def test_extract_class(self):
        code = """
class User {
    constructor(name) {
        this.name = name;
    }
    
    getName() {
        return this.name;
    }
}
"""
        symbols = extract_symbols(code, "javascript", "test.js")
        assert len(symbols) >= 2
        class_sym = next(s for s in symbols if s.symbol_type == "class")
        assert class_sym.symbol_name == "User"
        method_sym = next(s for s in symbols if s.symbol_name == "getName")
        assert method_sym.symbol_type == "method"
        assert method_sym.parent_symbol == "User"

    def test_extract_test_file(self):
        code = """
function testAdd() {
    return true;
}
"""
        symbols = extract_symbols(code, "javascript", "add.test.js")
        assert len(symbols) == 1
        assert symbols[0].category == "test"


class TestTypeScriptSymbolExtraction:
    """Test TypeScript symbol extraction."""

    def test_extract_interface(self):
        code = """
interface User {
    name: string;
    age: number;
}
"""
        symbols = extract_symbols(code, "typescript", "types.ts")
        assert len(symbols) == 1
        assert symbols[0].symbol_name == "User"
        assert symbols[0].symbol_type == "interface"

    def test_extract_type_alias(self):
        code = """
type Status = "active" | "inactive";
"""
        symbols = extract_symbols(code, "typescript", "types.ts")
        assert len(symbols) == 1
        assert symbols[0].symbol_name == "Status"

    def test_extract_exported_function(self):
        code = """
export function fetchUser(id: string): Promise<User> {
    return fetch(`/users/${id}`);
}
"""
        symbols = extract_symbols(code, "typescript", "api.ts")
        assert len(symbols) == 1
        assert symbols[0].symbol_name == "fetchUser"
        assert symbols[0].symbol_type == "function"


class TestRustSymbolExtraction:
    """Test Rust symbol extraction."""

    def test_extract_function(self):
        code = """
fn greet(name: &str) -> String {
    format!("Hello, {}", name)
}
"""
        symbols = extract_symbols(code, "rust", "lib.rs")
        assert len(symbols) == 1
        assert symbols[0].symbol_name == "greet"
        assert symbols[0].symbol_type == "function"
        assert symbols[0].visibility == "private"

    def test_extract_pub_function(self):
        code = """
pub fn public_greet(name: &str) -> String {
    format!("Hello, {}", name)
}
"""
        symbols = extract_symbols(code, "rust", "lib.rs")
        assert len(symbols) == 1
        assert symbols[0].visibility == "public"

    def test_extract_struct(self):
        code = """
pub struct User {
    name: String,
    age: u32,
}
"""
        symbols = extract_symbols(code, "rust", "models.rs")
        assert len(symbols) == 1
        assert symbols[0].symbol_name == "User"
        assert symbols[0].symbol_type == "class"
        assert symbols[0].visibility == "public"

    def test_extract_enum(self):
        code = """
pub enum Status {
    Active,
    Inactive,
}
"""
        symbols = extract_symbols(code, "rust", "types.rs")
        assert len(symbols) == 1
        assert symbols[0].symbol_name == "Status"
        assert symbols[0].symbol_type == "class"

    def test_extract_trait(self):
        code = """
pub trait Greetable {
    fn greet(&self) -> String;
}
"""
        symbols = extract_symbols(code, "rust", "traits.rs")
        assert len(symbols) == 1
        assert symbols[0].symbol_name == "Greetable"
        assert symbols[0].symbol_type == "interface"

    def test_extract_impl_methods(self):
        code = """
struct User {
    name: String,
}

impl User {
    pub fn new(name: String) -> Self {
        User { name }
    }
    
    fn get_name(&self) -> &str {
        &self.name
    }
}
"""
        symbols = extract_symbols(code, "rust", "user.rs")
        struct_sym = next((s for s in symbols if s.symbol_name == "User" and s.symbol_type == "class"), None)
        assert struct_sym is not None
        
        new_method = next((s for s in symbols if s.symbol_name == "new"), None)
        assert new_method is not None
        assert new_method.symbol_type == "method"
        assert new_method.parent_symbol == "User"
        assert new_method.visibility == "public"
        
        get_name = next((s for s in symbols if s.symbol_name == "get_name"), None)
        assert get_name is not None
        assert get_name.visibility == "private"

    def test_extract_test_function(self):
        code = """
#[test]
fn test_greet() {
    assert_eq!(greet("World"), "Hello, World");
}
"""
        symbols = extract_symbols(code, "rust", "lib.rs")
        assert len(symbols) == 1
        assert symbols[0].symbol_name == "test_greet"
        assert symbols[0].category == "test"

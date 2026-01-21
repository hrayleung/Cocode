"""Integration tests for graph expansion with AST import extraction."""

import pytest
from src.retrieval.graph_expansion import extract_imports


class TestASTImportExtraction:
    """Test that graph_expansion uses AST-based import extraction."""

    def test_python_ast_extraction(self):
        """Test Python import extraction uses AST."""
        code = """
import os
import sys
from pathlib import Path
"""
        imports = extract_imports(code, "python")
        assert {"os", "sys", "pathlib"}.issubset(set(imports))

    def test_go_ast_extraction(self):
        """Test Go import extraction uses AST."""
        code = """
package main

import (
    "fmt"
    "strings"
)
"""
        imports = extract_imports(code, "go")
        assert {"fmt", "strings"}.issubset(set(imports))

    def test_rust_ast_extraction(self):
        """Test Rust import extraction uses AST."""
        code = """
use std::collections::HashMap;
use tokio::sync::Mutex;
mod utils;
"""
        imports = extract_imports(code, "rust")
        assert "std" in imports
        assert "tokio" in imports
        assert "utils" in imports

    def test_javascript_ast_extraction(self):
        """Test JavaScript import extraction uses AST."""
        code = """
import React from 'react';
import { useState } from 'react';
const express = require('express');
"""
        imports = extract_imports(code, "javascript")
        assert "react" in imports
        assert "express" in imports

    def test_typescript_ast_extraction(self):
        """Test TypeScript import extraction uses AST."""
        code = """
import { Component } from '@angular/core';
import type { User } from './types';
"""
        imports = extract_imports(code, "typescript")
        assert "@angular/core" in imports
        assert "./types" in imports

    def test_invalid_syntax_handles_gracefully(self):
        """Test that invalid syntax is handled gracefully (returns list, may be partial)."""
        # Invalid Python syntax
        code = """
import os
def broken(
"""
        # Should return a list - may contain partial results from valid portion
        imports = extract_imports(code, "python")
        assert isinstance(imports, list)

    def test_empty_code(self):
        """Test with empty code."""
        imports = extract_imports("", "python")
        assert imports == []

    def test_code_with_no_imports(self):
        """Test code without imports."""
        code = """
def hello():
    print("Hello, world!")
"""
        imports = extract_imports(code, "python")
        assert imports == []

    def test_unsupported_language_returns_empty(self):
        """Test that unsupported languages return empty list."""
        imports = extract_imports("import foo", "unknown")
        assert imports == []

    def test_complex_python_imports(self):
        """Test complex Python import patterns."""
        code = """
import os.path
from collections.abc import Mapping
from typing import List, Optional, Dict
import numpy as np
"""
        imports = extract_imports(code, "python")
        assert "os.path" in imports
        assert "collections.abc" in imports
        assert "typing" in imports
        assert "numpy" in imports

    def test_complex_rust_imports(self):
        """Test complex Rust import patterns."""
        code = """
use std::io::{Read, Write};
use std::collections::HashMap as Map;
use tokio::{fs, env};
extern crate serde;
mod config;
"""
        imports = extract_imports(code, "rust")
        assert "std" in imports
        assert "tokio" in imports
        assert "serde" in imports
        assert "config" in imports

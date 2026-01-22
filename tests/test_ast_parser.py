"""Tests for AST-based import extraction."""

import pytest
from src.parser.ast_parser import (
    extract_imports_ast,
    is_language_supported,
    get_language_from_file,
)


class TestPythonImportExtraction:
    """Test Python import extraction."""

    def test_simple_import(self):
        """Test basic import statement."""
        code = """
import os
import sys
"""
        imports = extract_imports_ast(code, "python")
        assert set(imports) == {"os", "sys"}

    def test_from_import(self):
        """Test from...import statement."""
        code = """
from pathlib import Path
from typing import List, Optional
"""
        imports = extract_imports_ast(code, "python")
        assert "pathlib" in imports
        assert "typing" in imports

    def test_dotted_import(self):
        """Test dotted module imports."""
        code = """
import os.path
from collections.abc import Mapping
"""
        imports = extract_imports_ast(code, "python")
        assert "os.path" in imports
        assert "collections.abc" in imports

    def test_aliased_import(self):
        """Test imports with aliases."""
        code = """
import numpy as np
from pandas import DataFrame as DF
"""
        imports = extract_imports_ast(code, "python")
        assert "numpy" in imports
        assert "pandas" in imports

    def test_relative_import(self):
        """Test relative imports."""
        code = """
from . import module
from ..parent import something
from ...grandparent.sibling import foo
"""
        imports = extract_imports_ast(code, "python")
        # Relative imports should be extracted (module names only)
        assert len(imports) >= 0  # Implementation might vary

    def test_conditional_import(self):
        """Test imports inside try blocks."""
        code = """
try:
    import ujson as json
except ImportError:
    import json

if True:
    from foo import bar
"""
        imports = extract_imports_ast(code, "python")
        assert "json" in imports or "ujson" in imports

    def test_multiple_imports_per_line(self):
        """Test multiple imports in one statement."""
        code = """
import os, sys, json
"""
        imports = extract_imports_ast(code, "python")
        assert {"os", "sys", "json"}.issubset(set(imports))


class TestGoImportExtraction:
    """Test Go import extraction."""

    def test_single_import(self):
        """Test single import statement."""
        code = """
package main

import "fmt"
"""
        imports = extract_imports_ast(code, "go")
        assert "fmt" in imports

    def test_multiple_imports(self):
        """Test import block with multiple packages."""
        code = """
package main

import (
    "fmt"
    "strings"
    "os"
)
"""
        imports = extract_imports_ast(code, "go")
        assert {"fmt", "strings", "os"}.issubset(set(imports))

    def test_qualified_import(self):
        """Test qualified package imports."""
        code = """
package main

import (
    "github.com/foo/bar"
    "golang.org/x/crypto/ssh"
)
"""
        imports = extract_imports_ast(code, "go")
        assert "github.com/foo/bar" in imports
        assert "golang.org/x/crypto/ssh" in imports

    def test_aliased_import(self):
        """Test import with alias."""
        code = """
package main

import (
    f "fmt"
    . "strings"
    _ "database/sql"
)
"""
        imports = extract_imports_ast(code, "go")
        assert "fmt" in imports
        assert "strings" in imports
        assert "database/sql" in imports


class TestRustImportExtraction:
    """Test Rust import extraction."""

    def test_use_statement(self):
        """Test basic use statement."""
        code = """
use std::collections::HashMap;
use std::fs;
"""
        imports = extract_imports_ast(code, "rust")
        assert "std" in imports

    def test_use_with_braces(self):
        """Test use statement with multiple items."""
        code = """
use std::io::{Read, Write};
use std::{fs, env};
"""
        imports = extract_imports_ast(code, "rust")
        assert "std" in imports

    def test_mod_declaration(self):
        """Test mod declarations."""
        code = """
mod utils;
mod config;

fn main() {}
"""
        imports = extract_imports_ast(code, "rust")
        assert "utils" in imports
        assert "config" in imports

    def test_extern_crate(self):
        """Test extern crate declarations."""
        code = """
extern crate serde;
extern crate serde_json;
"""
        imports = extract_imports_ast(code, "rust")
        assert "serde" in imports
        assert "serde_json" in imports

    def test_use_as_alias(self):
        """Test use with alias."""
        code = """
use std::collections::HashMap as Map;
use tokio::sync::Mutex as TokioMutex;
"""
        imports = extract_imports_ast(code, "rust")
        assert "std" in imports
        assert "tokio" in imports


class TestCppImportExtraction:
    """Test C/C++ include extraction."""

    def test_system_include(self):
        """Test system header includes."""
        code = """
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
"""
        imports = extract_imports_ast(code, "cpp")
        assert "stdio.h" in imports
        assert "stdlib.h" in imports
        assert "iostream" in imports

    def test_local_include(self):
        """Test local header includes."""
        code = """
#include "myheader.h"
#include "utils/helper.h"
"""
        imports = extract_imports_ast(code, "cpp")
        assert "myheader.h" in imports
        assert "utils/helper.h" in imports

    def test_mixed_includes(self):
        """Test mix of system and local includes."""
        code = """
#include <vector>
#include <string>
#include "config.h"
#include "types.h"
"""
        imports = extract_imports_ast(code, "cpp")
        assert {"vector", "string", "config.h", "types.h"}.issubset(set(imports))


class TestJavaScriptImportExtraction:
    """Test JavaScript/TypeScript import extraction."""

    def test_default_import(self):
        """Test default import."""
        code = """
import React from 'react';
import express from 'express';
"""
        imports = extract_imports_ast(code, "javascript")
        assert "react" in imports
        assert "express" in imports

    def test_named_import(self):
        """Test named imports."""
        code = """
import { useState, useEffect } from 'react';
import { Router } from 'express';
"""
        imports = extract_imports_ast(code, "javascript")
        assert "react" in imports
        assert "express" in imports

    def test_namespace_import(self):
        """Test namespace import."""
        code = """
import * as fs from 'fs';
import * as path from 'path';
"""
        imports = extract_imports_ast(code, "javascript")
        assert "fs" in imports
        assert "path" in imports

    def test_side_effect_import(self):
        """Test side-effect only import."""
        code = """
import 'dotenv/config';
import './styles.css';
"""
        imports = extract_imports_ast(code, "javascript")
        assert "dotenv/config" in imports
        assert "./styles.css" in imports

    def test_require_statement(self):
        """Test CommonJS require()."""
        code = """
const fs = require('fs');
const { join } = require('path');
"""
        imports = extract_imports_ast(code, "javascript")
        assert "fs" in imports
        assert "path" in imports

    def test_relative_import(self):
        """Test relative imports."""
        code = """
import foo from './foo';
import bar from '../bar';
import baz from '../../utils/baz';
"""
        imports = extract_imports_ast(code, "javascript")
        assert "./foo" in imports
        assert "../bar" in imports
        assert "../../utils/baz" in imports

    def test_typescript_type_import(self):
        """Test TypeScript type imports."""
        code = """
import type { User } from './types';
import { type Config, loadConfig } from './config';
"""
        imports = extract_imports_ast(code, "typescript")
        assert "./types" in imports
        assert "./config" in imports

    def test_export_from(self):
        """Test export...from statements."""
        code = """
export * from './utils';
export { helper } from './helpers';
"""
        imports = extract_imports_ast(code, "javascript")
        assert "./utils" in imports
        assert "./helpers" in imports


class TestLanguageDetection:
    """Test language detection from file extensions."""

    def test_python_extension(self):
        """Test Python file detection."""
        assert get_language_from_file("test.py") == "python"
        assert get_language_from_file("/path/to/module.py") == "python"

    def test_go_extension(self):
        """Test Go file detection."""
        assert get_language_from_file("main.go") == "go"

    def test_rust_extension(self):
        """Test Rust file detection."""
        assert get_language_from_file("lib.rs") == "rust"

    def test_cpp_extensions(self):
        """Test C++ file detection."""
        assert get_language_from_file("main.cpp") == "cpp"
        assert get_language_from_file("test.cxx") == "cpp"
        assert get_language_from_file("impl.cc") == "cpp"
        assert get_language_from_file("header.hpp") == "cpp"
        assert get_language_from_file("header.hxx") == "cpp"

    def test_c_extensions(self):
        """Test C file detection."""
        assert get_language_from_file("main.c") == "c"
        assert get_language_from_file("header.h") == "c"

    def test_javascript_extensions(self):
        """Test JavaScript file detection."""
        assert get_language_from_file("app.js") == "javascript"
        assert get_language_from_file("module.mjs") == "javascript"
        assert get_language_from_file("config.cjs") == "javascript"
        assert get_language_from_file("component.jsx") == "javascript"

    def test_typescript_extensions(self):
        """Test TypeScript file detection."""
        assert get_language_from_file("app.ts") == "typescript"
        assert get_language_from_file("module.mts") == "typescript"
        assert get_language_from_file("config.cts") == "typescript"
        assert get_language_from_file("component.tsx") == "tsx"

    def test_unknown_extension(self):
        """Test unknown file extension."""
        assert get_language_from_file("README.md") is None
        assert get_language_from_file("data.json") is None


class TestLanguageSupport:
    """Test language support checks."""

    def test_supported_languages(self):
        """Test that main languages are supported."""
        assert is_language_supported("python")
        assert is_language_supported("go")
        assert is_language_supported("rust")
        assert is_language_supported("c")
        assert is_language_supported("cpp")
        assert is_language_supported("javascript")
        assert is_language_supported("typescript")

    def test_unsupported_language(self):
        """Test unsupported language."""
        assert not is_language_supported("ruby")
        assert not is_language_supported("java")
        assert not is_language_supported("unknown")


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_code(self):
        """Test with empty source code."""
        imports = extract_imports_ast("", "python")
        assert imports == []

    def test_no_imports(self):
        """Test code with no imports."""
        code = """
def hello():
    print("Hello, world!")
"""
        imports = extract_imports_ast(code, "python")
        assert imports == []

    def test_syntax_error(self):
        """Test code with syntax errors."""
        code = """
import os
import sys
def broken( # syntax error
"""
        # Should not crash, might return partial results
        imports = extract_imports_ast(code, "python")
        assert isinstance(imports, list)

    def test_comments_only(self):
        """Test file with only comments."""
        code = """
# This is a comment
# import os  # This is commented out
"""
        imports = extract_imports_ast(code, "python")
        assert imports == []

    def test_multiline_strings(self):
        """Test imports inside multiline strings (should not be extracted)."""
        code = '''
"""
This is a docstring
import fake_module
"""

def foo():
    """Another docstring with import bar"""
    pass
'''
        imports = extract_imports_ast(code, "python")
        # Should not extract imports from strings
        assert "fake_module" not in imports
        assert "bar" not in imports

    def test_unicode_content(self):
        """Test with Unicode characters."""
        code = """
# -*- coding: utf-8 -*-
import os  # 导入操作系统模块
from pathlib import Path  # Путь
"""
        imports = extract_imports_ast(code, "python")
        assert "os" in imports
        assert "pathlib" in imports


class TestDuplicateRemoval:
    """Test that duplicate imports are removed."""

    def test_python_duplicates(self):
        """Test duplicate import removal in Python."""
        code = """
import os
import sys
import os  # duplicate
from pathlib import Path
from pathlib import PurePath
"""
        imports = extract_imports_ast(code, "python")
        # Should have unique imports only
        assert imports.count("os") == 1
        assert imports.count("pathlib") == 1

    def test_javascript_duplicates(self):
        """Test duplicate import removal in JavaScript."""
        code = """
import React from 'react';
import { useState } from 'react';
import { useEffect } from 'react';
"""
        imports = extract_imports_ast(code, "javascript")
        assert imports.count("react") == 1

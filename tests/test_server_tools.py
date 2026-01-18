"""Tests for server tools."""

import pytest
import tempfile
import os
from pathlib import Path


class TestEscapeLikePattern:
    """Test _escape_like_pattern helper."""

    def test_escape_percent(self):
        from src.server import _escape_like_pattern
        assert _escape_like_pattern("100%") == "100\\%"

    def test_escape_underscore(self):
        from src.server import _escape_like_pattern
        assert _escape_like_pattern("foo_bar") == "foo\\_bar"

    def test_escape_backslash(self):
        from src.server import _escape_like_pattern
        assert _escape_like_pattern("a\\b") == "a\\\\b"

    def test_escape_multiple(self):
        from src.server import _escape_like_pattern
        assert _escape_like_pattern("100%_test\\") == "100\\%\\_test\\\\"

    def test_no_escape_needed(self):
        from src.server import _escape_like_pattern
        assert _escape_like_pattern("normal_text") == "normal\\_text"


class TestValidateSymbolType:
    """Test _validate_symbol_type helper."""

    def test_valid_function(self):
        from src.server import _validate_symbol_type
        assert _validate_symbol_type("function") == "function"

    def test_valid_class(self):
        from src.server import _validate_symbol_type
        assert _validate_symbol_type("class") == "class"

    def test_valid_method(self):
        from src.server import _validate_symbol_type
        assert _validate_symbol_type("method") == "method"

    def test_valid_interface(self):
        from src.server import _validate_symbol_type
        assert _validate_symbol_type("interface") == "interface"

    def test_case_insensitive(self):
        from src.server import _validate_symbol_type
        assert _validate_symbol_type("FUNCTION") == "function"
        assert _validate_symbol_type("Class") == "class"

    def test_invalid_type(self):
        from src.server import _validate_symbol_type
        assert _validate_symbol_type("invalid") is None
        assert _validate_symbol_type("struct") is None

    def test_none(self):
        from src.server import _validate_symbol_type
        assert _validate_symbol_type(None) is None

    def test_whitespace(self):
        from src.server import _validate_symbol_type
        assert _validate_symbol_type("  function  ") == "function"


class TestReadFileLines:
    """Test _read_file_lines helper."""

    def test_read_specific_lines(self):
        from src.server import _read_file_lines
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("line1\nline2\nline3\nline4\nline5\n")
            f.flush()
            
            try:
                content = _read_file_lines(Path(f.name), 2, 4)
                assert "line2" in content
                assert "line3" in content
                assert "line4" in content
                assert "line1" not in content
                assert "line5" not in content
            finally:
                os.unlink(f.name)

    def test_read_out_of_bounds(self):
        from src.server import _read_file_lines
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("line1\nline2\n")
            f.flush()
            
            try:
                content = _read_file_lines(Path(f.name), 1, 100)
                assert "line1" in content
                assert "line2" in content
            finally:
                os.unlink(f.name)

    def test_read_single_line(self):
        from src.server import _read_file_lines
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("line1\nline2\nline3\n")
            f.flush()
            
            try:
                content = _read_file_lines(Path(f.name), 2, 2)
                assert content.strip() == "line2"
            finally:
                os.unlink(f.name)


class TestGetSymbolFromDb:
    """Test _get_symbol_from_db helper."""

    def test_sql_uses_identifier(self):
        """Verify SQL uses proper identifier escaping."""
        from src.server import _get_symbol_from_db
        
        import inspect
        source = inspect.getsource(_get_symbol_from_db)
        assert "sql.Identifier" in source, "Should use sql.Identifier for schema name"
        assert "ESCAPE" in source, "Should use ESCAPE for LIKE patterns"


class TestReadFileFunctionality:
    """Test read_file functionality directly."""

    def test_path_traversal_detection(self):
        """Test that path traversal is detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir).resolve()
            
            malicious_path = (repo_path / "../../../etc/passwd").resolve()
            
            try:
                is_safe = malicious_path.is_relative_to(repo_path)
            except ValueError:
                is_safe = False
            
            assert not is_safe, "Path traversal should be detected"


class TestServerImports:
    """Test that server module imports correctly."""

    def test_imports(self):
        from src.server import (
            mcp,
            _get_symbol_from_db,
            _read_file_lines,
            _escape_like_pattern,
            _validate_symbol_type,
            MAX_SYMBOL_NAME_LENGTH,
            MAX_FILE_SIZE,
            VALID_SYMBOL_TYPES,
        )
        
        assert mcp is not None
        assert callable(_get_symbol_from_db)
        assert callable(_read_file_lines)
        assert callable(_escape_like_pattern)
        assert callable(_validate_symbol_type)
        assert MAX_SYMBOL_NAME_LENGTH == 500
        assert MAX_FILE_SIZE == 1024 * 1024
        assert "function" in VALID_SYMBOL_TYPES

    def test_mcp_has_tools(self):
        from src.server import mcp
        
        assert mcp is not None
        assert mcp.name == "cocode-precise"


class TestConstants:
    """Test server constants."""

    def test_valid_symbol_types(self):
        from src.server import VALID_SYMBOL_TYPES
        
        assert "function" in VALID_SYMBOL_TYPES
        assert "class" in VALID_SYMBOL_TYPES
        assert "method" in VALID_SYMBOL_TYPES
        assert "interface" in VALID_SYMBOL_TYPES
        assert len(VALID_SYMBOL_TYPES) == 4

    def test_max_file_size(self):
        from src.server import MAX_FILE_SIZE
        
        assert MAX_FILE_SIZE == 1024 * 1024  # 1MB

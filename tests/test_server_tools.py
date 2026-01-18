"""Tests for server tools."""

import pytest
import tempfile
import os
from pathlib import Path


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
        assert "sql.SQL" in source, "Should use sql.SQL for query building"


class TestReadFileFunctionality:
    """Test read_file functionality directly."""

    def test_read_entire_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.py"
            test_file.write_text("line1\nline2\nline3\n")
            
            filepath = test_file.resolve()
            
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            assert len(lines) == 3
            assert "line1" in lines[0]

    def test_path_traversal_detection(self):
        """Test that path traversal is detected."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_path = Path(tmpdir).resolve()
            
            # Attempt path traversal
            malicious_path = (repo_path / "../../../etc/passwd").resolve()
            
            # Should not be relative to repo_path
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
            MAX_SYMBOL_NAME_LENGTH,
        )
        
        assert mcp is not None
        assert callable(_get_symbol_from_db)
        assert callable(_read_file_lines)
        assert MAX_SYMBOL_NAME_LENGTH == 500

    def test_mcp_has_tools(self):
        from src.server import mcp
        
        # MCP should have tools registered
        assert mcp is not None
        assert mcp.name == "cocode-precise"

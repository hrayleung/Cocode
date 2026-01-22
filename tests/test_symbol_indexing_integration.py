"""Integration tests for symbol indexing."""

import pytest
from unittest.mock import Mock, patch
from src.indexer.symbol_indexing import (
    create_symbols_table,
    index_file_symbols,
    delete_file_symbols,
    generate_symbol_text,
)
from src.parser.symbol_extractor import Symbol


class TestSymbolTextGeneration:
    """Test symbol text generation for embeddings."""

    def test_generate_symbol_text_function(self):
        """Test text generation for a function."""
        symbol = Symbol(
            symbol_name="authenticate",
            symbol_type="function",
            line_start=10,
            line_end=20,
            signature="def authenticate(user: str, pwd: str) -> bool:",
            docstring="Authenticate a user with credentials.",
            visibility="public",
            category="implementation",
        )

        text = generate_symbol_text(symbol, "auth/service.py")

        assert "# File: auth/service.py" in text
        assert "# Symbol: authenticate (function)" in text
        assert "def authenticate(user: str, pwd: str) -> bool:" in text
        assert "Authenticate a user with credentials." in text

    def test_generate_symbol_text_method_with_parent(self):
        """Test text generation for a method with parent class."""
        symbol = Symbol(
            symbol_name="get_user",
            symbol_type="method",
            line_start=50,
            line_end=55,
            signature="def get_user(self, user_id: int):",
            docstring="Get user by ID.",
            parent_symbol="UserService",
            visibility="public",
            category="implementation",
        )

        text = generate_symbol_text(symbol, "services/user.py")

        assert "# File: services/user.py" in text
        assert "# Symbol: get_user (method)" in text
        assert "# Parent: UserService" in text
        assert "def get_user(self, user_id: int):" in text

    def test_generate_symbol_text_no_docstring(self):
        """Test text generation without docstring."""
        symbol = Symbol(
            symbol_name="helper",
            symbol_type="function",
            line_start=1,
            line_end=3,
            signature="def helper():",
            visibility="internal",
            category="implementation",
        )

        text = generate_symbol_text(symbol, "utils.py")

        assert "# File: utils.py" in text
        assert "def helper():" in text
        # Should not have docstring section
        assert '"""' not in text


class TestSymbolIndexingUnit:
    """Unit tests for symbol indexing functions."""

    @patch("src.indexer.symbol_indexing.get_connection")
    @patch("src.indexer.symbol_indexing.get_provider")
    @patch("src.indexer.symbol_indexing.extract_symbols")
    def test_index_file_symbols_success(self, mock_extract, mock_provider, mock_conn):
        """Test successful symbol indexing for a file."""
        # Mock symbol extraction
        mock_symbols = [
            Symbol(
                symbol_name="test_func",
                symbol_type="function",
                line_start=1,
                line_end=5,
                signature="def test_func():",
                visibility="public",
                category="test",
            )
        ]
        mock_extract.return_value = mock_symbols

        # Mock embedding provider
        mock_embed = Mock()
        mock_embed.get_embeddings_batch.return_value = [[0.1] * 1024]
        mock_provider.return_value = mock_embed

        # Mock database
        mock_cursor = Mock()
        mock_conn_instance = Mock()
        mock_conn_instance.__enter__ = Mock(return_value=mock_conn_instance)
        mock_conn_instance.__exit__ = Mock(return_value=None)
        mock_conn_instance.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn_instance.cursor.return_value.__exit__ = Mock(return_value=None)
        mock_conn.return_value = mock_conn_instance

        # Run indexing
        count, success = index_file_symbols("test_repo", "test.py", "def test_func(): pass", mock_embed)

        # Verify
        assert count == 1
        assert success is True
        mock_extract.assert_called_once()
        mock_embed.get_embeddings_batch.assert_called_once()
        mock_cursor.execute.assert_called_once()

    @patch("src.indexer.symbol_indexing.get_connection")
    def test_delete_file_symbols(self, mock_conn):
        """Test deleting symbols for a file."""
        # Mock database
        mock_cursor = Mock()
        mock_cursor.rowcount = 3
        mock_conn_instance = Mock()
        mock_conn_instance.__enter__ = Mock(return_value=mock_conn_instance)
        mock_conn_instance.__exit__ = Mock(return_value=None)
        mock_conn_instance.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn_instance.cursor.return_value.__exit__ = Mock(return_value=None)
        mock_conn.return_value = mock_conn_instance

        # Delete symbols
        count = delete_file_symbols("test_repo", "old_file.py")

        # Verify
        assert count == 3
        mock_cursor.execute.assert_called_once()

    @patch("src.indexer.symbol_indexing.extract_symbols")
    def test_index_file_symbols_no_symbols(self, mock_extract):
        """Test indexing when no symbols are found."""
        mock_extract.return_value = []

        count, success = index_file_symbols("test_repo", "empty.py", "# Just a comment")

        assert count == 0
        assert success is True
        mock_extract.assert_called_once()

    @patch("src.indexer.symbol_indexing.extract_symbols")
    @patch("src.indexer.symbol_indexing.get_language_from_file")
    def test_index_file_symbols_unknown_language(self, mock_lang, mock_extract):
        """Test indexing with unknown language."""
        mock_lang.return_value = None

        count, success = index_file_symbols("test_repo", "unknown.xyz", "content")

        assert count == 0
        assert success is True
        mock_extract.assert_not_called()


class TestIncrementalIndexing:
    """Test incremental symbol indexing scenarios."""

    @patch("src.indexer.symbol_indexing.get_connection")
    @patch("src.indexer.symbol_indexing.get_provider")
    @patch("src.indexer.symbol_indexing.extract_symbols")
    def test_update_file_symbols(self, mock_extract, mock_provider, mock_conn):
        """Test updating symbols when file changes."""
        # Scenario: File initially has 2 symbols, then changes to have 3 different symbols

        # Mock initial symbols
        initial_symbols = [
            Symbol("old_func1", "function", 1, 5, "def old_func1():", category="implementation"),
            Symbol("old_func2", "function", 7, 10, "def old_func2():", category="implementation"),
        ]

        # Mock new symbols after file change
        new_symbols = [
            Symbol("new_func1", "function", 1, 5, "def new_func1():", category="implementation"),
            Symbol("new_func2", "function", 7, 10, "def new_func2():", category="implementation"),
            Symbol("new_func3", "function", 12, 15, "def new_func3():", category="implementation"),
        ]

        # Mock embedding provider
        mock_embed = Mock()
        mock_embed.get_embeddings_batch.return_value = [[0.1] * 1024] * 3
        mock_provider.return_value = mock_embed

        # Mock database
        mock_cursor = Mock()
        mock_conn_instance = Mock()
        mock_conn_instance.__enter__ = Mock(return_value=mock_conn_instance)
        mock_conn_instance.__exit__ = Mock(return_value=None)
        mock_conn_instance.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn_instance.cursor.return_value.__exit__ = Mock(return_value=None)
        mock_conn.return_value = mock_conn_instance

        # First: delete old symbols
        mock_cursor.rowcount = 2
        deleted = delete_file_symbols("test_repo", "changed.py")
        assert deleted == 2

        # Then: index new symbols
        mock_extract.return_value = new_symbols
        count, success = index_file_symbols("test_repo", "changed.py", "new content", mock_embed)
        assert count == 3
        assert success is True

    @patch("src.indexer.symbol_indexing.get_connection")
    @patch("src.indexer.symbol_indexing.get_provider")
    @patch("src.indexer.symbol_indexing.extract_symbols")
    def test_symbol_upsert_on_conflict(self, mock_extract, mock_provider, mock_conn):
        """Test that ON CONFLICT DO UPDATE works for symbol updates."""
        # Scenario: Same symbol name/file/line_start but updated signature/docstring

        mock_symbols = [
            Symbol(
                symbol_name="updated_func",
                symbol_type="function",
                line_start=10,
                line_end=20,
                signature="def updated_func(new_param: str):",  # Updated signature
                docstring="Updated docstring",  # Updated docstring
                category="implementation",
            )
        ]
        mock_extract.return_value = mock_symbols

        # Mock embedding
        mock_embed = Mock()
        mock_embed.get_embeddings_batch.return_value = [[0.2] * 1024]
        mock_provider.return_value = mock_embed

        # Mock database
        mock_cursor = Mock()
        mock_conn_instance = Mock()
        mock_conn_instance.__enter__ = Mock(return_value=mock_conn_instance)
        mock_conn_instance.__exit__ = Mock(return_value=None)
        mock_conn_instance.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn_instance.cursor.return_value.__exit__ = Mock(return_value=None)
        mock_conn.return_value = mock_conn_instance

        # Index (will upsert due to unique constraint)
        count, success = index_file_symbols("test_repo", "file.py", "content", mock_embed)

        assert count == 1
        assert success is True
        # Verify INSERT ... ON CONFLICT DO UPDATE was called
        call_args = mock_cursor.execute.call_args
        assert call_args is not None
        sql_query = str(call_args[0][0])
        assert "ON CONFLICT" in sql_query
        assert "DO UPDATE" in sql_query


class TestBatchProcessing:
    """Test batch processing of multiple files."""

    @patch("src.indexer.symbol_indexing.Path")
    @patch("src.retrieval.graph_cache.create_graph_cache_table")
    @patch("src.indexer.symbol_indexing.create_edges_table")
    @patch("src.indexer.symbol_indexing.create_symbols_table")
    @patch("src.indexer.symbol_indexing.get_provider")
    @patch("src.indexer.symbol_indexing.index_file_symbols")
    def test_repository_indexing_disabled(self, mock_index, mock_provider, mock_create_symbols, mock_create_edges, mock_create_cache, mock_path):
        """Test that indexing is skipped when disabled."""
        from src.indexer.symbol_indexing import index_repository_symbols

        with patch("src.indexer.symbol_indexing.settings") as mock_settings:
            mock_settings.enable_symbol_indexing = False

            result = index_repository_symbols("test_repo", "/path/to/repo")

            assert result["files_processed"] == 0
            assert result["symbols_indexed"] == 0
            mock_create_symbols.assert_not_called()
            mock_index.assert_not_called()

    @patch("src.indexer.symbol_indexing.Path")
    @patch("src.retrieval.graph_cache.create_graph_cache_table")
    @patch("src.indexer.symbol_indexing.create_edges_table")
    @patch("src.indexer.symbol_indexing.create_symbols_table")
    @patch("src.indexer.symbol_indexing.get_provider")
    @patch("src.indexer.symbol_indexing.index_file_symbols")
    def test_repository_indexing_error_handling(self, mock_index, mock_provider, mock_create_symbols, mock_create_edges, mock_create_cache, mock_path):
        """Test error handling during batch processing."""
        from src.indexer.symbol_indexing import index_repository_symbols

        # Mock file discovery
        mock_file1 = Mock()
        mock_file1.is_file.return_value = True
        mock_file1.read_text.return_value = "content1"
        mock_file1.relative_to.return_value = "file1.py"

        mock_file2 = Mock()
        mock_file2.is_file.return_value = True
        mock_file2.read_text.side_effect = Exception("Read error")
        mock_file2.relative_to.return_value = "file2.py"

        mock_repo_path = Mock()
        mock_repo_path.glob.return_value = [mock_file1, mock_file2]
        mock_path.return_value = mock_repo_path

        mock_index.return_value = (5, True)

        with patch("src.indexer.symbol_indexing.settings") as mock_settings:
            mock_settings.enable_symbol_indexing = True
            mock_settings.included_extensions = [".py"]
            mock_settings.excluded_patterns = []

            result = index_repository_symbols("test_repo", "/path/to/repo")

            # file1 succeeds, file2 errors
            assert result["files_processed"] == 1
            assert result["symbols_indexed"] == 5
            assert result["errors"] == 1


class TestSymbolCategorization:
    """Test symbol category detection."""

    def test_test_function_category(self):
        """Test that test functions are categorized correctly."""
        from src.parser.symbol_extractor import extract_symbols

        code = """
def test_authentication():
    assert True
"""
        symbols = extract_symbols(code, "python", "test_auth.py")

        assert len(symbols) == 1
        assert symbols[0].category == "test"

    def test_implementation_function_category(self):
        """Test that regular functions are implementation."""
        from src.parser.symbol_extractor import extract_symbols

        code = """
def authenticate(user, pwd):
    return True
"""
        symbols = extract_symbols(code, "python", "auth.py")

        assert len(symbols) == 1
        assert symbols[0].category == "implementation"

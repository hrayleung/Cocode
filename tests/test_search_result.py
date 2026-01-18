"""Tests for SearchResult dataclass - TDD for extracting to shared module."""

import pytest


class TestSearchResultLocation:
    """Test that SearchResult is importable from a canonical location."""

    def test_search_result_importable_from_models(self):
        """SearchResult should be importable from src.models."""
        from src.models import SearchResult
        
        result = SearchResult(
            filename="test.py",
            location="1-10",
            content="def foo(): pass",
            score=0.95,
        )
        assert result.filename == "test.py"
        assert result.score == 0.95

    def test_search_result_backward_compatible_import(self):
        """SearchResult should still be importable from vector_search for compatibility."""
        from src.retrieval.vector_search import SearchResult
        
        result = SearchResult(
            filename="test.py",
            location="1-10",
            content="def foo(): pass",
            score=0.95,
        )
        assert result.filename == "test.py"

    def test_search_result_same_class_from_both_imports(self):
        """Both import paths should resolve to the same class."""
        from src.models import SearchResult as SR1
        from src.retrieval.vector_search import SearchResult as SR2
        
        assert SR1 is SR2

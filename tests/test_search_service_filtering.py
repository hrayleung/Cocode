"""Tests for SearchService score filtering behavior."""

import pytest

from src.models import SearchResult
from src.retrieval.service import SearchService


def test_search_service_does_not_filter_away_requested_file_count(monkeypatch):
    """Regression: avoid returning a single file when more are available."""

    def fake_hybrid_search(*_args, **_kwargs):
        return [
            SearchResult(filename="a.py", location="1:1", content="a", score=0.9),
            SearchResult(filename="b.py", location="1:1", content="b", score=0.1),
            SearchResult(filename="c.py", location="1:1", content="c", score=0.1),
        ]

    monkeypatch.setattr("src.retrieval.service.hybrid_search", fake_hybrid_search)

    results = SearchService().search(repo_name="repo", query="q", top_k=3, expand_related=False)

    assert [r.filename for r in results] == ["a.py", "b.py", "c.py"]


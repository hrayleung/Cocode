"""Tests for hybrid search failure handling."""

import pytest

from src.exceptions import SearchError
import src.retrieval.hybrid as hybrid


@pytest.mark.parametrize("parallel", [True, False])
def test_hybrid_search_raises_when_vector_and_bm25_fail_and_symbols_disabled(monkeypatch, parallel):
    """Regression test: don't mask total failure when symbol indexing is disabled."""
    monkeypatch.setattr(hybrid.settings, "enable_symbol_indexing", False, raising=False)

    def boom(*_args, **_kwargs):
        raise RuntimeError("backend down")

    def should_not_call(*_args, **_kwargs):
        raise AssertionError("symbol_hybrid_search should not be called when symbol indexing is disabled")

    # Avoid real embedding/provider calls.
    monkeypatch.setattr(hybrid, "get_query_embedding", lambda _q: [0.0])
    monkeypatch.setattr(hybrid, "vector_search", boom)
    monkeypatch.setattr(hybrid, "bm25_search", boom)
    monkeypatch.setattr(hybrid, "symbol_hybrid_search", should_not_call)

    with pytest.raises(SearchError):
        hybrid.hybrid_search("repo", "query", parallel=parallel)

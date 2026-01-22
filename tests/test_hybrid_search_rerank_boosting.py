"""Tests for hybrid_search behavior when Cohere reranking is enabled."""

from __future__ import annotations

from config.settings import settings
from src.models import SearchResult


def test_hybrid_search_reapplies_category_boost_after_rerank(monkeypatch):
    """Reranking replaces scores; we should still prefer implementation over docs."""

    import src.retrieval.hybrid as hybrid

    # Avoid any DB-based centrality lookups for this unit test.
    monkeypatch.setattr(settings, "centrality_weight", 0.0)

    # Force reranker path.
    monkeypatch.setattr(settings, "cohere_api_key", "dummy")

    # Avoid embedding provider calls.
    monkeypatch.setattr(hybrid, "get_query_embedding", lambda _q: [0.0] * settings.embedding_dimensions)

    doc = SearchResult(filename="CLAUDE.md", location="", content="Incremental indexing docs", score=0.9)
    impl = SearchResult(filename="src/indexer/service.py", location="", content="def ensure_indexed(...)", score=0.8)

    # Make both backends return doc first so reranker will see doc.
    monkeypatch.setattr(hybrid, "vector_search", lambda *_a, **_k: [doc, impl])
    monkeypatch.setattr(hybrid, "bm25_search", lambda *_a, **_k: [doc, impl])

    # Simulate Cohere preferring docs.
    monkeypatch.setattr(
        hybrid,
        "rerank_results",
        lambda _q, _results, top_n: [
            SearchResult(filename=doc.filename, location="", content=doc.content, score=0.9),
            SearchResult(filename=impl.filename, location="", content=impl.content, score=0.8),
        ][:top_n],
    )

    results = hybrid.hybrid_search(
        repo_name="repo",
        query="How does incremental indexing work?",
        top_k=2,
        use_reranker=True,
        include_symbols=False,
        parallel=False,
        rerank_candidates=2,
    )

    assert [r.filename for r in results] == ["src/indexer/service.py", "CLAUDE.md"]

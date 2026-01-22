"""Tests for curated retrieval section packaging."""

from __future__ import annotations

from config.settings import settings
from src.models import SearchResult
from src.retrieval.curation import curate_code_sections, merge_line_ranges
from src.retrieval.symbol_implementation import SymbolMatch


def test_merge_line_ranges_merges_overlapping_and_nearby():
    assert merge_line_ranges([(1, 5), (6, 9)], gap=0) == [(1, 5), (6, 9)]
    assert merge_line_ranges([(1, 5), (6, 9)], gap=1) == [(1, 9)]
    assert merge_line_ranges([(10, 12), (1, 3), (4, 8)], gap=1) == [(1, 8), (10, 12)]
    assert merge_line_ranges([(10, 12), (1, 3), (4, 8)], gap=2) == [(1, 12)]


def test_curate_code_sections_prefers_symbols_and_respects_budget(monkeypatch, tmp_path):
    # Build a tiny repo structure.
    (tmp_path / "src" / "indexer").mkdir(parents=True)
    (tmp_path / "tests").mkdir()

    service = tmp_path / "src" / "indexer" / "service.py"
    service.write_text("\n".join([f"line {i}" for i in range(1, 201)]) + "\n", encoding="utf-8")

    readme = tmp_path / "README.md"
    readme.write_text("\n".join([f"doc {i}" for i in range(1, 101)]) + "\n", encoding="utf-8")

    testfile = tmp_path / "tests" / "test_something.py"
    testfile.write_text("\n".join([f"test {i}" for i in range(1, 50)]) + "\n", encoding="utf-8")

    # Avoid any external embedding calls.
    monkeypatch.setattr(
        "src.retrieval.curation.get_query_embedding",
        lambda _q: [0.0] * settings.embedding_dimensions,
    )

    # File ranking returns impl + doc + test; curation should drop tests by default.
    def fake_hybrid_search(*_args, **_kwargs):
        return [
            SearchResult(filename="src/indexer/service.py", location="[0, 30)", content="", score=1.0),
            SearchResult(filename="README.md", location="[0, 30)", content="", score=0.9),
            SearchResult(filename="tests/test_something.py", location="[0, 30)", content="", score=0.8),
        ]

    monkeypatch.setattr("src.retrieval.curation.hybrid_search", fake_hybrid_search)
    monkeypatch.setattr("src.retrieval.curation.expand_results_with_related", lambda *_a, **_k: [])

    # Symbol hits only for service.py.
    def fake_symbol_search(*_args, **_kwargs):
        return [
            SymbolMatch(
                filename="src/indexer/service.py",
                symbol_name="_incremental_update",
                symbol_type="method",
                line_start=20,
                line_end=60,
                signature="def _incremental_update(...):",
                docstring=None,
                parent_symbol="IndexerService",
                visibility="internal",
                category="implementation",
                score=0.02,
            )
        ]

    monkeypatch.setattr("src.retrieval.curation.symbol_hybrid_search_with_metadata", fake_symbol_search)

    # Chunk fallback for README.
    def fake_vector_search(*_args, **_kwargs):
        return [SearchResult(filename="README.md", location="[0, 120)", content="", score=0.5)]

    def fake_bm25_search(*_args, **_kwargs):
        return [SearchResult(filename="README.md", location="[0, 120)", content="", score=0.4)]

    monkeypatch.setattr("src.retrieval.curation.vector_search", fake_vector_search)
    monkeypatch.setattr("src.retrieval.curation.bm25_search", fake_bm25_search)

    sections = curate_code_sections(
        repo_name="repo",
        repo_path=tmp_path,
        query="How does incremental indexing work?",
        max_output_chars=180,
        max_files=4,
        max_sections=5,
        include_docs=True,
    )

    assert sections, "should return at least one section"

    # Test files should not appear when query doesn't mention tests.
    assert all(not s["filename"].startswith("tests/") for s in sections)

    # Symbol-first: service should come before README.
    assert sections[0]["filename"] == "src/indexer/service.py"

    # Budget should be respected (allowing minor overhead differences).
    total = sum(len(s["content"]) for s in sections)
    assert total <= 180

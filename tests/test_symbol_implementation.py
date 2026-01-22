"""Unit tests for symbol implementation retrieval helpers."""

import pytest

from src.retrieval.symbol_implementation import (
    SymbolMatch,
    extract_symbol_code,
    reciprocal_rank_fusion_symbols,
    select_top_symbols,
)


def test_extract_symbol_code_exact_range(tmp_path):
    (tmp_path / "a.py").write_text("l1\nl2\nl3\nl4\n", encoding="utf-8")

    info = extract_symbol_code(
        repo_path=tmp_path,
        filename="a.py",
        line_start=2,
        line_end=3,
        max_code_chars=None,
    )

    assert info["code"] == "l2\nl3\n"
    assert info["extracted_line_start"] == 2
    assert info["extracted_line_end"] == 3
    assert info["file_line_count"] == 4
    assert info["truncated"] is False


def test_extract_symbol_code_clamps_end_of_file(tmp_path):
    (tmp_path / "a.py").write_text("l1\nl2\n", encoding="utf-8")

    info = extract_symbol_code(
        repo_path=tmp_path,
        filename="a.py",
        line_start=1,
        line_end=999,
        max_code_chars=None,
    )

    assert info["code"] == "l1\nl2\n"
    assert info["extracted_line_end"] == 2
    # Requested beyond EOF should be marked as truncated/mismatched.
    assert info["truncated"] is True


def test_extract_symbol_code_respects_max_code_chars(tmp_path):
    (tmp_path / "a.py").write_text("".join([f"line{i}\n" for i in range(1, 11)]), encoding="utf-8")

    info = extract_symbol_code(
        repo_path=tmp_path,
        filename="a.py",
        line_start=1,
        line_end=10,
        max_code_chars=12,
    )

    assert info["code"]  # non-empty
    assert len(info["code"]) <= 12
    assert info["truncated"] is True
    assert info["extracted_line_end"] >= 1
    assert info["extracted_line_end"] < 10


def test_extract_symbol_code_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError):
        extract_symbol_code(
            repo_path=tmp_path,
            filename="../outside.py",
            line_start=1,
            line_end=1,
            max_code_chars=1000,
        )


def test_select_top_symbols_enforces_per_file_limit():
    hits = [
        SymbolMatch(
            filename="a.py",
            symbol_name=f"f{i}",
            symbol_type="function",
            line_start=1,
            line_end=2,
            signature=None,
            docstring=None,
            parent_symbol=None,
            visibility=None,
            category=None,
            score=1.0 - i * 0.01,
        )
        for i in range(10)
    ]

    selected = select_top_symbols(hits, max_symbols=10, max_symbols_per_file=3)

    assert len(selected) == 3
    assert all(s.filename == "a.py" for s in selected)


def test_reciprocal_rank_fusion_symbols_dedupes_same_symbol_key():
    s1 = SymbolMatch(
        filename="a.py",
        symbol_name="foo",
        symbol_type="function",
        line_start=1,
        line_end=3,
        signature=None,
        docstring=None,
        parent_symbol=None,
        visibility=None,
        category=None,
        score=0.9,
    )
    s2 = SymbolMatch(
        filename="a.py",
        symbol_name="foo",
        symbol_type="function",
        line_start=1,
        line_end=3,
        signature=None,
        docstring=None,
        parent_symbol=None,
        visibility=None,
        category=None,
        score=0.1,
    )

    fused = reciprocal_rank_fusion_symbols([[s1], [s2]])

    assert len(fused) == 1
    assert fused[0].symbol_name == "foo"
    # RRF score should not equal the original raw score.
    assert fused[0].score != s1.score

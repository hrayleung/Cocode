"""Python wrapper for Rust utility functions."""

from cocode_rust import (
    compute_file_hash as _rust_compute_file_hash,
    jaccard_similarity as _rust_jaccard_similarity,
    jaccard_similarity_batch as _rust_jaccard_similarity_batch,
    mmr_select_indices as _rust_mmr_select_indices,
    extract_code_by_line_range as _rust_extract_code_by_line_range,
)


def compute_file_hash(content: str) -> str:
    """Compute a truncated SHA256 hash for file content."""
    return _rust_compute_file_hash(content)


def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two strings (whitespace-token level)."""
    return float(_rust_jaccard_similarity(text1, text2))


def jaccard_similarity_batch(query: str, texts: list[str]) -> list[float]:
    """Compute Jaccard similarity of one query against many texts."""
    return [float(x) for x in _rust_jaccard_similarity_batch(query, texts)]


def mmr_select_indices(
    scores: list[float],
    contents: list[str],
    target_count: int,
    lambda_param: float = 0.7,
) -> list[int]:
    """Select indices using Maximal Marginal Relevance (MMR)."""
    return list(_rust_mmr_select_indices(scores, contents, int(target_count), float(lambda_param)))


def extract_code_by_line_range(
    repo_path: str,
    filename: str,
    line_start: int,
    line_end: int,
    max_code_chars: int | None = None,
) -> dict:
    """Extract a line range from disk, returning a dict compatible with symbol extraction."""
    return _rust_extract_code_by_line_range(
        repo_path,
        filename,
        int(line_start),
        int(line_end),
        max_code_chars,
    )

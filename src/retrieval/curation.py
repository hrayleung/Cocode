"""Curated context packaging for LLM-friendly retrieval.

Augment-style retrieval is less about returning many ranked files and more about
returning a small set of *code sections* that are:
- highly relevant (entry points + key helpers)
- contiguous (line ranges)
- bounded by an output budget

This module builds those sections by combining:
- file ranking from hybrid_search (dense + sparse)
- symbol-level matches (preferred for functions/methods)
- chunk-level matches as fallback
- optional graph expansion to pull in adjacent dependencies

The output is designed for MCP tools to return directly.
"""

from __future__ import annotations

import bisect
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from config.settings import settings
from src.retrieval.file_categorizer import categorize_file
from src.retrieval.graph_expansion import expand_results_with_related
from src.retrieval.hybrid import get_query_embedding, hybrid_search
from src.retrieval.symbol_implementation import extract_symbol_code, symbol_hybrid_search_with_metadata
from src.retrieval.vector_search import vector_search
from src.retrieval.bm25_search import bm25_search

logger = logging.getLogger(__name__)

# Conservative defaults inspired by Augment's SDK defaults.
DEFAULT_MAX_OUTPUT_CHARS = 20_000
DEFAULT_MAX_FILES = 4
DEFAULT_MAX_SECTIONS = 8
DEFAULT_MAX_SECTION_CHARS = 8_000

# Merge nearby spans to avoid fragmentation.
DEFAULT_MERGE_GAP_LINES = 4

# Prefer smaller symbols by default to avoid dumping huge classes.
DEFAULT_MAX_SYMBOL_LINES = 220

# Chunk fallback: expand around the matched span a bit.
DEFAULT_CONTEXT_LINES = 10

_CHUNK_LOC_RE = re.compile(r"\[(\d+),\s*(\d+)\)")


@dataclass(frozen=True)
class LineSpan:
    filename: str
    line_start: int
    line_end: int
    score: float
    source: str  # 'symbol' | 'chunk' | 'doc'
    label: str | None = None


def _query_mentions_tests(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in ("test", "tests", "pytest", "unit test", "integration test"))


def _parse_chunk_location(location: str) -> tuple[int, int] | None:
    """Parse a chunk location like "[123, 456)" into byte offsets."""
    m = _CHUNK_LOC_RE.match(location.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _get_newlines_for_file(
    *,
    repo_path: str | Path,
    filename: str,
    file_cache: dict[str, bytes],
    newline_cache: dict[str, list[int]],
) -> list[int] | None:
    repo_root = Path(repo_path)
    path = repo_root / filename
    try:
        if filename not in file_cache:
            file_cache[filename] = path.read_bytes()
        if filename not in newline_cache:
            newline_cache[filename] = [i for i, b in enumerate(file_cache[filename]) if b == 10]
        return newline_cache[filename]
    except Exception:
        return None


def chunk_location_to_line_range(
    *,
    repo_path: str | Path,
    filename: str,
    location: str,
    file_cache: dict[str, bytes],
    newline_cache: dict[str, list[int]],
) -> tuple[int, int] | None:
    """Convert a chunk location string into (line_start, line_end)."""
    byte_range = _parse_chunk_location(location)
    if not byte_range:
        return None
    start, end = byte_range

    newlines = _get_newlines_for_file(
        repo_path=repo_path,
        filename=filename,
        file_cache=file_cache,
        newline_cache=newline_cache,
    )
    if not newlines:
        return None

    start_line = bisect.bisect_right(newlines, max(start, 0) - 1) + 1
    end_line = bisect.bisect_right(newlines, max(end - 1, start) - 1) + 1
    if end_line < start_line:
        end_line = start_line
    return start_line, end_line


def merge_line_ranges(ranges: list[tuple[int, int]], *, gap: int = DEFAULT_MERGE_GAP_LINES) -> list[tuple[int, int]]:
    """Merge overlapping/nearby line ranges."""
    if not ranges:
        return []
    sorted_ranges = sorted(ranges)
    merged: list[tuple[int, int]] = []
    cur_s, cur_e = sorted_ranges[0]

    for s, e in sorted_ranges[1:]:
        if s <= cur_e + gap:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e

    merged.append((cur_s, cur_e))
    return merged


def _prioritize_files(
    files: list[str],
    *,
    include_docs: bool,
    include_tests: bool,
    max_files: int,
) -> list[str]:
    """Select a small set of files, biased towards implementation."""
    out: list[str] = []
    has_doc = False

    for f in files:
        if len(out) >= max_files:
            break
        cat = categorize_file(f)
        if cat == "test" and not include_tests:
            continue
        if cat == "documentation":
            if not include_docs or has_doc:
                continue
            has_doc = True
        out.append(f)

    return out


def _select_symbol_spans(
    *,
    repo_name: str,
    repo_path: str | Path,
    query: str,
    filenames: list[str],
    query_embedding: list[float] | None,
    max_sections: int,
    max_per_file: int,
    max_symbol_lines: int,
) -> list[LineSpan]:
    if not filenames or max_sections <= 0:
        return []

    hits = symbol_hybrid_search_with_metadata(
        repo_name,
        query,
        top_k=min(max_sections * 10, 200),
        filenames=filenames,
        query_embedding=query_embedding,
    )

    if not hits:
        return []

    # Prefer functions/methods (avoid huge class dumps) but fall back if needed.
    preferred = [h for h in hits if h.symbol_type in ("function", "method")]
    if preferred:
        hits = preferred

    # Filter out very large symbols if we have enough remaining.
    filtered = [h for h in hits if (h.line_end - h.line_start + 1) <= max_symbol_lines]
    if filtered:
        hits = filtered

    # Enforce per-file and global caps.
    spans: list[LineSpan] = []
    per_file: dict[str, int] = defaultdict(int)

    for h in hits:
        if len(spans) >= max_sections:
            break
        if per_file[h.filename] >= max_per_file:
            continue
        per_file[h.filename] += 1
        label = h.signature or h.symbol_name
        spans.append(
            LineSpan(
                filename=h.filename,
                line_start=h.line_start,
                line_end=h.line_end,
                score=float(h.score),
                source="symbol",
                label=label,
            )
        )

    return spans


def _select_chunk_spans(
    *,
    repo_name: str,
    repo_path: str | Path,
    query: str,
    filenames: list[str],
    query_embedding: list[float] | None,
    max_sections: int,
    max_per_file: int,
    context_lines: int,
) -> list[LineSpan]:
    if not filenames or max_sections <= 0:
        return []

    # Pull a small pool of chunk matches then map to line ranges.
    vec = []
    try:
        vec = vector_search(repo_name, query, top_k=60, query_embedding=query_embedding)
    except Exception as e:
        logger.debug(f"Vector search unavailable for curation chunk fallback: {e}")

    bm25 = []
    try:
        bm25 = bm25_search(repo_name, query, top_k=60)
    except Exception as e:
        logger.debug(f"BM25 search unavailable for curation chunk fallback: {e}")

    candidates = vec + bm25

    file_cache: dict[str, bytes] = {}
    newline_cache: dict[str, list[int]] = {}

    spans: list[LineSpan] = []
    per_file: dict[str, int] = defaultdict(int)

    for r in candidates:
        if len(spans) >= max_sections:
            break
        if r.filename not in filenames:
            continue
        if per_file[r.filename] >= max_per_file:
            continue
        lr = chunk_location_to_line_range(
            repo_path=repo_path,
            filename=r.filename,
            location=r.location,
            file_cache=file_cache,
            newline_cache=newline_cache,
        )
        if not lr:
            continue
        ls, le = lr
        ls = max(1, ls - context_lines)
        le = max(ls, le + context_lines)
        per_file[r.filename] += 1
        spans.append(
            LineSpan(
                filename=r.filename,
                line_start=ls,
                line_end=le,
                score=float(r.score),
                source="chunk",
                label=None,
            )
        )

    return spans


def curate_code_sections(
    *,
    repo_name: str,
    repo_path: str | Path,
    query: str,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    max_files: int = DEFAULT_MAX_FILES,
    max_sections: int = DEFAULT_MAX_SECTIONS,
    max_section_chars: int = DEFAULT_MAX_SECTION_CHARS,
    include_docs: bool = True,
    include_tests: bool | None = None,
    max_related_files: int = 2,
    max_symbol_lines: int = DEFAULT_MAX_SYMBOL_LINES,
) -> list[dict]:
    """Return a small set of code sections for the query.

    Output dict format is MCP-friendly and stable:
      {filename, lines, line_start, line_end, content, source, score}
    """
    if max_output_chars <= 0 or max_sections <= 0 or max_files <= 0:
        return []

    q = query.strip()
    if not q:
        return []

    query_embedding = None
    try:
        query_embedding = get_query_embedding(q)
    except Exception:
        # Still proceed with BM25/symbol BM25 fallbacks.
        query_embedding = None

    include_tests_resolved = _query_mentions_tests(q) if include_tests is None else include_tests

    # Rank files using chunk-based hybrid search (avoid symbol backend dominating file ranking).
    file_rank_results = hybrid_search(
        repo_name=repo_name,
        query=q,
        top_k=max(20, max_files * 5),
        include_symbols=False,
        use_reranker=bool(settings.cohere_api_key) and bool(settings.enable_reranker),
        query_embedding=query_embedding,
    )

    ranked_files: list[str] = []
    seen: set[str] = set()
    for r in file_rank_results:
        if r.filename in seen:
            continue
        seen.add(r.filename)
        ranked_files.append(r.filename)

    if not ranked_files:
        return []

    # Optionally pull in immediate dependencies of the top implementation file.
    top_impl = next((f for f in ranked_files if categorize_file(f) == "implementation"), ranked_files[0])
    related: list[str] = []
    try:
        related = expand_results_with_related(repo_name, [top_impl], max_expansion=max_related_files) or []
    except Exception as e:
        logger.debug(f"Graph expansion failed during curation: {e}")

    # Combine ranked + related, then pick a small set.
    combined = ranked_files + [f for f in related if f not in seen]
    selected_files = _prioritize_files(
        combined,
        include_docs=include_docs,
        include_tests=include_tests_resolved,
        max_files=max_files,
    )

    if not selected_files:
        return []

    # Symbol-first spans.
    spans: list[LineSpan] = []
    spans.extend(
        _select_symbol_spans(
            repo_name=repo_name,
            repo_path=repo_path,
            query=q,
            filenames=selected_files,
            query_embedding=query_embedding,
            max_sections=max_sections,
            max_per_file=2,
            max_symbol_lines=max_symbol_lines,
        )
    )

    # Chunk fallback spans for files not yet covered.
    covered = {s.filename for s in spans}
    remaining_files = [f for f in selected_files if f not in covered]
    remaining_slots = max(0, max_sections - len(spans))
    if remaining_slots > 0 and remaining_files:
        spans.extend(
            _select_chunk_spans(
                repo_name=repo_name,
                repo_path=repo_path,
                query=q,
                filenames=remaining_files,
                query_embedding=query_embedding,
                max_sections=remaining_slots,
                max_per_file=1,
                context_lines=DEFAULT_CONTEXT_LINES,
            )
        )

    if not spans:
        return []

    # Merge nearby spans per file to keep sections contiguous.
    by_file: dict[str, list[LineSpan]] = defaultdict(list)
    for s in spans:
        by_file[s.filename].append(s)

    merged_spans: list[LineSpan] = []
    for filename in selected_files:
        file_spans = by_file.get(filename, [])
        if not file_spans:
            continue
        ranges = [(s.line_start, s.line_end) for s in file_spans]
        merged = merge_line_ranges(ranges, gap=DEFAULT_MERGE_GAP_LINES)
        # Use the best score among merged members.
        best_score = max((s.score for s in file_spans), default=0.0)
        source = file_spans[0].source
        for ls, le in merged:
            merged_spans.append(
                LineSpan(
                    filename=filename,
                    line_start=ls,
                    line_end=le,
                    score=best_score,
                    source=source,
                    label=None,
                )
            )

    # Enforce global budget while extracting code.
    remaining = max_output_chars
    sections: list[dict] = []

    for s in merged_spans:
        if remaining <= 0:
            break

        code_info = extract_symbol_code(
            repo_path=repo_path,
            filename=s.filename,
            line_start=s.line_start,
            line_end=s.line_end,
            max_code_chars=min(max_section_chars, remaining),
        )

        code = code_info.get("code", "")
        if not code:
            continue

        sections.append(
            {
                "filename": s.filename,
                "line_start": int(code_info["extracted_line_start"]),
                "line_end": int(code_info["extracted_line_end"]),
                "lines": [f"L{int(code_info['extracted_line_start'])}-{int(code_info['extracted_line_end'])}"],
                "content": code,
                "source": s.source,
                "score": round(float(s.score), 6),
            }
        )
        remaining -= len(code)

        if len(sections) >= max_sections:
            break

    return sections

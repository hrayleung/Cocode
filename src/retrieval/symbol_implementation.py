"""Symbol implementation retrieval.

This module complements semantic chunk retrieval by allowing callers to:
- search the indexed `symbols` table for the most relevant functions/classes
- return the *full implementation* by extracting the exact line range from disk

This is intended for MCP tools that need full code bodies (similar to Augment's
line-precise symbol results), without requiring a separate file-view tool.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

from psycopg import sql

from config.settings import settings
from src.storage.postgres import get_connection
from src.storage.schema import sanitize_repo_name

logger = logging.getLogger(__name__)


@dataclass
class SymbolMatch:
    """A symbol match from the symbols table."""

    filename: str
    symbol_name: str
    symbol_type: str
    line_start: int
    line_end: int
    signature: str | None
    docstring: str | None
    parent_symbol: str | None
    visibility: str | None
    category: str | None
    score: float

    def key(self) -> tuple:
        return (self.filename, self.symbol_name, self.symbol_type, self.line_start, self.line_end)

    @staticmethod
    def from_row(row: tuple) -> "SymbolMatch":
        """Create a SymbolMatch from a database row."""
        return SymbolMatch(
            filename=row[0],
            symbol_name=row[1],
            symbol_type=row[2],
            line_start=int(row[3]),
            line_end=int(row[4]),
            signature=row[5],
            docstring=row[6],
            parent_symbol=row[7],
            visibility=row[8],
            category=row[9],
            score=float(row[10]),
        )


def _normalize_weights(weights: list[float] | None, n: int) -> list[float]:
    if not weights:
        return [1.0] * n
    total = sum(weights)
    if total <= 0:
        return [1.0 / n] * n
    return [w / total for w in weights]


def _symbol_key(hit: SymbolMatch) -> str:
    """Build a stable key for RRF fusion."""
    return f"{hit.filename}:{hit.symbol_name}:{hit.symbol_type}:{hit.line_start}:{hit.line_end}"


def reciprocal_rank_fusion_symbols(
    result_lists: list[list[SymbolMatch]],
    *,
    k: int = 40,
    weights: list[float] | None = None,
) -> list[SymbolMatch]:
    """Fuse multiple ranked symbol lists using Rust-accelerated RRF."""

    if not result_lists:
        return []

    from src.rust_bridge import reciprocal_rank_fusion_weighted

    norm_weights = _normalize_weights(weights, len(result_lists))

    ranked_lists: list[list[tuple[str, float]]] = []
    best: dict[str, SymbolMatch] = {}

    for results in result_lists:
        out: list[tuple[str, float]] = []
        for hit in results:
            key = _symbol_key(hit)
            out.append((key, float(hit.score)))
            if key not in best or hit.score > best[key].score:
                best[key] = hit
        ranked_lists.append(out)

    fused_scores = reciprocal_rank_fusion_weighted(ranked_lists, norm_weights, k=float(k))

    fused: list[SymbolMatch] = []
    for key, fused_score in fused_scores:
        rep = best.get(key)
        if rep is None:
            continue
        fused.append(replace(rep, score=float(fused_score)))

    return fused


def symbol_vector_search(
    repo_name: str,
    query: str,
    *,
    top_k: int = 30,
    filenames: list[str] | None = None,
    query_embedding: list[float] | None = None,
) -> list[SymbolMatch]:
    """Vector similarity search over symbols."""

    if not settings.enable_symbol_indexing:
        return []

    if query_embedding is None:
        from src.retrieval.hybrid import get_query_embedding

        query_embedding = get_query_embedding(query)

    schema_name = sanitize_repo_name(repo_name)
    table = sql.Identifier(schema_name, "symbols")

    # Build query with optional filename filter
    filename_filter = sql.SQL(" AND filename = ANY(%s)") if filenames is not None else sql.SQL("")

    query_sql = sql.SQL(
        """
        SELECT
            filename,
            symbol_name,
            symbol_type,
            line_start,
            line_end,
            signature,
            docstring,
            parent_symbol,
            visibility,
            category,
            1 - (embedding <=> %s::vector) AS similarity
        FROM {table}
        WHERE embedding IS NOT NULL{filename_filter}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
    ).format(table=table, filename_filter=filename_filter)

    # Params order: similarity %s, [filename ANY(%s) if filter], ORDER BY %s, LIMIT %s
    if filenames is not None:
        params = [query_embedding, filenames, query_embedding, top_k]
    else:
        params = [query_embedding, query_embedding, top_k]

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query_sql, params)
            rows = cur.fetchall()

    return [SymbolMatch.from_row(row) for row in rows]


def symbol_bm25_search(
    repo_name: str,
    query: str,
    *,
    top_k: int = 30,
    filenames: list[str] | None = None,
) -> list[SymbolMatch]:
    """BM25 full-text search over symbols."""

    if not settings.enable_symbol_indexing:
        return []

    schema_name = sanitize_repo_name(repo_name)
    table = sql.Identifier(schema_name, "symbols")

    # Build query with optional filename filter
    filename_filter = sql.SQL(" AND filename = ANY(%s)") if filenames is not None else sql.SQL("")

    query_sql = sql.SQL(
        """
        SELECT
            filename,
            symbol_name,
            symbol_type,
            line_start,
            line_end,
            signature,
            docstring,
            parent_symbol,
            visibility,
            category,
            ts_rank_cd(content_tsv, q, 1) AS rank
        FROM {table}, plainto_tsquery('english', %s) q
        WHERE content_tsv @@ q{filename_filter}
        ORDER BY rank DESC
        LIMIT %s
        """
    ).format(table=table, filename_filter=filename_filter)

    if filenames is not None:
        params = [query, filenames, top_k]
    else:
        params = [query, top_k]

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query_sql, params)
            rows = cur.fetchall()

    return [SymbolMatch.from_row(row) for row in rows]


def symbol_hybrid_search_with_metadata(
    repo_name: str,
    query: str,
    *,
    top_k: int = 30,
    filenames: list[str] | None = None,
    query_embedding: list[float] | None = None,
    weights: list[float] | None = None,
) -> list[SymbolMatch]:
    """Hybrid symbol search (vector + BM25) returning metadata.

    This is intentionally resilient:
    - If vector search fails (e.g., embedding dimension mismatch), we still return BM25 matches.
    - If BM25 fails, we still return vector matches.
    """

    vec: list[SymbolMatch] = []
    bm25: list[SymbolMatch] = []

    try:
        vec = symbol_vector_search(
            repo_name,
            query,
            top_k=top_k * 2,
            filenames=filenames,
            query_embedding=query_embedding,
        )
    except Exception as e:
        logger.warning(f"Symbol vector search failed: {e}")

    try:
        bm25 = symbol_bm25_search(repo_name, query, top_k=top_k * 2, filenames=filenames)
    except Exception as e:
        logger.warning(f"Symbol BM25 search failed: {e}")

    if not vec and not bm25:
        return []

    fused = reciprocal_rank_fusion_symbols(
        [vec, bm25],
        weights=weights or [settings.vector_weight, settings.bm25_weight],
    )
    return fused[:top_k]


def select_top_symbols(
    hits: list[SymbolMatch],
    *,
    max_symbols: int,
    max_symbols_per_file: int,
) -> list[SymbolMatch]:
    """Select top symbols while limiting per-file concentration."""

    selected: list[SymbolMatch] = []
    per_file: dict[str, int] = defaultdict(int)

    for hit in hits:
        if len(selected) >= max_symbols:
            break
        if per_file[hit.filename] >= max_symbols_per_file:
            continue
        per_file[hit.filename] += 1
        selected.append(hit)

    return selected


def extract_symbol_code(
    *,
    repo_path: str | Path,
    filename: str,
    line_start: int,
    line_end: int,
    max_code_chars: int | None = 50_000,
) -> dict:
    """Extract exact symbol implementation text from disk by line range.

    Returns a dict with:
    - code
    - extracted_line_start / extracted_line_end
    - file_line_count
    - truncated

    This function intentionally does not fall back to Python.
    """

    from src.rust_bridge import extract_code_by_line_range

    return extract_code_by_line_range(
        str(repo_path),
        filename,
        line_start,
        line_end,
        max_code_chars,
    )

"""Search service - handles all code search operations."""

import bisect
import logging
import re
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from config.settings import settings
from src.exceptions import SearchError
from src.retrieval.hybrid import hybrid_search
from src.retrieval.graph_expansion import expand_results_with_related

logger = logging.getLogger(__name__)

MIN_SCORE_RATIO = 0.4
FULL_CODE_COUNT = 3
MAX_CHUNKS_PER_FILE = 3
MAX_FILE_CACHE_SIZE = 100
MAX_NEWLINE_CACHE_SIZE = 100

SIGNATURE_KEYWORDS = (
    "def ", "async def ", "class ", "function ", "const ", "let ",
    "var ", "fn ", "func ", "pub fn ", "impl ", "struct ", "enum ",
    "interface ", "type ", "export ", "@",
)
COMMENT_PREFIXES = ("#", "//", "/*", '"""', "'''")


def extract_signature(content: str, language: str = "python") -> str:
    """Extract function/class signature from code content."""
    lines = content.strip().split("\n")
    signatures = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(COMMENT_PREFIXES):
            continue
        if any(stripped.startswith(kw) for kw in SIGNATURE_KEYWORDS):
            signatures.append(stripped.split("{")[0].rstrip(" {:"))
            if len(signatures) >= 2:
                break

    if signatures:
        return "; ".join(signatures)

    for line in lines[:5]:
        stripped = line.strip()
        if stripped and not stripped.startswith(COMMENT_PREFIXES):
            return stripped[:100]

    return lines[0][:80] if lines else ""


def _format_line_range(start: int, end: int) -> str:
    """Format a line range as L{start}-{end} or L{start} for single lines."""
    return f"L{start}-{end}" if end > start else f"L{start}"


def parse_location(loc_str: str) -> str:
    """Convert location string to human-readable format."""
    loc = str(loc_str)

    # Symbol format: "line_start:line_end"
    if ':' in loc and '[' not in loc:
        parts = loc.split(':')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return _format_line_range(int(parts[0]), int(parts[1]))

    # Chunk format: "[start, end)"
    match = re.match(r'\[(\d+),\s*(\d+)\)', loc)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        start_line, end_line = start // 40 + 1, end // 40
        return f"~L{start_line}-{end_line}" if end_line > start_line else f"~L{start_line}"

    return loc


def _evict_oldest(cache: dict, max_size: int, name: str) -> None:
    """Evict oldest entry from cache if at capacity."""
    if len(cache) >= max_size:
        first_key = next(iter(cache))
        del cache[first_key]
        logger.debug(f"Evicted {name} cache entry for {first_key}")


def location_to_lines(
    filename: str,
    loc_str: str,
    repo_path: str | None,
    file_cache: dict[str, bytes],
    newline_cache: dict[str, list[int]],
) -> str:
    """Convert a location string into 1-indexed line ranges."""
    loc = str(loc_str)

    # Symbol format
    if ':' in loc and '[' not in loc:
        parts = loc.split(':')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return _format_line_range(int(parts[0]), int(parts[1]))

    # Chunk format
    match = re.match(r'\[(\d+),\s*(\d+)\)', loc)
    if not match or not repo_path:
        return parse_location(loc_str)

    start, end = int(match.group(1)), int(match.group(2))

    # Get or compute newline positions
    if filename not in newline_cache:
        try:
            _evict_oldest(newline_cache, MAX_NEWLINE_CACHE_SIZE, "newline")
            if filename not in file_cache:
                _evict_oldest(file_cache, MAX_FILE_CACHE_SIZE, "file")
                file_cache[filename] = (Path(repo_path) / filename).read_bytes()
            newline_cache[filename] = [i for i, b in enumerate(file_cache[filename]) if b == 10]
        except Exception:
            return parse_location(loc_str)

    newlines = newline_cache[filename]
    start_line = bisect.bisect_right(newlines, max(start, 0) - 1) + 1
    end_line = bisect.bisect_right(newlines, max(end - 1, start) - 1) + 1

    return _format_line_range(start_line, end_line)


@dataclass
class CodeSnippet:
    """A code snippet from search results."""
    filename: str
    content: str
    score: float
    locations: list[str] = field(default_factory=list)
    is_reference_only: bool = False

    def to_dict(self) -> dict:
        result = {"filename": self.filename, "score": round(self.score, 4)}
        if self.is_reference_only:
            result["reference"] = self.content
            result["lines"] = self.locations
        else:
            result["content"] = self.content
            result["locations"] = self.locations
        return result


class SearchService:
    """Service for searching indexed codebases."""

    def search(
        self,
        repo_name: str,
        query: str,
        top_k: int = 10,
        full_code_count: int = FULL_CODE_COUNT,
        expand_related: bool = True,
        max_related: int = 3,
    ) -> list[CodeSnippet]:
        """Search a repository using hybrid semantic + keyword search."""
        if not query or not query.strip():
            raise SearchError("Query cannot be empty")

        try:
            results = hybrid_search(
                repo_name=repo_name,
                query=query.strip(),
                top_k=top_k * 3,
                use_reranker=bool(settings.cohere_api_key) and bool(settings.enable_reranker),
            )
            if not results:
                return []

            repo_path = self._get_repo_path(repo_name)
            filtered = self._filter_by_score(results, top_k)
            file_results = self._aggregate_results(filtered, top_k, full_code_count, repo_path)

            if expand_related and file_results:
                self._add_related_files(repo_name, file_results, max_related)

            return file_results

        except Exception as e:
            logger.error(f"Search failed for {repo_name}: {e}")
            raise SearchError(f"Search failed: {e}") from e

    def _get_repo_path(self, repo_name: str) -> str | None:
        try:
            from src.indexer.repo_manager import RepoManager
            repo = RepoManager().get_repo(repo_name)
            return repo.path if repo else None
        except Exception:
            return None

    def _filter_by_score(self, results: list, top_k: int) -> list:
        """Apply adaptive score filtering."""
        top_score = results[0].score
        filtered = [r for r in results if r.score >= top_score * MIN_SCORE_RATIO]

        total_files = len({r.filename for r in results})
        if len({r.filename for r in filtered}) < min(top_k, total_files):
            return results
        return filtered

    def _add_related_files(self, repo_name: str, file_results: list[CodeSnippet], max_related: int) -> None:
        """Add related files via import graph expansion."""
        try:
            related = expand_results_with_related(
                repo_name, [r.filename for r in file_results], max_expansion=max_related
            )
            for filename in related or []:
                file_results.append(CodeSnippet(
                    filename=filename,
                    content="[Related via imports]",
                    score=0.0,
                    is_reference_only=True,
                ))
        except Exception as e:
            logger.debug(f"Graph expansion failed: {e}")

    def _aggregate_results(
        self,
        results: list,
        top_k: int,
        full_code_count: int,
        repo_path: str | None,
    ) -> list[CodeSnippet]:
        """Aggregate results with tiered presentation."""
        from src.retrieval.file_categorizer import categorize_file

        # Group chunks by file
        file_chunks: dict[str, list] = defaultdict(list)
        file_scores: dict[str, float] = defaultdict(float)
        file_categories: dict[str, str] = {}

        for r in results:
            file_chunks[r.filename].append(r)
            file_scores[r.filename] = max(file_scores[r.filename], r.score)
            if r.filename not in file_categories:
                file_categories[r.filename] = categorize_file(r.filename)

        # Select diverse files
        candidates = sorted(file_chunks, key=lambda f: file_scores[f], reverse=True)
        selected = self._mmr_diversify(candidates, file_scores, file_categories, top_k, settings.diversity_lambda)

        # Build snippets
        file_cache: dict[str, bytes] = {}
        newline_cache: dict[str, list[int]] = {}

        return [
            self._build_snippet(
                f, file_chunks[f], file_scores[f], repo_path, file_cache, newline_cache,
                full=(i < full_code_count)
            )
            for i, f in enumerate(selected)
        ]

    def _mmr_diversify(
        self,
        candidates: list[str],
        scores: dict[str, float],
        categories: dict[str, str],
        k: int,
        lambda_param: float,
    ) -> list[str]:
        """Maximal Marginal Relevance for diverse result selection."""
        selected: list[str] = []
        remaining = list(candidates)
        category_counts: dict[str, int] = defaultdict(int)

        while len(selected) < k and remaining:
            best_file, best_score = None, float('-inf')

            for candidate in remaining:
                divisor = max(len(selected), 1)
                penalty = category_counts[categories[candidate]] / divisor
                mmr = lambda_param * scores[candidate] - (1 - lambda_param) * penalty

                if mmr > best_score:
                    best_score, best_file = mmr, candidate

            if best_file:
                selected.append(best_file)
                remaining.remove(best_file)
                category_counts[categories[best_file]] += 1

        return selected

    def _build_snippet(
        self,
        filename: str,
        chunks: list,
        score: float,
        repo_path: str | None,
        file_cache: dict[str, bytes],
        newline_cache: dict[str, list[int]],
        full: bool,
    ) -> CodeSnippet:
        """Build a code snippet (full or reference) for a file."""
        if full:
            return self._build_full_snippet(filename, chunks, score, repo_path, file_cache, newline_cache)
        return self._build_reference_snippet(filename, chunks, score, repo_path, file_cache, newline_cache)

    def _build_full_snippet(
        self,
        filename: str,
        chunks: list,
        score: float,
        repo_path: str | None,
        file_cache: dict[str, bytes],
        newline_cache: dict[str, list[int]],
    ) -> CodeSnippet:
        """Build full code snippet for top results."""
        top_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)[:MAX_CHUNKS_PER_FILE]
        sorted_chunks = sorted(top_chunks, key=lambda c: c.location)

        content_parts, locations, seen = [], [], set()
        for chunk in sorted_chunks:
            h = hash(chunk.content.strip())
            if h not in seen:
                seen.add(h)
                content_parts.append(chunk.content)
                if chunk.location:
                    locations.append(location_to_lines(filename, str(chunk.location), repo_path, file_cache, newline_cache))

        return CodeSnippet(filename=filename, content="\n\n".join(content_parts), score=score, locations=locations)

    def _build_reference_snippet(
        self,
        filename: str,
        chunks: list,
        score: float,
        repo_path: str | None,
        file_cache: dict[str, bytes],
        newline_cache: dict[str, list[int]],
    ) -> CodeSnippet:
        """Build compact reference for lower-relevance results."""
        best = max(chunks, key=lambda c: c.score)
        locations = [
            location_to_lines(filename, loc, repo_path, file_cache, newline_cache)
            for loc in sorted(set(str(c.location) for c in chunks if c.location))[:3]
        ]
        return CodeSnippet(
            filename=filename,
            content=extract_signature(best.content),
            score=score,
            locations=locations,
            is_reference_only=True,
        )


# Singleton instance
_searcher: SearchService | None = None
_searcher_lock = threading.Lock()


def get_searcher() -> SearchService:
    """Get the singleton SearchService instance."""
    global _searcher
    if _searcher is None:
        with _searcher_lock:
            if _searcher is None:
                _searcher = SearchService()
    return _searcher

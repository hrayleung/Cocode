"""Search service - handles all code search operations.

Implements modern code retrieval best practices:
1. Tiered results (full code for top, references for rest)
2. Score-based adaptive filtering
3. Token-efficient output format
4. Graph-based expansion for related files
"""

import logging
import re
import threading
from dataclasses import dataclass, field

from config.settings import settings
from src.exceptions import SearchError
from src.retrieval.hybrid import hybrid_search
from src.retrieval.graph_expansion import expand_results_with_related

logger = logging.getLogger(__name__)

# Adaptive filtering settings
MIN_SCORE_RATIO = 0.4
FULL_CODE_COUNT = 3
MAX_CHUNKS_PER_FILE = 3



def extract_signature(content: str, language: str = "python") -> str:
    """Extract function/class signature from code content."""
    lines = content.strip().split("\n")
    signatures = []

    for line in lines:
        stripped = line.strip()
        # Skip empty lines, comments, docstrings
        if not stripped or stripped.startswith(("#", "//", "/*", '"""', "'''")):
            continue
        # Match common code signatures
        if any(stripped.startswith(kw) for kw in [
            "def ", "async def ", "class ", "function ", "const ", "let ",
            "var ", "fn ", "func ", "pub fn ", "impl ", "struct ", "enum ",
            "interface ", "type ", "export ", "@",
        ]):
            # Clean up the signature
            sig = stripped.split("{")[0].rstrip(" {:")
            signatures.append(sig)
            if len(signatures) >= 2:
                break

    if signatures:
        return "; ".join(signatures)

    # Fallback: first non-empty, non-comment line
    for line in lines[:5]:
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "//", "/*", '"""', "'''")):
            return stripped[:100]

    return lines[0][:80] if lines else ""


def parse_location(loc_str: str) -> str:
    """Convert location string to human-readable format."""
    # Handle Range format like "[0, 1570)" -> "~L1-39"
    match = re.match(r'\[(\d+),\s*(\d+)\)', str(loc_str))
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        # Approximate line numbers (~40 chars/line average)
        start_line = start // 40 + 1
        end_line = end // 40
        if end_line > start_line:
            return f"~L{start_line}-{end_line}"
        return f"~L{start_line}"
    return str(loc_str)


@dataclass
class CodeSnippet:
    """A code snippet from search results."""
    filename: str
    content: str
    score: float
    locations: list[str] = field(default_factory=list)
    is_reference_only: bool = False  # True if this is a compact reference

    def to_dict(self) -> dict:
        result = {
            "filename": self.filename,
            "score": round(self.score, 4),
        }
        if self.is_reference_only:
            # Compact format for lower-relevance results
            result["reference"] = self.content  # Contains signature/preview
            result["lines"] = self.locations
        else:
            # Full format for top results
            result["content"] = self.content
            result["locations"] = self.locations
        return result


class SearchService:
    """Service for searching indexed codebases.

    Uses tiered presentation: full code for top results,
    compact references for lower-relevance matches.
    """

    def search(
        self,
        repo_name: str,
        query: str,
        top_k: int = 10,
        full_code_count: int = FULL_CODE_COUNT,
        expand_related: bool = True,
        max_related: int = 3,
    ) -> list[CodeSnippet]:
        """Search a repository using hybrid semantic + keyword search.

        Returns tiered results:
        - Top N results: full code content
        - Remaining: filename:lines with signature preview
        - Related files: files that import or are imported by results

        Args:
            repo_name: Name of the indexed repository
            query: Natural language search query
            top_k: Total number of files to return
            full_code_count: How many top results get full code
            expand_related: Whether to include related files via import graph
            max_related: Maximum number of related files to include

        Returns:
            List of relevant code snippets

        Raises:
            SearchError: If search fails
        """
        if not query or not query.strip():
            raise SearchError("Query cannot be empty")

        try:
            # Fetch candidates
            candidates = top_k * 3

            results = hybrid_search(
                repo_name=repo_name,
                query=query.strip(),
                top_k=candidates,
                use_reranker=bool(settings.cohere_api_key),
            )

            if not results:
                return []

            # Apply adaptive score filtering
            top_score = results[0].score
            min_score = top_score * MIN_SCORE_RATIO
            filtered = [r for r in results if r.score >= min_score]

            # Aggregate by file with tiered presentation
            file_results = self._aggregate_tiered(filtered, top_k, full_code_count)

            # Expand with related files (imports/imported-by)
            if expand_related and file_results:
                result_filenames = [r.filename for r in file_results]
                try:
                    related_files = expand_results_with_related(
                        repo_name, result_filenames, max_expansion=max_related
                    )
                    if related_files:
                        logger.debug(f"Found {len(related_files)} related files")
                        # Add related files as reference snippets
                        for related_file in related_files:
                            file_results.append(CodeSnippet(
                                filename=related_file,
                                content=f"[Related via imports]",
                                score=0.0,  # Mark as related, not matched
                                locations=[],
                                is_reference_only=True,
                            ))
                except Exception as e:
                    logger.debug(f"Graph expansion failed: {e}")

            return file_results

        except Exception as e:
            logger.error(f"Search failed for {repo_name}: {e}")
            raise SearchError(f"Search failed: {e}") from e

    def _aggregate_tiered(
        self,
        results: list,
        top_k: int,
        full_code_count: int,
    ) -> list[CodeSnippet]:
        """Aggregate results with tiered presentation.

        Top results get full code, rest get compact references.
        Ensures minimum ratio of implementation files.
        """
        from src.retrieval.file_categorizer import categorize_file
        
        file_chunks: dict[str, list] = {}
        file_scores: dict[str, float] = {}
        file_categories: dict[str, str] = {}

        for r in results:
            filename = r.filename
            if filename not in file_chunks:
                file_chunks[filename] = []
                file_scores[filename] = 0.0
                file_categories[filename] = categorize_file(filename)

            file_chunks[filename].append(r)
            file_scores[filename] = max(file_scores[filename], r.score)

        all_files = sorted(
            file_chunks.keys(),
            key=lambda f: file_scores[f],
            reverse=True
        )
        
        selected = self._mmr_diversify(
            all_files, file_scores, file_categories, top_k, settings.diversity_lambda
        )

        aggregated = []
        for i, filename in enumerate(selected):
            chunks = file_chunks[filename]
            is_top_result = i < full_code_count

            if is_top_result:
                snippet = self._build_full_snippet(filename, chunks, file_scores[filename])
            else:
                snippet = self._build_reference_snippet(filename, chunks, file_scores[filename])

            aggregated.append(snippet)

        return aggregated

    def _mmr_diversify(
        self,
        candidates: list[str],
        scores: dict[str, float],
        categories: dict[str, str],
        k: int,
        lambda_param: float,
    ) -> list[str]:
        """Maximal Marginal Relevance for diverse result selection.

        MMR = λ * Relevance(d) - (1-λ) * max Similarity(d, S)

        Balances relevance with diversity by penalizing results similar
        to already-selected ones. In our case, "similarity" means same category.
        """
        selected = []
        remaining = list(candidates)

        category_counts = {
            "implementation": 0,
            "test": 0,
            "documentation": 0,
            "config": 0,
        }

        while len(selected) < k and remaining:
            best_file = None
            best_mmr_score = float('-inf')

            for candidate in remaining:
                relevance = scores[candidate]
                category = categories[candidate]

                divisor = len(selected) if len(selected) > 0 else 1
                diversity_penalty = category_counts.get(category, 0) / divisor

                mmr_score = (
                    lambda_param * relevance
                    - (1 - lambda_param) * diversity_penalty
                )

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_file = candidate

            if best_file:
                selected.append(best_file)
                remaining.remove(best_file)
                category = categories[best_file]
                category_counts[category] = category_counts.get(category, 0) + 1

        return selected

    def _build_full_snippet(
        self,
        filename: str,
        chunks: list,
        score: float,
    ) -> CodeSnippet:
        """Build full code snippet for top results.

        Returns complete function/class content without truncation.
        """
        # Sort by score and take top chunks
        chunks_by_score = sorted(chunks, key=lambda c: c.score, reverse=True)
        top_chunks = chunks_by_score[:MAX_CHUNKS_PER_FILE]

        # Sort selected chunks by location for readability
        chunks_sorted = sorted(top_chunks, key=lambda c: c.location)

        # Merge content without truncation - return full functions/classes
        content_parts = []
        locations = []
        seen_content = set()

        for chunk in chunks_sorted:
            content_hash = hash(chunk.content.strip())
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                content_parts.append(chunk.content)
                if chunk.location:
                    locations.append(str(chunk.location))

        return CodeSnippet(
            filename=filename,
            content="\n\n".join(content_parts),
            score=score,
            locations=locations,
            is_reference_only=False,
        )

    def _build_reference_snippet(
        self,
        filename: str,
        chunks: list,
        score: float,
    ) -> CodeSnippet:
        """Build compact reference for lower-relevance results."""
        # Get best chunk for signature extraction
        best_chunk = max(chunks, key=lambda c: c.score)
        signature = extract_signature(best_chunk.content)

        # Collect and format locations
        raw_locations = sorted(set(str(c.location) for c in chunks if c.location))
        locations = [parse_location(loc) for loc in raw_locations[:3]]  # Limit to 3

        return CodeSnippet(
            filename=filename,
            content=signature,  # Just the signature/preview
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

"""File categorization for search result boosting."""

import re
from functools import lru_cache
from config.settings import settings

_CATEGORY_PATTERNS: dict[str, list[re.Pattern]] = {}


def _compile_patterns() -> None:
    """Compile regex patterns for file categorization."""
    global _CATEGORY_PATTERNS

    patterns = {
        "test": [
            r"test_[^/]*\.py$", r"[^/]*_test\.py$",
            r"[^/]*\.test\.[jt]sx?$", r"[^/]*\.spec\.[jt]sx?$",
            r"[^/]*_test\.go$", r"[^/]*_test\.rs$",
            r"(^|/)tests?/", r"(^|/)__tests__/", r"(^|/)spec/",
            r"[^/]*_spec\.rb$", r"[^/]*Test\.java$", r"[^/]*Tests?\.cs$",
        ],
        "documentation": [
            r"[^/]*\.md$", r"[^/]*\.mdx$", r"[^/]*\.rst$",
            r"(^|/)README", r"(^|/)CHANGELOG", r"(^|/)CONTRIBUTING",
            r"(^|/)LICENSE", r"(^|/)docs?/",
        ],
        "config": [
            r"[^/]*\.ya?ml$", r"[^/]*\.toml$", r"[^/]*\.json$",
            r"(^|/)\.[^/]*rc$", r"(^|/)\.env",
            r"(^|/)Makefile$", r"(^|/)Dockerfile", r"(^|/)docker-compose",
        ],
    }

    _CATEGORY_PATTERNS = {
        category: [re.compile(p, re.IGNORECASE) for p in pattern_list]
        for category, pattern_list in patterns.items()
    }


_compile_patterns()


@lru_cache(maxsize=10000)
def categorize_file(filename: str) -> str:
    """Categorize a file as test, documentation, config, or implementation."""
    normalized = filename.replace("\\", "/")
    for category, patterns in _CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(normalized):
                return category
    return "implementation"


def get_category_boost(filename: str) -> float:
    """Get score multiplier for a file based on its category."""
    category = categorize_file(filename)
    weights = {
        "implementation": settings.implementation_weight,
        "documentation": settings.documentation_weight,
        "test": settings.test_weight,
        "config": settings.config_weight,
    }
    return weights.get(category, 1.0)


def apply_category_boosting(results: list, sort: bool = True) -> list:
    """Apply category-based score boosting to search results."""
    for result in results:
        result.score *= get_category_boost(result.filename)
    if sort:
        results.sort(key=lambda r: r.score, reverse=True)
    return results

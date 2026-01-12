"""File categorization for search result boosting.

Categorizes files as implementation, test, documentation, or config
to enable priority-based ranking in search results.
"""

import re
from functools import lru_cache

from config.settings import settings

# File category patterns - compiled once for performance
# Order matters: first match wins
_CATEGORY_PATTERNS: dict[str, list[re.Pattern]] = {}


def _compile_patterns() -> None:
    """Compile regex patterns for file categorization."""
    global _CATEGORY_PATTERNS

    patterns = {
        "test": [
            r"test_[^/]*\.py$",        # Python: test_foo.py
            r"[^/]*_test\.py$",        # Python: foo_test.py
            r"[^/]*\.test\.[jt]sx?$",  # JS/TS: foo.test.ts, foo.test.jsx
            r"[^/]*\.spec\.[jt]sx?$",  # JS/TS: foo.spec.ts, foo.spec.jsx
            r"[^/]*_test\.go$",        # Go: foo_test.go
            r"[^/]*_test\.rs$",        # Rust: foo_test.rs
            r"(^|/)tests?/",           # Any file in tests/ or test/ dir
            r"(^|/)__tests__/",        # Jest convention
            r"(^|/)spec/",             # RSpec/Jest spec directories
            r"[^/]*_spec\.rb$",        # Ruby: foo_spec.rb
            r"[^/]*Test\.java$",       # Java: FooTest.java
            r"[^/]*Tests?\.cs$",       # C#: FooTest.cs or FooTests.cs
        ],
        "documentation": [
            r"[^/]*\.md$",             # Markdown files
            r"[^/]*\.mdx$",            # MDX files
            r"[^/]*\.rst$",            # reStructuredText
            r"(^|/)README",            # README files (any extension)
            r"(^|/)CHANGELOG",         # Changelog files
            r"(^|/)CONTRIBUTING",      # Contributing guides
            r"(^|/)LICENSE",           # License files
            r"(^|/)docs?/",            # docs/ or doc/ directories
        ],
        "config": [
            r"[^/]*\.ya?ml$",          # YAML configs
            r"[^/]*\.toml$",           # TOML configs (pyproject.toml, etc.)
            r"[^/]*\.json$",           # JSON configs (package.json, etc.)
            r"(^|/)\.[^/]*rc$",        # RC files (.eslintrc, .prettierrc)
            r"(^|/)\.env",             # Environment files
            r"(^|/)Makefile$",         # Makefiles
            r"(^|/)Dockerfile",        # Dockerfiles
            r"(^|/)docker-compose",    # Docker compose files
        ],
    }

    _CATEGORY_PATTERNS = {
        category: [re.compile(p, re.IGNORECASE) for p in pattern_list]
        for category, pattern_list in patterns.items()
    }


# Compile patterns on module load
_compile_patterns()


@lru_cache(maxsize=10000)
def categorize_file(filename: str) -> str:
    """Categorize a file based on its path and name.

    Categories:
    - "test": Test files and test directories
    - "documentation": Markdown, RST, README, docs/
    - "config": YAML, TOML, JSON, dotfiles
    - "implementation": Everything else (default, highest priority)

    Args:
        filename: File path (relative or absolute)

    Returns:
        Category string
    """
    # Normalize path separators
    normalized = filename.replace("\\", "/")

    for category, patterns in _CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(normalized):
                return category

    return "implementation"


def get_category_boost(filename: str) -> float:
    """Get the score multiplier for a file based on its category.

    Args:
        filename: File path

    Returns:
        Score multiplier (1.0 = no change, <1.0 = deprioritize)
    """
    category = categorize_file(filename)

    weights = {
        "implementation": settings.implementation_weight,
        "documentation": settings.documentation_weight,
        "test": settings.test_weight,
        "config": settings.config_weight,
    }

    return weights.get(category, 1.0)


def apply_category_boosting(results: list, sort: bool = True) -> list:
    """Apply category-based score boosting to search results.

    Modifies results in place and optionally re-sorts by boosted score.

    Args:
        results: List of SearchResult objects with 'filename' and 'score' attrs
        sort: Whether to re-sort results after boosting

    Returns:
        The same list with modified scores
    """
    for result in results:
        boost = get_category_boost(result.filename)
        result.score *= boost

    if sort:
        results.sort(key=lambda r: r.score, reverse=True)

    return results

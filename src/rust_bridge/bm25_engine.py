"""Python wrapper for Rust-accelerated BM25 scoring engine."""

from cocode_rust import BM25Engine as _RustBM25Engine


class BM25Engine:
    """High-performance BM25+ scoring engine."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 1.0):
        """Create engine with BM25+ parameters: k1 (term saturation), b (length norm), delta."""
        self._engine = _RustBM25Engine(k1, b, delta)

    def index(self, documents: list[str]) -> None:
        """Index a corpus of documents (tokenizes and caches for scoring)."""
        self._engine.index(documents)

    def score(
        self,
        query: str,
        top_k: int | None = None,
        score_threshold: float = 0.0
    ) -> list[tuple[int, float]]:
        """Score indexed documents against query. Returns (doc_index, score) sorted by score desc."""
        return self._engine.score(query, top_k, score_threshold)

    def get_stats(self) -> dict[str, float]:
        """Get corpus statistics."""
        return self._engine.get_stats()

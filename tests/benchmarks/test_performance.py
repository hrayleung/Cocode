"""
Benchmarks for Rust-accelerated operations.

Run with: pytest tests/benchmarks/test_performance.py -v --benchmark-only
"""

import pytest
import numpy as np

from src.rust_bridge import vector_ops, graph_algos, BM25Engine


class TestVectorOperationsBenchmarks:
    """Benchmarks for vector operations."""

    @pytest.fixture
    def vectors_small(self):
        """Small vectors (1536 dimensions - typical embedding size)."""
        np.random.seed(42)
        return (
            np.random.randn(1536).astype(np.float32),
            np.random.randn(1536).astype(np.float32),
        )

    @pytest.fixture
    def vectors_batch(self):
        """Batch of vectors for similarity computation."""
        np.random.seed(42)
        query = np.random.randn(1536).astype(np.float32)
        documents = np.random.randn(1000, 1536).astype(np.float32)
        return query, documents

    @pytest.fixture
    def ranked_lists(self):
        """Sample ranked lists for RRF."""
        list1 = [(f"doc{i}", float(100 - i)) for i in range(100)]
        list2 = [(f"doc{i}", float(95 - i)) for i in range(50, 150)]
        list3 = [(f"doc{i}", float(90 - i)) for i in range(25, 125)]
        return [list1, list2, list3]

    def test_cosine_similarity_rust(self, benchmark, vectors_small):
        """Benchmark Rust cosine similarity."""
        vec1, vec2 = vectors_small
        result = benchmark(vector_ops.cosine_similarity, vec1, vec2)
        assert -1.0 <= result <= 1.0

    def test_cosine_similarity_batch_rust(self, benchmark, vectors_batch):
        """Benchmark Rust batch cosine similarity."""
        query, documents = vectors_batch
        result = benchmark(vector_ops.cosine_similarity_batch, query, documents)
        assert len(result) == len(documents)

    def test_rrf_rust(self, benchmark, ranked_lists):
        """Benchmark Rust RRF."""
        result = benchmark(vector_ops.reciprocal_rank_fusion, ranked_lists)
        assert len(result) > 0


class TestBM25Benchmarks:
    """Benchmarks for BM25 scoring."""

    @pytest.fixture
    def documents(self):
        """Sample documents for BM25."""
        return [
            "the quick brown fox jumps over the lazy dog",
            "a fast brown fox leaps across a sleepy canine",
            "the speedy auburn vulpine bounds over the drowsy hound",
            "machine learning and artificial intelligence are transforming technology",
            "deep neural networks enable advanced pattern recognition",
            "natural language processing helps computers understand human text",
        ] * 100

    def test_bm25_index_rust(self, benchmark, documents):
        """Benchmark Rust BM25 indexing."""
        def index_documents():
            engine = BM25Engine()
            engine.index(documents)
            return engine
        engine = benchmark(index_documents)
        assert engine is not None

    def test_bm25_score_rust(self, benchmark, documents):
        """Benchmark Rust BM25 scoring."""
        engine = BM25Engine()
        engine.index(documents)
        result = benchmark(engine.score, "machine learning neural networks", top_k=10)
        assert len(result) <= 10


class TestGraphAlgorithmsBenchmarks:
    """Benchmarks for graph algorithms."""

    @pytest.fixture
    def small_graph(self):
        """Small graph with 100 nodes."""
        edges = []
        for i in range(100):
            for j in range(i + 1, min(i + 5, 100)):
                edges.append((f"node{i}", f"node{j}"))
        return edges

    @pytest.fixture
    def medium_graph(self):
        """Medium graph with 1000 nodes."""
        edges = []
        for i in range(1000):
            for j in range(i + 1, min(i + 3, 1000)):
                edges.append((f"node{i}", f"node{j}"))
        return edges

    def test_pagerank_rust_small(self, benchmark, small_graph):
        """Benchmark Rust PageRank on small graph."""
        result = benchmark(graph_algos.pagerank, small_graph)
        assert len(result) > 0

    def test_pagerank_rust_medium(self, benchmark, medium_graph):
        """Benchmark Rust PageRank on medium graph."""
        result = benchmark(graph_algos.pagerank, medium_graph)
        assert len(result) > 0

    def test_bfs_expansion_rust(self, benchmark, medium_graph):
        """Benchmark Rust BFS expansion."""
        result = benchmark(
            graph_algos.bfs_expansion,
            medium_graph,
            ["node0", "node1"],
            max_hops=3,
            max_results=50,
        )
        assert len(result) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--benchmark-only"])

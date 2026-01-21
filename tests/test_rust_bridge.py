"""
Unit tests for Rust bridge modules.

Run with: pytest tests/test_rust_bridge.py -v
"""

import pytest
import numpy as np
from src.rust_bridge import vector_ops, graph_algos, BM25Engine


class TestVectorOperations:
    """Tests for vector operations."""

    def test_cosine_similarity_identical(self):
        """Test cosine similarity of identical vectors."""
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        similarity = vector_ops.cosine_similarity(vec, vec)
        assert abs(similarity - 1.0) < 1e-6

    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity of orthogonal vectors."""
        vec1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        vec2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        similarity = vector_ops.cosine_similarity(vec1, vec2)
        assert abs(similarity) < 1e-6

    def test_cosine_similarity_batch(self):
        """Test batch cosine similarity."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        documents = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, 0.0],
        ], dtype=np.float32)

        similarities = vector_ops.cosine_similarity_batch(query, documents)

        assert len(similarities) == 3
        assert abs(similarities[0] - 1.0) < 1e-6  # Identical
        assert abs(similarities[1]) < 1e-6  # Orthogonal
        assert 0.0 < similarities[2] < 1.0  # Partial overlap

    def test_rrf_basic(self):
        """Test basic RRF functionality."""
        list1 = [("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)]
        list2 = [("doc2", 0.95), ("doc1", 0.85), ("doc4", 0.75)]

        result = vector_ops.reciprocal_rank_fusion([list1, list2])

        # doc1 and doc2 should be top results
        top_docs = [doc_id for doc_id, _ in result[:2]]
        assert "doc1" in top_docs
        assert "doc2" in top_docs

    def test_rrf_weighted(self):
        """Test weighted RRF."""
        list1 = [("doc1", 0.9)]
        list2 = [("doc2", 0.9)]

        result = vector_ops.reciprocal_rank_fusion_weighted(
            [list1, list2],
            [0.7, 0.3]
        )

        # doc1 should rank higher due to higher weight
        assert result[0][0] == "doc1"
        assert result[1][0] == "doc2"


class TestBM25Engine:
    """Tests for BM25 engine."""

    def test_bm25_basic(self):
        """Test basic BM25 functionality."""
        engine = BM25Engine()
        docs = [
            "the quick brown fox",
            "the lazy dog",
            "quick brown dogs",
        ]

        engine.index(docs)
        results = engine.score("quick brown")

        # First and third documents should score higher
        assert len(results) >= 2
        assert results[0][1] > 0.0

    def test_bm25_threshold(self):
        """Test BM25 with score threshold."""
        engine = BM25Engine()
        docs = [
            "the quick brown fox",
            "the lazy dog",
            "completely unrelated content",
        ]

        engine.index(docs)
        results = engine.score("quick brown", score_threshold=5.0)

        # Should filter out low-scoring docs
        assert len(results) <= 2

    def test_bm25_top_k(self):
        """Test BM25 with top_k limit."""
        engine = BM25Engine()
        docs = [f"document number {i}" for i in range(100)]

        engine.index(docs)
        results = engine.score("document", top_k=5)

        assert len(results) == 5

    def test_bm25_stats(self):
        """Test BM25 statistics."""
        engine = BM25Engine(k1=1.2, b=0.75, delta=0.5)
        docs = ["test document one", "test document two"]

        engine.index(docs)
        stats = engine.get_stats()

        assert stats["num_docs"] == 2.0
        assert abs(stats["k1"] - 1.2) < 1e-6
        assert abs(stats["b"] - 0.75) < 1e-6
        assert abs(stats["delta"] - 0.5) < 1e-6


class TestGraphAlgorithms:
    """Tests for graph algorithms."""

    def test_pagerank_simple(self):
        """Test PageRank on simple graph."""
        edges = [
            ("A", "B"),
            ("B", "C"),
            ("C", "A"),
        ]

        scores = graph_algos.pagerank(edges)

        # All nodes should have similar scores in a cycle
        assert len(scores) == 3
        assert all(score > 0.0 for score in scores.values())

    def test_pagerank_hub(self):
        """Test PageRank with hub node."""
        edges = [
            ("A", "B"),
            ("C", "B"),
            ("D", "B"),
        ]

        scores = graph_algos.pagerank(edges)

        # B should have highest score
        assert scores["B"] > scores["A"]
        assert scores["B"] > scores["C"]
        assert scores["B"] > scores["D"]

    def test_bfs_expansion(self):
        """Test BFS expansion."""
        edges = [
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
            ("A", "E"),
        ]

        result = graph_algos.bfs_expansion(
            edges,
            ["A"],
            max_hops=2,
            max_results=10,
            bidirectional=False
        )

        # Should find nodes within 2 hops
        assert result["A"] == 0
        assert result["B"] == 1
        assert result["E"] == 1
        assert result["C"] == 2

    def test_bfs_bidirectional(self):
        """Test bidirectional BFS."""
        edges = [
            ("A", "B"),
            ("B", "C"),
        ]

        result = graph_algos.bfs_expansion(
            edges,
            ["B"],
            max_hops=1,
            max_results=10,
            bidirectional=True
        )

        # Should find both forward (C) and backward (A) neighbors
        assert result["B"] == 0
        assert result["A"] == 1  # backward
        assert result["C"] == 1  # forward

    def test_strongly_connected_components(self):
        """Test SCC detection."""
        edges = [
            ("A", "B"),
            ("B", "C"),
            ("C", "A"),
            ("D", "E"),
        ]

        sccs = graph_algos.strongly_connected_components(edges)

        # Should find one SCC with A, B, C
        assert len(sccs) >= 2

        # Find the SCC containing A, B, C
        abc_scc = next((scc for scc in sccs if "A" in scc), None)
        assert abc_scc is not None
        assert len(abc_scc) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

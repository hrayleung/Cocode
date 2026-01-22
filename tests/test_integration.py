"""Integration tests for new production-ready features.

Tests the integration between:
- Embedding cache
- Query classification
- Symbol extraction (TypeScript/React)
- Knowledge graph relationships
- Monitoring metrics
- Incremental indexing
"""

import pytest


class TestEmbeddingCacheIntegration:
    """Test embedding cache with query classification."""

    def test_cache_hit_rate_tracking(self):
        from src.retrieval.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(max_size=10, ttl_seconds=60)

        # First access - miss
        assert cache.get("query1") is None
        cache.set("query1", [0.1, 0.2, 0.3])

        # Second access - hit
        result = cache.get("query1")
        assert result == [0.1, 0.2, 0.3]

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_cache_lru_eviction(self):
        from src.retrieval.embedding_cache import EmbeddingCache

        cache = EmbeddingCache(max_size=3, ttl_seconds=60)

        cache.set("q1", [1.0])
        cache.set("q2", [2.0])
        cache.set("q3", [3.0])

        # Access q1 to make it recently used
        cache.get("q1")

        # Add q4 - should evict q2 (least recently used)
        cache.set("q4", [4.0])

        assert cache.get("q1") is not None
        assert cache.get("q2") is None  # evicted
        assert cache.get("q3") is not None
        assert cache.get("q4") is not None


class TestQueryClassificationIntegration:
    """Test query classification with search weights."""

    def test_exact_match_queries(self):
        from src.retrieval.hybrid import classify_query

        # Snake case identifiers
        result = classify_query("get_user_by_id")
        assert result["intent"] == "exact"
        assert result["bm25_boost"] > result["vector_boost"]

        # Quoted strings
        result = classify_query('"authenticate_user"')
        assert result["intent"] == "exact"

    def test_semantic_queries(self):
        from src.retrieval.hybrid import classify_query

        result = classify_query("How does authentication work?")
        assert result["intent"] == "semantic"
        assert result["vector_boost"] > result["bm25_boost"]

    def test_hybrid_queries(self):
        from src.retrieval.hybrid import classify_query

        result = classify_query("user authentication flow")
        assert result["intent"] == "hybrid"


class TestTypeScriptSymbolExtraction:
    """Test TypeScript/React symbol extraction."""

    def test_react_component_detection(self):
        from src.parser.symbol_extractor import extract_symbols

        code = """
const UserCard = ({ name }: Props) => {
  return <div>{name}</div>;
};
"""
        symbols = extract_symbols(code, "typescript", "UserCard.tsx")
        assert len(symbols) == 1
        assert symbols[0].symbol_name == "UserCard"
        assert symbols[0].symbol_type == "component"

    def test_hook_detection(self):
        from src.parser.symbol_extractor import extract_symbols

        code = """
const useAuth = () => {
  return { user: null };
};
"""
        symbols = extract_symbols(code, "typescript", "hooks.ts")
        assert len(symbols) == 1
        assert symbols[0].symbol_name == "useAuth"
        assert symbols[0].symbol_type == "hook"

    def test_interface_and_type_extraction(self):
        from src.parser.symbol_extractor import extract_symbols

        code = """
interface User {
  name: string;
}

type Status = "active" | "inactive";

enum Role {
  Admin,
  User
}
"""
        symbols = extract_symbols(code, "typescript", "types.ts")
        names = {s.symbol_name: s.symbol_type for s in symbols}

        assert names["User"] == "interface"
        assert names["Status"] == "type"
        assert names["Role"] == "enum"


class TestKnowledgeGraphRelationships:
    """Test inheritance/implementation relationship extraction."""

    def test_python_inheritance(self):
        from src.parser.symbol_extractor import extract_relationships

        code = """
class Animal:
    pass

class Dog(Animal):
    pass
"""
        rels = extract_relationships(code, "python")
        assert len(rels) == 1
        assert rels[0].source_name == "Dog"
        assert rels[0].target_name == "Animal"
        assert rels[0].relationship_type == "extends"

    def test_typescript_implements(self):
        from src.parser.symbol_extractor import extract_relationships

        code = """
class UserService implements IService {
  getData() {}
}
"""
        rels = extract_relationships(code, "typescript")
        assert len(rels) == 1
        assert rels[0].source_name == "UserService"
        assert rels[0].target_name == "IService"
        assert rels[0].relationship_type == "implements"

    def test_multiple_inheritance(self):
        from src.parser.symbol_extractor import extract_relationships

        code = """
class Cat(Animal, Serializable):
    pass
"""
        rels = extract_relationships(code, "python")
        assert len(rels) == 2
        targets = {r.target_name for r in rels}
        assert targets == {"Animal", "Serializable"}


class TestMonitoringMetrics:
    """Test production monitoring metrics."""

    def test_metrics_collection(self):
        from src.monitoring.metrics import (
            MetricsCollector,
            record_search_latency,
            record_cache_access,
        )

        # Use fresh collector
        collector = MetricsCollector()

        # Record some operations
        with collector._lock:
            collector.search_latency.record(50.0)
            collector.search_latency.record(75.0)
            collector.search_operations.success += 2
            collector.cache_hits += 3
            collector.cache_misses += 1

        metrics = collector.to_dict()

        assert metrics["search"]["latency"]["count"] == 2
        assert metrics["search"]["latency"]["avg_ms"] == 62.5
        assert metrics["search"]["operations"]["success"] == 2
        assert metrics["cache"]["hit_rate"] == 0.75

    def test_timed_decorator(self):
        from src.monitoring.metrics import timed
        import time

        latencies = []

        @timed(lambda ms: latencies.append(ms))
        def slow_function():
            time.sleep(0.01)
            return "done"

        result = slow_function()
        assert result == "done"
        assert len(latencies) == 1
        assert latencies[0] >= 5  # allow scheduler jitter


class TestIncrementalIndexing:
    """Test incremental symbol re-indexing."""

    def test_file_hash_computation(self):
        from src.indexer.symbol_indexing import _compute_file_hash

        h1 = _compute_file_hash("def foo(): pass")
        h2 = _compute_file_hash("def foo(): pass")
        h3 = _compute_file_hash("def bar(): pass")

        assert h1 == h2  # same content
        assert h1 != h3  # different content
        assert len(h1) == 16  # truncated hash

    def test_file_needs_reindex_new_file(self):
        from src.indexer.symbol_indexing import (
            file_needs_reindex,
            _file_hash_cache,
        )

        # Clear cache for test
        _file_hash_cache.clear()

        # New file always needs indexing
        result = file_needs_reindex("test_repo", "new_file.py", "content")
        assert result is True

    def test_cache_management(self):
        from src.indexer.symbol_indexing import (
            _set_cached_hash,
            _get_cached_hash,
            _clear_cached_hash,
            _file_hash_cache,
        )

        _file_hash_cache.clear()

        _set_cached_hash("repo1", "file.py", "abc123")
        assert _get_cached_hash("repo1", "file.py") == "abc123"

        _clear_cached_hash("repo1", "file.py")
        assert _get_cached_hash("repo1", "file.py") is None


class TestModelDimensions:
    """Test embedding model dimensions reference."""

    def test_model_dimensions_exist(self):
        from src.embeddings.provider import MODEL_DIMENSIONS

        # Check all providers have dimensions
        assert "openai" in MODEL_DIMENSIONS
        assert "jina" in MODEL_DIMENSIONS
        assert "mistral" in MODEL_DIMENSIONS

    def test_jina_dimensions(self):
        from src.embeddings.provider import MODEL_DIMENSIONS

        assert MODEL_DIMENSIONS["jina"]["jina-embeddings-v3"] == 1024

    def test_openai_dimensions(self):
        from src.embeddings.provider import MODEL_DIMENSIONS

        assert MODEL_DIMENSIONS["openai"]["text-embedding-3-large"] == 3072
        assert MODEL_DIMENSIONS["openai"]["text-embedding-3-small"] == 1536


class TestEvaluationMetrics:
    """Test IR evaluation metrics."""

    def test_mrr_calculation(self):
        from src.evaluation import mean_reciprocal_rank

        # Perfect ranking - relevant item at position 1
        queries = [(["a", "b", "c"], {"a"})]
        assert mean_reciprocal_rank(queries) == 1.0

        # Second position
        queries = [(["b", "a", "c"], {"a"})]
        assert mean_reciprocal_rank(queries) == 0.5

    def test_ndcg_calculation(self):
        from src.evaluation import ndcg_at_k

        results = ["a", "b", "c", "d"]
        relevance = {"a": 3, "b": 2, "c": 1, "d": 0}

        score = ndcg_at_k(results, relevance, k=4)
        assert 0 < score <= 1.0

    def test_precision_recall(self):
        from src.evaluation import precision_at_k, recall_at_k

        results = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c", "e", "f"}

        # Precision@3: 2 relevant in top 3 = 2/3
        assert precision_at_k(results, relevant, k=3) == pytest.approx(2 / 3)

        # Recall@5: 3 of 4 relevant found = 3/4
        assert recall_at_k(results, relevant, k=5) == pytest.approx(3 / 4)

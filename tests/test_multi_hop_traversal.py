"""Tests for multi-hop BFS graph traversal."""

import pytest
from src.retrieval.graph_expansion import (
    multi_hop_traversal,
    get_related_files,
    FileRelation,
)


class TestMultiHopTraversal:
    """Test multi-hop BFS traversal algorithm."""

    def test_one_hop_forward(self):
        """Test 1-hop traversal finds direct imports."""
        import_graph = {
            "a.py": ["b.py", "c.py"],
            "b.py": ["d.py"],
            "c.py": ["e.py"],
        }
        reverse_graph = {
            "b.py": ["a.py"],
            "c.py": ["a.py"],
            "d.py": ["b.py"],
            "e.py": ["c.py"],
        }

        results = multi_hop_traversal(
            start_files=["a.py"],
            import_graph=import_graph,
            reverse_graph=reverse_graph,
            max_hops=1,
            max_results=10,
        )

        # Should find b.py and c.py at hop distance 1
        assert len(results) >= 2
        files_found = {(target, hop) for _, target, _, hop in results}
        assert ("b.py", 1) in files_found
        assert ("c.py", 1) in files_found

    def test_two_hop_forward(self):
        """Test 2-hop traversal finds transitive imports."""
        import_graph = {
            "a.py": ["b.py"],
            "b.py": ["c.py"],
            "c.py": ["d.py"],
        }
        reverse_graph = {
            "b.py": ["a.py"],
            "c.py": ["b.py"],
            "d.py": ["c.py"],
        }

        results = multi_hop_traversal(
            start_files=["a.py"],
            import_graph=import_graph,
            reverse_graph=reverse_graph,
            max_hops=2,
            max_results=10,
        )

        # Should find b.py (hop 1), c.py (hop 2)
        files_found = {(target, hop) for _, target, _, hop in results}
        assert ("b.py", 1) in files_found
        assert ("c.py", 2) in files_found
        # Should NOT find d.py (would be hop 3)
        assert not any(target == "d.py" for _, target, _, _ in results)

    def test_three_hop_comprehensive(self):
        """Test 3-hop traversal finds deep dependencies."""
        import_graph = {
            "a.py": ["b.py"],
            "b.py": ["c.py"],
            "c.py": ["d.py"],
            "d.py": ["e.py"],
        }
        reverse_graph = {
            "b.py": ["a.py"],
            "c.py": ["b.py"],
            "d.py": ["c.py"],
            "e.py": ["d.py"],
        }

        results = multi_hop_traversal(
            start_files=["a.py"],
            import_graph=import_graph,
            reverse_graph=reverse_graph,
            max_hops=3,
            max_results=10,
        )

        # Should find b.py (hop 1), c.py (hop 2), d.py (hop 3)
        files_found = {(target, hop) for _, target, _, hop in results}
        assert ("b.py", 1) in files_found
        assert ("c.py", 2) in files_found
        assert ("d.py", 3) in files_found
        # Should NOT find e.py (would be hop 4)
        assert not any(target == "e.py" for _, target, _, _ in results)

    def test_cycle_detection(self):
        """Test that cycles don't cause infinite loops."""
        # Create circular dependency: a -> b -> c -> a
        import_graph = {
            "a.py": ["b.py"],
            "b.py": ["c.py"],
            "c.py": ["a.py"],
        }
        reverse_graph = {
            "a.py": ["c.py"],
            "b.py": ["a.py"],
            "c.py": ["b.py"],
        }

        results = multi_hop_traversal(
            start_files=["a.py"],
            import_graph=import_graph,
            reverse_graph=reverse_graph,
            max_hops=3,
            max_results=10,
        )

        # Should terminate and find b.py and c.py without infinite loop
        assert len(results) > 0
        files_found = {target for _, target, _, _ in results}
        assert "b.py" in files_found
        assert "c.py" in files_found
        # a.py is start file, shouldn't appear in results
        assert "a.py" not in files_found

    def test_max_results_limiting(self):
        """Test that max_results limit is respected."""
        # Create graph with many connections
        import_graph = {
            "a.py": [f"file{i}.py" for i in range(20)],
        }
        reverse_graph = {}

        results = multi_hop_traversal(
            start_files=["a.py"],
            import_graph=import_graph,
            reverse_graph=reverse_graph,
            max_hops=2,
            max_results=5,
        )

        # Should respect max_results limit
        assert len(results) <= 5

    def test_empty_start_files(self):
        """Test behavior with empty start files."""
        import_graph = {"a.py": ["b.py"]}
        reverse_graph = {"b.py": ["a.py"]}

        results = multi_hop_traversal(
            start_files=[],
            import_graph=import_graph,
            reverse_graph=reverse_graph,
            max_hops=3,
            max_results=10,
        )

        assert len(results) == 0

    def test_bidirectional_relationships(self):
        """Test that both imports and imported_by edges are followed."""
        import_graph = {
            "a.py": ["b.py"],  # a imports b
            "c.py": ["a.py"],  # c imports a
        }
        reverse_graph = {
            "a.py": ["c.py"],  # a is imported by c
            "b.py": ["a.py"],  # b is imported by a
        }

        results = multi_hop_traversal(
            start_files=["a.py"],
            import_graph=import_graph,
            reverse_graph=reverse_graph,
            max_hops=1,
            max_results=10,
        )

        # Should find both b.py (via imports) and c.py (via imported_by)
        files_found = {target for _, target, _, _ in results}
        assert "b.py" in files_found
        assert "c.py" in files_found

        # Check relation types
        relations = {target: rel_type for _, target, rel_type, _ in results}
        assert relations["b.py"] == "imports"
        assert relations["c.py"] == "imported_by"

    def test_complex_graph_structure(self):
        """Test traversal on a complex graph with multiple paths."""
        # Diamond structure: a imports b and c, both import d
        import_graph = {
            "a.py": ["b.py", "c.py"],
            "b.py": ["d.py"],
            "c.py": ["d.py"],
        }
        reverse_graph = {
            "b.py": ["a.py"],
            "c.py": ["a.py"],
            "d.py": ["b.py", "c.py"],
        }

        results = multi_hop_traversal(
            start_files=["a.py"],
            import_graph=import_graph,
            reverse_graph=reverse_graph,
            max_hops=2,
            max_results=10,
        )

        # Should find b.py, c.py (hop 1) and d.py (hop 2)
        files_found = {target for _, target, _, _ in results}
        assert "b.py" in files_found
        assert "c.py" in files_found
        assert "d.py" in files_found

        # d.py should only appear once despite multiple paths
        d_count = sum(1 for _, target, _, _ in results if target == "d.py")
        assert d_count == 1

    def test_hop_distance_accuracy(self):
        """Test that hop distances are correctly tracked."""
        import_graph = {
            "a.py": ["b.py"],
            "b.py": ["c.py"],
            "c.py": ["d.py"],
        }
        reverse_graph = {
            "b.py": ["a.py"],
            "c.py": ["b.py"],
            "d.py": ["c.py"],
        }

        results = multi_hop_traversal(
            start_files=["a.py"],
            import_graph=import_graph,
            reverse_graph=reverse_graph,
            max_hops=3,
            max_results=10,
        )

        # Create map of file -> hop distance
        hop_distances = {target: hop for _, target, _, hop in results}

        assert hop_distances.get("b.py") == 1
        assert hop_distances.get("c.py") == 2
        assert hop_distances.get("d.py") == 3

    def test_multiple_start_files(self):
        """Test traversal starting from multiple files."""
        import_graph = {
            "a.py": ["common.py"],
            "b.py": ["common.py"],
            "common.py": ["util.py"],
        }
        reverse_graph = {
            "common.py": ["a.py", "b.py"],
            "util.py": ["common.py"],
        }

        results = multi_hop_traversal(
            start_files=["a.py", "b.py"],
            import_graph=import_graph,
            reverse_graph=reverse_graph,
            max_hops=2,
            max_results=10,
        )

        # Should find common.py (hop 1 from both) and util.py (hop 2)
        files_found = {target for _, target, _, _ in results}
        assert "common.py" in files_found
        assert "util.py" in files_found


class TestGetRelatedFilesMultiHop:
    """Test get_related_files with multi-hop support."""

    def test_file_relation_has_hop_distance(self):
        """Test that FileRelation objects include hop_distance."""
        relation = FileRelation(
            source_file="a.py",
            target_file="b.py",
            relation_type="imports",
            hop_distance=2,
        )

        assert relation.hop_distance == 2
        assert relation.source_file == "a.py"
        assert relation.target_file == "b.py"
        assert relation.relation_type == "imports"

    def test_default_hop_distance(self):
        """Test that hop_distance defaults to 1."""
        relation = FileRelation(
            source_file="a.py",
            target_file="b.py",
            relation_type="imports",
        )

        assert relation.hop_distance == 1


class TestEdgeCases:
    """Test edge cases in multi-hop traversal."""

    def test_self_reference_ignored(self):
        """Test that self-references are ignored."""
        import_graph = {
            "a.py": ["a.py"],  # Self-reference
        }
        reverse_graph = {
            "a.py": ["a.py"],
        }

        results = multi_hop_traversal(
            start_files=["a.py"],
            import_graph=import_graph,
            reverse_graph=reverse_graph,
            max_hops=3,
            max_results=10,
        )

        # Should not infinitely loop or include a.py in results
        files_found = {target for _, target, _, _ in results}
        assert "a.py" not in files_found

    def test_disconnected_graph(self):
        """Test traversal on disconnected graph components."""
        import_graph = {
            "a.py": ["b.py"],
            "c.py": ["d.py"],  # Disconnected component
        }
        reverse_graph = {
            "b.py": ["a.py"],
            "d.py": ["c.py"],
        }

        results = multi_hop_traversal(
            start_files=["a.py"],
            import_graph=import_graph,
            reverse_graph=reverse_graph,
            max_hops=3,
            max_results=10,
        )

        # Should only find files reachable from a.py
        files_found = {target for _, target, _, _ in results}
        assert "b.py" in files_found
        assert "c.py" not in files_found
        assert "d.py" not in files_found

    def test_empty_import_graph(self):
        """Test behavior with empty import graph."""
        results = multi_hop_traversal(
            start_files=["a.py"],
            import_graph={},
            reverse_graph={},
            max_hops=3,
            max_results=10,
        )

        assert len(results) == 0

    def test_max_hops_zero(self):
        """Test that max_hops=0 returns no results."""
        import_graph = {"a.py": ["b.py"]}
        reverse_graph = {"b.py": ["a.py"]}

        results = multi_hop_traversal(
            start_files=["a.py"],
            import_graph=import_graph,
            reverse_graph=reverse_graph,
            max_hops=0,
            max_results=10,
        )

        assert len(results) == 0

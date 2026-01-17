"""Unit tests for expand_results_with_related()."""

from unittest.mock import patch

from src.retrieval.graph_expansion import FileRelation, expand_results_with_related


def test_expand_results_with_related_uses_target_for_imported_by():
    relations = [
        FileRelation(
            source_file="a.py",
            target_file="c.py",
            relation_type="imported_by",
            hop_distance=1,
        )
    ]

    with patch("src.retrieval.graph_expansion.get_related_files", return_value=relations), patch(
        "src.retrieval.file_categorizer.categorize_file", return_value="implementation"
    ):
        related = expand_results_with_related("repo", ["a.py"], max_expansion=3)

    assert related == ["c.py"]


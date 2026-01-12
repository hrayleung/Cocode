"""Unit tests for SQL generation in storage schema helpers."""

from src.storage.schema import get_create_chunks_table_sql


def test_get_create_chunks_table_sql_index_names_are_valid():
    sql_obj = get_create_chunks_table_sql("myrepo", dimensions=1024)
    rendered = sql_obj.as_string(None)

    # Regression guard: identifiers must not be quoted and then concatenated.
    assert '"myrepo"_chunks_embedding_idx' not in rendered
    assert '"myrepo"_chunks_content_tsv_idx' not in rendered
    assert '"myrepo"_chunks_filename_idx' not in rendered

    # Expected identifiers are fully quoted as a single token.
    assert 'CREATE INDEX IF NOT EXISTS "myrepo_chunks_embedding_idx"' in rendered
    assert 'CREATE INDEX IF NOT EXISTS "myrepo_chunks_content_tsv_idx"' in rendered
    assert 'CREATE INDEX IF NOT EXISTS "myrepo_chunks_filename_idx"' in rendered


def test_get_create_chunks_table_sql_schema_name_normalization():
    sql_obj = get_create_chunks_table_sql("My-Repo.Name", dimensions=1024)
    rendered = sql_obj.as_string(None)

    assert 'CREATE TABLE IF NOT EXISTS "my_repo_name"."chunks"' in rendered
    assert 'CREATE INDEX IF NOT EXISTS "my_repo_name_chunks_embedding_idx"' in rendered

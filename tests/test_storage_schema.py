"""Unit tests for SQL generation in storage schema helpers."""

from src.storage.schema import get_create_chunks_table_sql, get_create_symbols_table_sql


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


def test_get_create_symbols_table_sql_index_names_are_valid():
    """Test that symbols table index names are correctly formatted."""
    sql_obj = get_create_symbols_table_sql("myrepo", dimensions=1024)
    rendered = sql_obj.as_string(None)

    # Regression guard: identifiers must not be quoted and then concatenated.
    assert '"myrepo"_symbols_embedding_idx' not in rendered
    assert '"myrepo"_symbols_content_tsv_idx' not in rendered
    assert '"myrepo"_symbols_filename_idx' not in rendered
    assert '"myrepo"_symbols_name_idx' not in rendered
    assert '"myrepo"_symbols_type_idx' not in rendered

    # Expected identifiers are fully quoted as a single token.
    assert 'CREATE INDEX IF NOT EXISTS "myrepo_symbols_embedding_idx"' in rendered
    assert 'CREATE INDEX IF NOT EXISTS "myrepo_symbols_content_tsv_idx"' in rendered
    assert 'CREATE INDEX IF NOT EXISTS "myrepo_symbols_filename_idx"' in rendered
    assert 'CREATE INDEX IF NOT EXISTS "myrepo_symbols_name_idx"' in rendered
    assert 'CREATE INDEX IF NOT EXISTS "myrepo_symbols_type_idx"' in rendered


def test_get_create_symbols_table_sql_schema_name_normalization():
    """Test that symbols table schema names are normalized correctly."""
    sql_obj = get_create_symbols_table_sql("My-Repo.Name", dimensions=1024)
    rendered = sql_obj.as_string(None)

    assert 'CREATE TABLE IF NOT EXISTS "my_repo_name"."symbols"' in rendered
    assert 'CREATE INDEX IF NOT EXISTS "my_repo_name_symbols_embedding_idx"' in rendered
    assert 'CREATE INDEX IF NOT EXISTS "my_repo_name_symbols_filename_idx"' in rendered


def test_get_create_symbols_table_sql_has_required_columns():
    """Test that symbols table has all required columns."""
    sql_obj = get_create_symbols_table_sql("test", dimensions=1024)
    rendered = sql_obj.as_string(None)

    # Check required columns
    assert "symbol_name TEXT NOT NULL" in rendered
    assert "symbol_type TEXT NOT NULL" in rendered
    assert "line_start INT NOT NULL" in rendered
    assert "line_end INT NOT NULL" in rendered
    assert "signature TEXT" in rendered
    assert "docstring TEXT" in rendered
    assert "parent_symbol TEXT" in rendered
    assert "visibility TEXT" in rendered
    assert "category TEXT" in rendered
    assert "embedding vector(1024)" in rendered
    assert "content_tsv tsvector" in rendered


"""Unit tests for incremental symbol indexing during IndexerService updates."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from src.indexer.service import IndexerService


@dataclass
class _RepoStub:
    last_indexed: datetime | None


def test_incremental_update_only_reindexes_modified_symbols(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    unchanged = repo_dir / "unchanged.py"
    unchanged.write_text("def unchanged():\n    return 1\n", encoding="utf-8")

    changed = repo_dir / "changed.py"
    changed.write_text("def before():\n    return 1\n", encoding="utf-8")

    # Set mtimes so only changed.py is newer than last_indexed.
    import os
    import time

    now = time.time()
    os.utime(unchanged, (now - 100, now - 100))
    os.utime(changed, (now, now))

    indexer = IndexerService()
    monkeypatch.setattr(
        indexer._repo_manager,
        "get_repo",
        lambda _name: _RepoStub(last_indexed=datetime.fromtimestamp(now - 50, tz=timezone.utc)),
    )

    class _FakeFlow:
        def setup(self):
            return None

        def update(self):
            return None

    import src.indexer.service as service_mod

    monkeypatch.setattr(indexer, "_init_cocoindex", lambda: None)
    monkeypatch.setattr(service_mod, "create_indexing_flow", lambda *_args, **_kwargs: _FakeFlow())
    monkeypatch.setattr(indexer, "_get_stats", lambda _repo_name: (0, 0))
    monkeypatch.setattr(indexer, "_verify_index_data_exists", lambda _repo_name: True)
    monkeypatch.setattr(indexer._repo_manager, "update_status", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(indexer, "_compute_centrality", lambda *_args, **_kwargs: None)

    import src.retrieval.graph_cache as graph_cache_mod

    monkeypatch.setattr(graph_cache_mod, "clear_graph_cache", lambda *_args, **_kwargs: 0)

    # Guard: incremental update should not invoke full-repo symbol indexing.
    import src.indexer.symbol_indexing as symbol_mod

    monkeypatch.setattr(
        symbol_mod,
        "index_repository_symbols",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("full symbol indexing should not run")),
    )

    delete_calls: list[str] = []
    index_calls: list[str] = []

    monkeypatch.setattr(symbol_mod, "create_symbols_table", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(symbol_mod, "delete_file_symbols", lambda _repo, filename: delete_calls.append(filename))
    monkeypatch.setattr(
        symbol_mod,
        "index_file_symbols",
        lambda _repo, filename, _content, embedding_provider=None: index_calls.append(filename),
    )

    import src.embeddings.provider as provider_mod

    monkeypatch.setattr(provider_mod, "get_provider", lambda: object())

    # Run the incremental update directly.
    indexer._incremental_update("repo", str(repo_dir))

    assert delete_calls == ["changed.py"]
    assert index_calls == ["changed.py"]


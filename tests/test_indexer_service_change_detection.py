"""Tests for IndexerService change detection fast path."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from src.indexer.service import IndexerService


@dataclass
class _RepoStub:
    last_indexed: datetime


def test_has_files_changed_is_cached_for_back_to_back_queries(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    file_path = repo_dir / "a.py"
    file_path.write_text("print('hi')\n", encoding="utf-8")

    last_mtime = file_path.stat().st_mtime
    stub = _RepoStub(last_indexed=datetime.fromtimestamp(last_mtime + 100, tz=timezone.utc))

    indexer = IndexerService()
    monkeypatch.setattr(indexer._repo_manager, "get_repo", lambda _name: stub)

    walk_calls = {"count": 0}

    import src.indexer.service as service_mod

    real_walk = service_mod.os.walk

    def counting_walk(*args, **kwargs):
        walk_calls["count"] += 1
        return real_walk(*args, **kwargs)

    # Control monotonic time so the second call hits the cache window.
    # Cache window is 1 second, so use 0.5 second difference to stay within cache
    times = iter([0.0, 0.5])
    monkeypatch.setattr(service_mod.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(service_mod.os, "walk", counting_walk)

    assert indexer._has_files_changed("repo", str(repo_dir)) is False
    assert indexer._has_files_changed("repo", str(repo_dir)) is False
    assert walk_calls["count"] == 1

"""Unit tests for repo-name resolution logic."""

from pathlib import Path

from src.indexer.service import IndexerService


class _RepoStub:
    def __init__(self, path: str):
        self.path = path


def test_resolve_repo_name_prefers_existing_base_match(tmp_path, monkeypatch):
    repo_dir = tmp_path / "My-Repo.Name"
    repo_dir.mkdir()
    resolved = str(repo_dir.resolve())

    indexer = IndexerService()

    def fake_get_repo(name: str):
        if name == "my_repo_name":
            return _RepoStub(path=resolved)
        raise AssertionError(f"unexpected repo name lookup: {name}")

    monkeypatch.setattr(indexer._repo_manager, "get_repo", fake_get_repo)

    assert indexer.resolve_repo_name(repo_dir) == "my_repo_name"


def test_resolve_repo_name_uses_hashed_suffix_on_collision(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    resolved = str(repo_dir.resolve())

    indexer = IndexerService()

    def fake_get_repo(name: str):
        if name == "repo":
            return _RepoStub(path="/some/other/path")
        if name.startswith("repo_"):
            return None
        raise AssertionError(f"unexpected repo name lookup: {name}")

    monkeypatch.setattr(indexer._repo_manager, "get_repo", fake_get_repo)

    resolved_name = indexer.resolve_repo_name(repo_dir)
    assert resolved_name.startswith("repo_")
    assert resolved_name != "repo"


def test_resolve_repo_name_returns_hashed_when_only_hashed_exists(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    resolved = str(repo_dir.resolve())

    indexer = IndexerService()

    # Compute expected hash prefix the same way as the implementation.
    import hashlib

    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()
    expected = f"repo_{digest[:8]}"

    def fake_get_repo(name: str):
        if name == "repo":
            return None
        if name == expected:
            return _RepoStub(path=resolved)
        raise AssertionError(f"unexpected repo name lookup: {name}")

    monkeypatch.setattr(indexer._repo_manager, "get_repo", fake_get_repo)

    assert indexer.resolve_repo_name(repo_dir) == expected


def test_resolve_repo_name_extends_hash_on_rare_hash_collision(tmp_path, monkeypatch):
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    resolved = str(repo_dir.resolve())

    indexer = IndexerService()

    import hashlib

    digest = hashlib.sha1(resolved.encode("utf-8")).hexdigest()
    hash8 = f"repo_{digest[:8]}"
    hash12 = f"repo_{digest[:12]}"
    hash16 = f"repo_{digest[:16]}"

    def fake_get_repo(name: str):
        if name == "repo":
            return _RepoStub(path="/some/other/path")
        if name == hash8:
            return _RepoStub(path="/another/other/path")
        if name == hash12:
            return _RepoStub(path="/yet/another/path")
        if name == hash16:
            return None  # This one is available
        raise AssertionError(f"unexpected repo name lookup: {name}")

    monkeypatch.setattr(indexer._repo_manager, "get_repo", fake_get_repo)

    assert indexer.resolve_repo_name(repo_dir) == hash16

"""Unit tests for CocoIndex flow caching behavior."""

import uuid

from src.indexer import flow as flow_mod


def test_create_indexing_flow_recreates_when_repo_path_changes(tmp_path, monkeypatch):
    class FakeFlow:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    def fake_open_flow(name, flow_def):
        return FakeFlow()

    monkeypatch.setattr(flow_mod.cocoindex, "open_flow", fake_open_flow)

    repo_name = f"flowcache_{uuid.uuid4().hex[:8]}"
    repo_path_1 = tmp_path / "repo1"
    repo_path_2 = tmp_path / "repo2"
    repo_path_1.mkdir()
    repo_path_2.mkdir()

    flow_mod.clear_flow_cache(repo_name)
    try:
        flow1 = flow_mod.create_indexing_flow(repo_name, str(repo_path_1))
        flow2 = flow_mod.create_indexing_flow(repo_name, str(repo_path_1))
        assert flow1 is flow2
        assert not flow1.closed

        flow3 = flow_mod.create_indexing_flow(repo_name, str(repo_path_2))
        assert flow3 is not flow1
        assert flow1.closed
    finally:
        flow_mod.clear_flow_cache(repo_name)

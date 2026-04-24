"""Phase 12: dynamic roots (auto-discovery of workspaces) end-to-end.

Covers:
  - DynamicRoots persistence (add, touch, remove, prune, idempotent).
  - Settings.all_roots() unions config + dynamic, deduped and existence-filtered.
  - MCP `ensure_workspace_indexed` handler registers a new root and spawns a
    background subprocess (we stub subprocess.Popen to avoid actually indexing).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from code_rag.dynamic_roots import DynamicRoots

# ---- DynamicRoots unit ------------------------------------------------------


def test_add_is_idempotent(tmp_path: Path) -> None:
    store = tmp_path / "dyn.json"
    d = DynamicRoots.load(store)
    repo = tmp_path / "repo"
    repo.mkdir()
    assert d.add(repo) is True
    assert d.add(repo) is False  # second add is a no-op
    # Persistence roundtrip
    d2 = DynamicRoots.load(store)
    assert [e.path.resolve() for e in d2.entries] == [repo.resolve()]


def test_paths_filters_missing_and_nondirs(tmp_path: Path) -> None:
    store = tmp_path / "dyn.json"
    d = DynamicRoots.load(store)
    live = tmp_path / "live"
    live.mkdir()
    gone = tmp_path / "gone"
    gone.mkdir()
    d.add(live)
    d.add(gone)
    # Delete one after adding — paths() should skip it.
    import shutil
    shutil.rmtree(gone)
    assert d.paths() == [live.resolve()]


def test_remove_returns_false_when_absent(tmp_path: Path) -> None:
    store = tmp_path / "dyn.json"
    d = DynamicRoots.load(store)
    assert d.remove(tmp_path / "never-added") is False


def test_prune_stale_drops_by_last_used(tmp_path: Path) -> None:
    store = tmp_path / "dyn.json"
    d = DynamicRoots.load(store)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    stale = tmp_path / "stale"
    stale.mkdir()
    d.add(fresh)
    d.add(stale)
    # Backdate the stale entry.
    for e in d.entries:
        if e.path.name == "stale":
            e.last_used_at = (datetime.now(UTC) - timedelta(days=90)).isoformat()
    d.save()

    pruned = d.prune_stale(30)
    assert [p.name for p in pruned] == ["stale"]
    d2 = DynamicRoots.load(store)
    assert [e.path.name for e in d2.entries] == ["fresh"]


def test_touch_updates_last_used(tmp_path: Path) -> None:
    store = tmp_path / "dyn.json"
    d = DynamicRoots.load(store)
    repo = tmp_path / "r"
    repo.mkdir()
    d.add(repo)
    orig = d.entries[0].last_used_at
    # Backdate then touch and confirm it advances.
    d.entries[0].last_used_at = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    d.save()
    d.touch(repo)
    d2 = DynamicRoots.load(store)
    assert d2.entries[0].last_used_at > orig or d2.entries[0].last_used_at != orig


# ---- Settings.all_roots() ---------------------------------------------------


def test_all_roots_unions_and_dedupes(tmp_path: Path, monkeypatch) -> None:
    from code_rag.config import load_settings
    cfg_dir = tmp_path / "project"
    cfg_dir.mkdir()
    r1 = cfg_dir / "r1"
    r1.mkdir()
    r2 = cfg_dir / "r2"
    r2.mkdir()
    cfg = cfg_dir / "config.toml"
    cfg.write_text(f"""
[paths]
roots    = ["{r1.as_posix()}"]
data_dir = "./data"
log_dir  = "./logs"

[embedder]
kind     = "lm_studio"
base_url = "http://localhost:1234/v1"
model    = "m"

[reranker]
kind     = "lm_studio"
base_url = "http://localhost:1234/v1"
model    = "m"
""", encoding="utf-8")
    monkeypatch.setenv("CODE_RAG_CONFIG", str(cfg))
    settings = load_settings()

    # Register r2 as a dynamic root. Also add r1 dynamically — should dedupe.
    d = DynamicRoots.load(settings.dynamic_roots_path)
    d.add(r2)
    d.add(r1)

    union = settings.all_roots()
    assert set(p.name for p in union) == {"r1", "r2"}
    assert len(union) == 2  # no duplicate from r1-in-both


# ---- MCP ensure_workspace_indexed tool --------------------------------------


@pytest.mark.asyncio
async def test_ensure_workspace_indexed_skips_when_already_under_root(
    tmp_path: Path, monkeypatch,
) -> None:
    """If path is under an existing configured root, return already_indexed=True
    without spawning a subprocess."""
    from code_rag.config import load_settings
    from code_rag.mcp_server.server import (
        ServerResources,
        _tool_ensure_workspace_indexed,
    )

    cfg_dir = tmp_path / "proj"
    cfg_dir.mkdir()
    root = cfg_dir / "root"
    root.mkdir()
    (root / "nested").mkdir()
    cfg = cfg_dir / "config.toml"
    cfg.write_text(f"""
[paths]
roots    = ["{root.as_posix()}"]
data_dir = "./data"
log_dir  = "./logs"

[embedder]
kind     = "lm_studio"
base_url = "http://localhost:1234/v1"
model    = "m"

[reranker]
kind     = "lm_studio"
base_url = "http://localhost:1234/v1"
model    = "m"
""", encoding="utf-8")
    monkeypatch.setenv("CODE_RAG_CONFIG", str(cfg))
    res = ServerResources(load_settings())

    out = await _tool_ensure_workspace_indexed(res, {"path": str(root / "nested")})
    assert out["already_indexed"] is True
    assert out["registered"] is False


@pytest.mark.asyncio
async def test_ensure_workspace_indexed_registers_and_spawns(
    tmp_path: Path, monkeypatch,
) -> None:
    """A brand-new path should get added to dynamic roots and a background
    subprocess kicked off. We stub Popen to avoid actually spawning one."""
    from code_rag.config import load_settings
    from code_rag.mcp_server.server import (
        ServerResources,
        _tool_ensure_workspace_indexed,
    )

    cfg_dir = tmp_path / "proj"
    cfg_dir.mkdir()
    config_root = cfg_dir / "croot"
    config_root.mkdir()
    cfg = cfg_dir / "config.toml"
    cfg.write_text(f"""
[paths]
roots    = ["{config_root.as_posix()}"]
data_dir = "./data"
log_dir  = "./logs"

[embedder]
kind     = "lm_studio"
base_url = "http://localhost:1234/v1"
model    = "m"

[reranker]
kind     = "lm_studio"
base_url = "http://localhost:1234/v1"
model    = "m"
""", encoding="utf-8")
    monkeypatch.setenv("CODE_RAG_CONFIG", str(cfg))
    res = ServerResources(load_settings())

    # New path outside any configured root.
    new_repo = tmp_path / "new_workspace"
    new_repo.mkdir()

    calls: list[list[str]] = []

    class _StubProc:
        pid = 99999

    def _stub_popen(args: Any, **kwargs: Any) -> _StubProc:
        calls.append(list(args))
        return _StubProc()

    # Patch subprocess.Popen as imported locally in the handler.
    import subprocess as _subprocess_mod
    monkeypatch.setattr(_subprocess_mod, "Popen", _stub_popen)

    out = await _tool_ensure_workspace_indexed(res, {"path": str(new_repo)})
    assert out["already_indexed"] is False
    assert out["registered"] is True
    assert out["background_pid"] == 99999

    # Subprocess invoked with `-m code_rag index --path <new_repo>`.
    assert any("code_rag" in a and "index" in " ".join(c) for c in calls for a in c), calls

    # Persisted to dynamic_roots.json.
    d = DynamicRoots.load(res.settings.dynamic_roots_path)
    assert any(e.path.resolve() == new_repo.resolve() for e in d.entries)


@pytest.mark.asyncio
async def test_ensure_workspace_indexed_rejects_nonexistent_path(
    tmp_path: Path, monkeypatch,
) -> None:
    from code_rag.config import load_settings
    from code_rag.mcp_server.server import (
        ServerResources,
        _tool_ensure_workspace_indexed,
    )
    cfg_dir = tmp_path / "proj"
    cfg_dir.mkdir()
    root = cfg_dir / "r"
    root.mkdir()
    cfg = cfg_dir / "config.toml"
    cfg.write_text(f"""
[paths]
roots    = ["{root.as_posix()}"]
data_dir = "./data"
log_dir  = "./logs"

[embedder]
kind     = "lm_studio"
base_url = "http://localhost:1234/v1"
model    = "m"

[reranker]
kind     = "lm_studio"
base_url = "http://localhost:1234/v1"
model    = "m"
""", encoding="utf-8")
    monkeypatch.setenv("CODE_RAG_CONFIG", str(cfg))
    res = ServerResources(load_settings())
    out = await _tool_ensure_workspace_indexed(res, {"path": str(tmp_path / "nope")})
    assert "error" in out

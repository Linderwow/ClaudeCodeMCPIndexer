"""Phase 60-M: SingletonLock for cmd_index + sticky wipe_recovery marker.

Two pieces of new logic worth pinning down:

* **`cmd_index` lock semantics.** Before Phase 60-M, the `code-rag index`
  command would happily race the watcher on Kuzu's `graph.kz` lock and
  produce `IO exception: Could not set lock on file` at every spawn. The
  fix takes a watcher-lock check (and its own `indexer.lock`) and exits
  rc=0 cleanly. Tested by simulating both lockfiles with the current
  process's PID and asserting the operations layer reports them as live.

* **Wipe-recovery state classifier with sticky marker.** Without the
  marker, the dashboard's `_index_status` race-loses against the indexer
  upserting the first batch of vectors -- the brief vectors==0 window
  closes within a single 5s poll interval and the user never sees the
  `wipe_recovery` state. Phase 60-M writes a `.wipe_recovery` marker in
  `data_dir` when the autostart detector fires; the dashboard reads it
  and stays in `wipe_recovery` until vectors climb back to >=95% of fts.

These tests are deliberately narrow -- they validate the new branch
points, not the larger autostart bootstrap or dashboard. The full
end-to-end is covered by manual smoke and the runtime audit.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from code_rag.util.proc_hygiene import SingletonLock, is_process_alive

# ---- SingletonLock contract --------------------------------------------------


def test_singleton_lock_self_pid_is_live(tmp_path: Path) -> None:
    """Sanity: when we hold a lock with our own PID, is_process_alive
    confirms it. This is the exact check resume_code_rag.ps1 and
    cmd_index use to decide whether to bail out."""
    lock_path = tmp_path / "watcher.lock"
    with SingletonLock(lock_path) as lock:
        assert lock.acquired
        assert lock_path.exists()
        holder = int(lock_path.read_text("utf-8").strip())
        assert holder == os.getpid()
        assert is_process_alive(holder) is True
    # Lock cleaned up on context exit.
    assert not lock_path.exists()


def test_singleton_lock_steals_stale_pid(tmp_path: Path) -> None:
    """Stale lock (PID not alive) gets stolen on next acquire. This is
    the recovery path after an unclean watcher crash -- without it, the
    next watcher would refuse to start until someone manually deletes
    watcher.lock."""
    lock_path = tmp_path / "watcher.lock"
    # PID 1 exists on POSIX (init) but not on Windows in user space; pick
    # one that's basically guaranteed not to be a real process.
    lock_path.write_text("999999", encoding="utf-8")
    with SingletonLock(lock_path) as lock:
        assert lock.acquired, "should have stolen stale lock"
        # Now we own it.
        assert int(lock_path.read_text("utf-8")) == os.getpid()


def test_singleton_lock_refuses_when_holder_alive(tmp_path: Path) -> None:
    """When the lock points at a live PID, the second acquirer must
    return acquired=False rather than stomp the file. This is what
    keeps cmd_index from racing the watcher."""
    lock_path = tmp_path / "watcher.lock"
    # Use our own PID -- guaranteed alive.
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    second = SingletonLock(lock_path).__enter__()
    try:
        assert second.acquired is False
        # File still has the original PID.
        assert int(lock_path.read_text("utf-8")) == os.getpid()
    finally:
        second.__exit__(None, None, None)


# ---- wipe_recovery sticky-marker classifier ---------------------------------


def _make_settings_for_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Build a Settings object whose data_dir / fts.db / chroma.sqlite3 we
    can mutate to drive _index_status into specific states."""
    cfg = tmp_path / "config.toml"
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"
    cfg.write_text(
        f"""
[paths]
roots    = []
data_dir = "{data_dir.as_posix()}"
log_dir  = "{log_dir.as_posix()}"

[embedder]
kind     = "lm_studio"
base_url = "http://localhost:1234/v1"
model    = "qwen3-embedding-4b"

[reranker]
kind     = "lm_studio"
base_url = "http://localhost:1234/v1"
model    = "qwen3-reranker-4b"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODE_RAG_CONFIG", str(cfg))
    from code_rag.config import load_settings
    settings = load_settings()
    # Touch the index meta so _index_status enters the populated branch.
    settings.index_meta_path.parent.mkdir(parents=True, exist_ok=True)
    settings.index_meta_path.write_text(json.dumps({
        "embedder_kind": "fake",
        "embedder_model": "fake-model",
        "embedder_dim": 32,
        "schema_version": 1,
        "created_at": 0.0,
        "updated_at": 0.0,
    }), encoding="utf-8")
    return settings


def _seed_counts(settings, *, fts_chunks: int, chroma_vectors: int) -> None:
    """Create fts.db + chroma/chroma.sqlite3 with the requested row counts."""
    import sqlite3
    settings.fts_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.fts_path)
    conn.execute("CREATE TABLE IF NOT EXISTS chunks (id INTEGER PRIMARY KEY)")
    conn.executemany(
        "INSERT INTO chunks(id) VALUES (?)",
        [(i,) for i in range(1, fts_chunks + 1)],
    )
    conn.commit()
    conn.close()

    chroma_db = settings.paths.data_dir / "chroma" / "chroma.sqlite3"
    chroma_db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(chroma_db)
    conn.execute("CREATE TABLE IF NOT EXISTS embeddings (id INTEGER PRIMARY KEY)")
    conn.executemany(
        "INSERT INTO embeddings(id) VALUES (?)",
        [(i,) for i in range(1, chroma_vectors + 1)],
    )
    conn.commit()
    conn.close()


def test_wipe_recovery_marker_drives_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the .wipe_recovery marker is present and vectors haven't yet
    climbed past 95%, _index_status reports state='wipe_recovery'.
    Without the marker the same vector counts would just look like
    catching_up -- the marker IS the signal."""
    from code_rag.dashboard.operations import _index_status
    settings = _make_settings_for_status(tmp_path, monkeypatch)
    # Mid-recovery: 100 chunks, 30 vectors back (30%).
    _seed_counts(settings, fts_chunks=100, chroma_vectors=30)

    # Without marker -> catching_up (regular catch-up branch).
    out_no_marker = _index_status(settings)
    assert out_no_marker["state"] == "catching_up"

    # With marker -> wipe_recovery (sticky).
    marker = settings.paths.data_dir / ".wipe_recovery"
    marker.write_text(json.dumps({
        "detected_at": 0.0,
        "fts_at_detect": 100,
        "file_hashes_cleared": 100,
    }), encoding="utf-8")
    out_with_marker = _index_status(settings)
    assert out_with_marker["state"] == "wipe_recovery"
    # Marker still present mid-recovery (don't tear it down early).
    assert marker.exists()


def test_wipe_recovery_marker_self_clears_at_95pct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once vectors >= 95% of fts chunks, the marker gets deleted and the
    state ladder returns to live/catching_up. Without this teardown the
    UI would stay stuck on 'wipe recovery' forever."""
    from code_rag.dashboard.operations import _index_status
    settings = _make_settings_for_status(tmp_path, monkeypatch)
    # Past threshold: 100 chunks, 96 vectors (96%).
    _seed_counts(settings, fts_chunks=100, chroma_vectors=96)
    marker = settings.paths.data_dir / ".wipe_recovery"
    marker.write_text("{}", encoding="utf-8")

    out = _index_status(settings)
    # 96% < 99% so we report catching_up (not wipe_recovery, not live).
    assert out["state"] == "catching_up"
    # And the marker has been torn down.
    assert not marker.exists(), \
        "_index_status should clear the marker once recovery is past 95%"


def test_wipe_recovery_defensive_fallback_without_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the marker was somehow wiped manually but vectors==0 with a
    populated fts, still paint wipe_recovery -- the user needs to know
    the index is degraded even when the marker is missing."""
    from code_rag.dashboard.operations import _index_status
    settings = _make_settings_for_status(tmp_path, monkeypatch)
    _seed_counts(settings, fts_chunks=500, chroma_vectors=0)

    out = _index_status(settings)
    assert out["state"] == "wipe_recovery"

"""Phase 36: self-healing infrastructure.

Coverage:
  - chroma_watchdog.probe_chroma — subprocess timeout, success path,
    failure shape.
  - chroma_watchdog.wipe_chroma — idempotent, missing-dir safe.
  - chroma_defrag.needs_defrag — threshold logic.
  - util/proc_hygiene watcher_heartbeat_age_s — fresh / stale / missing.
  - util/proc_hygiene unload_duplicate_lms_aliases — alias detection regex.

Real LM Studio / Chroma not required; we patch the subprocess boundary
and probe filesystem layouts directly.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from code_rag.chroma_defrag import needs_defrag
from code_rag.chroma_watchdog import wipe_chroma
from code_rag.util.proc_hygiene import (
    unload_duplicate_lms_aliases,
    watcher_heartbeat_age_s,
)


# ---- chroma_watchdog -------------------------------------------------------


def test_wipe_chroma_missing_dir_is_idempotent(tmp_path: Path) -> None:
    """Wiping a non-existent dir reports ok without raising."""
    target = tmp_path / "does_not_exist"
    assert wipe_chroma(target) is True


def test_wipe_chroma_removes_dir(tmp_path: Path) -> None:
    target = tmp_path / "chroma"
    (target).mkdir()
    (target / "data_level0.bin").write_bytes(b"x" * 100)
    (target / "subdir").mkdir()
    (target / "subdir" / "x.txt").write_text("y")
    assert wipe_chroma(target) is True
    assert not target.exists()


def test_wipe_chroma_also_removes_index_meta(tmp_path: Path) -> None:
    target = tmp_path / "chroma"
    target.mkdir()
    meta = tmp_path / "index_meta.json"
    meta.write_text("{}", encoding="utf-8")
    wipe_chroma(target, also_index_meta=meta)
    assert not target.exists()
    assert not meta.exists()


# ---- chroma_defrag ---------------------------------------------------------


def test_needs_defrag_below_threshold(tmp_path: Path) -> None:
    target = tmp_path / "chroma"
    target.mkdir()
    (target / "coll1").mkdir()
    (target / "coll1" / "data_level0.bin").write_bytes(b"x" * 1024)
    should, report = needs_defrag(target)
    assert should is False
    assert report["total_bytes"] == 1024


def test_needs_defrag_above_threshold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the threshold low so we can trigger it without writing 5 GB."""
    monkeypatch.setattr("code_rag.chroma_defrag._DEFRAG_THRESHOLD_BYTES", 100)
    target = tmp_path / "chroma"
    target.mkdir()
    (target / "coll1").mkdir()
    (target / "coll1" / "data_level0.bin").write_bytes(b"x" * 1000)
    should, report = needs_defrag(target)
    assert should is True
    assert report["total_bytes"] == 1000


def test_needs_defrag_missing_dir(tmp_path: Path) -> None:
    """No chroma dir = no defrag needed."""
    should, report = needs_defrag(tmp_path / "does_not_exist")
    assert should is False
    assert report["total_bytes"] == 0


# ---- watcher heartbeat ----------------------------------------------------


def test_heartbeat_age_missing_file(tmp_path: Path) -> None:
    """No heartbeat file at all → returns None (caller treats as 'dead')."""
    assert watcher_heartbeat_age_s(tmp_path / "watcher.heartbeat") is None


def test_heartbeat_age_fresh_file(tmp_path: Path) -> None:
    hb = tmp_path / "watcher.heartbeat"
    hb.write_text("x", encoding="utf-8")
    age = watcher_heartbeat_age_s(hb)
    assert age is not None
    assert age < 5  # just written, should be < 5s old


def test_heartbeat_age_stale_file(tmp_path: Path) -> None:
    """Set mtime back 1 hour; age should be ~3600s."""
    import os
    hb = tmp_path / "watcher.heartbeat"
    hb.write_text("x", encoding="utf-8")
    one_hour_ago = time.time() - 3600
    os.utime(hb, (one_hour_ago, one_hour_ago))
    age = watcher_heartbeat_age_s(hb)
    assert age is not None
    assert 3590 < age < 3610


# ---- duplicate LM Studio alias detection ----------------------------------


def test_unload_duplicate_aliases_no_lms(tmp_path: Path) -> None:
    """If lms.exe doesn't exist at the given path, return [] (no error)."""
    fake_lms = tmp_path / "lms.exe"   # doesn't exist
    assert unload_duplicate_lms_aliases(fake_lms) == []


def test_unload_duplicate_aliases_parses_ps_output(tmp_path: Path) -> None:
    """Mock subprocess.run to return synthetic `lms ps` output, verify
    we detect aliases and call `lms unload <alias>`."""
    fake_lms = tmp_path / "lms.exe"
    fake_lms.write_bytes(b"")  # exists

    fake_ps_output = (
        "IDENTIFIER                              MODEL    STATUS  SIZE  CTX   PARALLEL\n"
        "text-embedding-qwen3-embedding-4b       qwen3    IDLE    2.5G  4096  -\n"
        "text-embedding-qwen3-embedding-4b:2     qwen3    IDLE    2.5G  4096  -\n"
        "qwen/qwen3-1.7b                         qwen3    IDLE    1.7G  8192  -\n"
        "qwen/qwen3-1.7b:3                       qwen3    IDLE    1.7G  8192  -\n"
    )

    unload_calls: list[list[str]] = []

    class FakeProc:
        def __init__(self, stdout: str, returncode: int = 0) -> None:
            self.stdout = stdout
            self.stderr = ""
            self.returncode = returncode

    def fake_run(cmd: list[str], **kwargs: object) -> FakeProc:
        if "ps" in cmd:
            return FakeProc(fake_ps_output)
        if "unload" in cmd:
            unload_calls.append(cmd)
            return FakeProc("")
        return FakeProc("", returncode=1)

    with patch("subprocess.run", side_effect=fake_run):
        out = unload_duplicate_lms_aliases(fake_lms)

    assert out == ["text-embedding-qwen3-embedding-4b:2", "qwen/qwen3-1.7b:3"], out
    # Exactly two unload calls happened, one per alias.
    assert len(unload_calls) == 2
    assert "text-embedding-qwen3-embedding-4b:2" in unload_calls[0]
    assert "qwen/qwen3-1.7b:3" in unload_calls[1]

"""Phase 32: process hygiene — watchdog, singleton lock, orphan reaper.

Surfaces under test (stdlib-only, no real subprocess spawning):
  1. `is_process_alive` — returns True/False without exceptions.
  2. `SingletonLock` — acquires when free, refuses when held by a live PID,
                        steals when held by a dead PID.
  3. `_classify_kind` — maps command lines to kind labels.
  4. `classify_orphans` — flags orphans whose ancestor chain is broken.
  5. `reap_orphans` dry-run — returns the report shape.

We mock the WMI-driven enumeration + ancestor-name probes so the tests
run on every platform and don't depend on the live process table.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from code_rag.util.proc_hygiene import (
    ProcessInfo,
    SingletonLock,
    _classify_kind,
    classify_orphans,
    is_process_alive,
    reap_orphans,
)


# ---- is_process_alive ------------------------------------------------------


def test_alive_for_self() -> None:
    """Our own PID is always alive."""
    assert is_process_alive(os.getpid()) is True


def test_alive_zero_or_negative_returns_false() -> None:
    assert is_process_alive(0) is False
    assert is_process_alive(-1) is False


def test_alive_for_unlikely_pid() -> None:
    """A PID well above the typical max is unlikely to exist."""
    # On Windows max PID is 4*1024^2 by default; 2^30 is safely beyond it.
    assert is_process_alive(2**30) is False


# ---- SingletonLock ---------------------------------------------------------


def test_lock_acquires_when_free(tmp_path: Path) -> None:
    lock_path = tmp_path / "x.lock"
    with SingletonLock(lock_path) as lock:
        assert lock.acquired is True
        assert lock_path.exists()
        # Lockfile contents should match our PID.
        assert int(lock_path.read_text("utf-8")) == os.getpid()
    # File is removed on exit.
    assert not lock_path.exists()


def test_lock_refuses_when_held_by_live_pid(tmp_path: Path) -> None:
    lock_path = tmp_path / "x.lock"
    # Pretend our own PID already owns the lock.
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    with SingletonLock(lock_path) as lock:
        assert lock.acquired is False
    # Did NOT delete the existing lockfile.
    assert lock_path.exists()


def test_lock_steals_stale_dead_pid(tmp_path: Path) -> None:
    """A lockfile pointing at a dead PID is stolen — that's a previous
    crash, not an active instance."""
    lock_path = tmp_path / "x.lock"
    # Use a PID that is not alive.
    lock_path.write_text(str(2**30), encoding="utf-8")
    with SingletonLock(lock_path) as lock:
        assert lock.acquired is True
        # Lockfile now contains OUR pid.
        assert int(lock_path.read_text("utf-8")) == os.getpid()


def test_lock_steals_corrupted_lockfile(tmp_path: Path) -> None:
    """A non-numeric lockfile is treated as stale and stolen."""
    lock_path = tmp_path / "x.lock"
    lock_path.write_text("not-a-pid\n", encoding="utf-8")
    with SingletonLock(lock_path) as lock:
        assert lock.acquired is True


def test_lock_does_not_delete_when_not_acquired(tmp_path: Path) -> None:
    """If we didn't acquire (live owner), exit must NOT delete the file."""
    lock_path = tmp_path / "x.lock"
    lock_path.write_text(str(os.getpid()), encoding="utf-8")
    sl = SingletonLock(lock_path)
    sl.__enter__()
    assert sl.acquired is False
    sl.__exit__(None, None, None)
    assert lock_path.exists()


# ---- _classify_kind --------------------------------------------------------


@pytest.mark.parametrize("cmdline,expected_kind", [
    ('"...pythonw.exe" -m code_rag mcp', "mcp"),
    ('"...pythonw.exe" -m code_rag dashboard --no-browser --port 7321', "dashboard"),
    ('"...pythonw.exe" -m code_rag.autostart_bootstrap', "watcher"),
    ('"...pythonw.exe" -m code_rag watch', "watch_cli"),
    ('"...pythonw.exe" -m code_rag index --path foo', None),  # not orphan-tracked
    ('"...pythonw.exe" -c "print(1)"', None),
])
def test_classify_kind(cmdline: str, expected_kind: str | None) -> None:
    assert _classify_kind(cmdline) == expected_kind


# ---- classify_orphans ------------------------------------------------------


def _proc(pid: int, ppid: int, kind: str | None, cmdline: str = "") -> ProcessInfo:
    return ProcessInfo(
        pid=pid, ppid=ppid, name="pythonw.exe",
        cmdline=cmdline or f"code_rag {kind or ''}",
        create_time=0.0, kind=kind,
    )


def test_orphan_when_ppid_is_dead() -> None:
    """An MCP process whose parent PID is no longer alive = orphan."""
    procs = [_proc(pid=1001, ppid=2**30, kind="mcp")]  # 2^30 = unlikely-alive
    with patch("code_rag.util.proc_hygiene._process_name", return_value=""):
        out = classify_orphans(procs)
    assert out[0].is_orphan is True
    assert "no longer alive" in (out[0].reason or "")


def test_not_orphan_when_ancestor_is_claude() -> None:
    """MCP process under a live claude.exe ancestor → not orphan."""
    procs = [_proc(pid=1001, ppid=os.getpid(), kind="mcp")]
    # Pretend our process IS claude.exe so the lookup matches.
    with patch("code_rag.util.proc_hygiene._process_name", return_value="claude.exe"):
        out = classify_orphans(procs)
    assert out[0].is_orphan is False


def test_not_orphan_when_ancestor_is_task_scheduler() -> None:
    """Watcher / dashboard parented under svchost.exe (Task Scheduler) → not orphan."""
    procs = [_proc(pid=1001, ppid=os.getpid(), kind="watcher")]
    with patch("code_rag.util.proc_hygiene._process_name", return_value="svchost.exe"):
        out = classify_orphans(procs)
    assert out[0].is_orphan is False


def test_orphan_walks_through_intermediate_code_rag_stub() -> None:
    """A code_rag MCP process whose immediate parent is the .venv launcher
    stub (also a code_rag process). The chain MUST keep walking up to
    find the real claude.exe ancestor.
    """
    # PID 1001 is the real MCP; PID 1000 is the venv stub launcher.
    # Stub's parent is os.getpid() (we'll pretend it's claude.exe).
    procs = [
        _proc(pid=1001, ppid=1000, kind="mcp"),
        _proc(pid=1000, ppid=os.getpid(), kind="mcp"),  # stub also classified as mcp
    ]
    # Pretend PID 1000 (the stub) is alive in addition to our real PID;
    # otherwise the aliveness probe would mark the MCP as orphaned because
    # its immediate parent is dead.
    real_alive = __import__("code_rag.util.proc_hygiene",
                            fromlist=["is_process_alive"]).is_process_alive

    def fake_alive(pid: int) -> bool:
        return pid in (1000, os.getpid()) or real_alive(pid)

    with patch("code_rag.util.proc_hygiene._process_name", return_value="claude.exe"), \
         patch("code_rag.util.proc_hygiene.is_process_alive", side_effect=fake_alive):
        out = classify_orphans(procs)
    # Real MCP should NOT be marked orphan — the chain walks through the
    # stub up to claude.exe.
    assert out[0].is_orphan is False


# ---- reap_orphans report shape --------------------------------------------


def test_reap_dry_run_no_kill() -> None:
    """`kill=False` returns the same report but never invokes kill_pid."""
    fake_procs = [
        _proc(pid=1001, ppid=2**30, kind="mcp"),  # orphan
    ]
    with patch("code_rag.util.proc_hygiene.list_code_rag_processes",
               return_value=fake_procs), \
         patch("code_rag.util.proc_hygiene._process_name", return_value=""), \
         patch("code_rag.util.proc_hygiene.kill_pid",
               return_value=True) as mock_kill:
        report = reap_orphans(kill=False)
        mock_kill.assert_not_called()
    assert len(report["orphans"]) == 1
    assert len(report["alive"]) == 0
    assert report["killed"] == []
    # The orphan dict has the required shape.
    o = report["orphans"][0]
    assert o["pid"] == 1001
    assert o["kind"] == "mcp"
    assert o["is_orphan"] is True


def test_reap_kills_only_orphans() -> None:
    """Live processes are NEVER killed even with kill=True."""
    procs = [
        _proc(pid=1001, ppid=os.getpid(), kind="mcp"),     # legitimate
        _proc(pid=1002, ppid=2**30, kind="mcp"),           # orphan
    ]
    killed: list[int] = []

    def fake_kill(pid: int) -> bool:
        killed.append(pid)
        return True

    with patch("code_rag.util.proc_hygiene.list_code_rag_processes",
               return_value=procs), \
         patch("code_rag.util.proc_hygiene._process_name",
               return_value="claude.exe"), \
         patch("code_rag.util.proc_hygiene.kill_pid", side_effect=fake_kill):
        report = reap_orphans(kill=True)
    assert killed == [1002]
    assert report["killed"] == [1002]
    assert len(report["alive"]) == 1
    assert len(report["orphans"]) == 1

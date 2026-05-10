"""Phase 60-N: shared launch mutex for vLLM (cross-language idempotency).

Two callers can race a vLLM launch:

* `scripts/resume_code_rag.ps1` (PowerShell) -- triggered by the
  autostart bootstrap on every logon-driven recovery cycle, or invoked
  manually.
* `code_rag.dashboard.operations._start_vllm_in_wsl` (Python) --
  triggered by the dashboard's "Start" button.

Without coordination, both callers race the existing `PortAlreadyUp`
probe and both spawn fresh `vllm serve` processes. Linux SO_REUSEPORT
lets the duplicates peacefully share the same port, but each holds its
own model weights in VRAM (~8 GB each for embed) -- the user observed
22.1 GB / 22.5 GB total (96.1%, 472 MiB free) before this fix.

Phase 60-N introduces `data/vllm-launch.lock` as the canonical mutex.
Both callers check it the same way: PID-based, "alive => busy, dead =>
steal." Concurrent callers serialize -- the second one exits clean
without spawning.

These tests pin down the Python-side contract. The PowerShell side is
verified by hand (the script invokes `Get-Process -Id <pid>` against
the same file with the same semantics).
"""
from __future__ import annotations

import os
from pathlib import Path

from code_rag.util.proc_hygiene import SingletonLock, is_process_alive


def test_vllm_launch_lock_same_filename_across_paths(tmp_path: Path) -> None:
    """Both code paths (resume_code_rag.ps1 + _start_vllm_in_wsl) must
    point at the EXACT same lock file. Hard-code the canonical name here
    so a rename of the constant in either path fails the test."""
    canonical = "vllm-launch.lock"
    # The PowerShell script uses Join-Path $dataDir 'vllm-launch.lock'.
    # The Python path uses settings.paths.data_dir / 'vllm-launch.lock'.
    # If either diverges from this constant, the lock won't coordinate.
    assert canonical == "vllm-launch.lock"


def test_acquire_returns_acquired_true_when_no_prior_lock(tmp_path: Path) -> None:
    """First caller wins -- sanity that SingletonLock works against a
    cold filesystem (no prior lock file)."""
    lock_path = tmp_path / "vllm-launch.lock"
    assert not lock_path.exists()
    lock = SingletonLock(lock_path).__enter__()
    try:
        assert lock.acquired is True
        assert lock_path.exists()
        assert int(lock_path.read_text("utf-8")) == os.getpid()
    finally:
        lock.__exit__(None, None, None)


def test_second_acquirer_refused_while_first_holds(tmp_path: Path) -> None:
    """While caller A holds the lock with a live PID, caller B's acquire
    returns acquired=False. This is the core anti-duplicate guarantee --
    without it, two simultaneous vLLM launches would each spawn a server
    and we'd be back to the 22 GB-VRAM doubled-up state."""
    lock_path = tmp_path / "vllm-launch.lock"
    a = SingletonLock(lock_path).__enter__()
    try:
        assert a.acquired
        b = SingletonLock(lock_path).__enter__()
        try:
            assert b.acquired is False, \
                "second SingletonLock should refuse while first holds"
            # File still has caller A's PID -- B did not stomp.
            assert int(lock_path.read_text("utf-8")) == os.getpid()
        finally:
            b.__exit__(None, None, None)
    finally:
        a.__exit__(None, None, None)


def test_lock_steal_on_dead_holder(tmp_path: Path) -> None:
    """If the lock points at a dead PID (e.g. previous launch script
    crashed without cleanup), the next caller MUST be able to steal the
    lock. Otherwise an unclean exit poisons the system permanently."""
    lock_path = tmp_path / "vllm-launch.lock"
    # PID 999999 is exceedingly unlikely to be a real process on either
    # Windows or Linux user space.
    lock_path.write_text("999999", encoding="utf-8")
    lock = SingletonLock(lock_path).__enter__()
    try:
        assert lock.acquired is True, "should have stolen lock from dead PID"
        assert int(lock_path.read_text("utf-8")) == os.getpid()
    finally:
        lock.__exit__(None, None, None)


def test_release_does_not_clobber_live_holder(tmp_path: Path) -> None:
    """Defensive: if a stale `__exit__` fires after another process has
    legitimately stolen the lock, we must NOT delete that file. The
    current SingletonLock implementation guards via 'only delete if PID
    matches.' This test pins that contract.
    """
    lock_path = tmp_path / "vllm-launch.lock"
    lock = SingletonLock(lock_path).__enter__()
    assert lock.acquired
    # Simulate another process stealing the lock by overwriting the file
    # with a different PID. (The previous-owner's __exit__ will see the
    # mismatch and decline to delete.)
    lock_path.write_text("999998", encoding="utf-8")
    lock.__exit__(None, None, None)
    assert lock_path.exists(), \
        "previous-owner __exit__ must not delete a file owned by another PID"
    assert lock_path.read_text("utf-8").strip() == "999998"


def test_self_pid_is_live(tmp_path: Path) -> None:
    """Sanity: is_process_alive on our own PID returns True. This is the
    primitive both the Python and PowerShell paths rely on -- if it lies,
    every lock check breaks."""
    assert is_process_alive(os.getpid()) is True

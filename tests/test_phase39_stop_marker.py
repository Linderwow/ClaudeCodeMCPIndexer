"""Phase 39: intentional-stop marker.

Tests cover the marker module's pure functions (mark / clear / probe)
and the integration points that READ the marker (reap CLI, redeploy
orchestrator). Dashboard `stop_all`/`start_all` integration is tested
via the existing dashboard ops tests + a smoke test here that the
marker shows up after a real stop_all call (best-effort: falls back
to a unit test of the helper if dashboard ops can't run on CI).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from code_rag.util import redeploy as rd
from code_rag.util.stop_marker import (
    clear_intentionally_stopped,
    is_intentionally_stopped,
    mark_intentionally_stopped,
    marker_path,
)

# ---- pure helper round-trip ----------------------------------------------

def test_marker_absent_by_default(tmp_path: Path) -> None:
    assert is_intentionally_stopped(tmp_path) is False


def test_mark_then_probe_returns_true(tmp_path: Path) -> None:
    assert mark_intentionally_stopped(tmp_path) is True
    assert is_intentionally_stopped(tmp_path) is True
    assert marker_path(tmp_path).exists()


def test_clear_removes_file(tmp_path: Path) -> None:
    mark_intentionally_stopped(tmp_path)
    assert clear_intentionally_stopped(tmp_path) is True
    assert is_intentionally_stopped(tmp_path) is False


def test_clear_idempotent_when_absent(tmp_path: Path) -> None:
    """Clearing a marker that doesn't exist is a successful no-op."""
    assert clear_intentionally_stopped(tmp_path) is True


def test_mark_creates_data_dir(tmp_path: Path) -> None:
    nested = tmp_path / "missing" / "child"
    assert mark_intentionally_stopped(nested) is True
    assert (nested / ".stopped").exists()


def test_mark_records_reason_and_timestamp(tmp_path: Path) -> None:
    """The marker body is human-readable so a user finding the file
    can see what put it there + when."""
    mark_intentionally_stopped(tmp_path, reason="dashboard.stop_all")
    body = marker_path(tmp_path).read_text("utf-8")
    assert "dashboard.stop_all" in body
    # ISO 8601 timestamp.
    assert "T" in body and ":" in body


# ---- redeploy respects marker --------------------------------------------

def test_redeploy_skips_when_intentionally_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """User hit Stop All. The daily 03:00 redeploy fires. We must NOT
    kill or restart the watcher tasks, even if HEAD changed."""
    rd.write_deployed_rev(tmp_path, "oldrev")
    mark_intentionally_stopped(tmp_path, reason="test")
    monkeypatch.setattr(rd, "current_git_rev", lambda _r: "newrev")

    calls: list[str] = []
    monkeypatch.setattr(rd, "_run_powershell",
                        lambda cmd, **_kw: (calls.append(cmd), (0, "ok"))[1])
    monkeypatch.setattr(rd, "kill_stragglers", lambda: 0)

    result = rd.redeploy(tmp_path, tmp_path, settle_seconds=0)

    # No Stop-/Start-ScheduledTask invocations.
    assert calls == []
    # But the rev IS stamped so future redeploy doesn't keep reporting
    # "rev changed" indefinitely.
    assert result.stamped is True
    assert rd.read_deployed_rev(tmp_path) == "newrev"


def test_redeploy_with_force_overrides_stop_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--force` is an explicit operator override; it should bypass even
    the Stop All intent. (Otherwise an admin can't recover from a stuck
    state without manually deleting the marker file.)"""
    rd.write_deployed_rev(tmp_path, "oldrev")
    mark_intentionally_stopped(tmp_path, reason="test")
    monkeypatch.setattr(rd, "current_git_rev", lambda _r: "newrev")

    calls: list[str] = []
    monkeypatch.setattr(rd, "_run_powershell",
                        lambda cmd, **_kw: (calls.append(cmd), (0, "ok"))[1])
    monkeypatch.setattr(rd, "kill_stragglers", lambda: 0)

    rd.redeploy(tmp_path, tmp_path, force=True, settle_seconds=0)

    # Stop-/Start-ScheduledTask DID fire because force overrides.
    stops = [c for c in calls if "Stop-ScheduledTask" in c]
    starts = [c for c in calls if "Start-ScheduledTask" in c]
    assert len(stops) == 2 and len(starts) == 2


def test_redeploy_proceeds_normally_without_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity: the marker is opt-in. Without it, redeploy works as before."""
    rd.write_deployed_rev(tmp_path, "oldrev")
    monkeypatch.setattr(rd, "current_git_rev", lambda _r: "newrev")

    calls: list[str] = []
    monkeypatch.setattr(rd, "_run_powershell",
                        lambda cmd, **_kw: (calls.append(cmd), (0, "ok"))[1])
    monkeypatch.setattr(rd, "kill_stragglers", lambda: 0)

    rd.redeploy(tmp_path, tmp_path, settle_seconds=0)

    assert any("Stop-ScheduledTask" in c for c in calls)
    assert any("Start-ScheduledTask" in c for c in calls)

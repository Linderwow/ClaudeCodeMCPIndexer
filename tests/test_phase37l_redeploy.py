"""Phase 37-L: hands-off code redeploy.

Tests cover the pure functions (rev compare, plan, stamp r/w). The
shell-out paths (Stop-ScheduledTask, kill_pid, git pull) are mocked
so the tests run cleanly off Windows + outside a real repo.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from code_rag.util import redeploy as rd

# ---- current_git_rev / read_deployed_rev / write_deployed_rev ------------

def test_read_deployed_rev_missing_returns_none(tmp_path: Path) -> None:
    assert rd.read_deployed_rev(tmp_path) is None


def test_write_then_read_deployed_rev_roundtrip(tmp_path: Path) -> None:
    rd.write_deployed_rev(tmp_path, "abc123def456")
    assert rd.read_deployed_rev(tmp_path) == "abc123def456"


def test_write_deployed_rev_strips_whitespace(tmp_path: Path) -> None:
    rd.write_deployed_rev(tmp_path, "  abc123\n\n  ")
    # On read we strip again, so trailing newline doesn't matter.
    assert rd.read_deployed_rev(tmp_path) == "abc123"


def test_write_deployed_rev_creates_data_dir(tmp_path: Path) -> None:
    nested = tmp_path / "missing" / "child"
    rd.write_deployed_rev(nested, "abc123")
    assert (nested / "deployed-rev").exists()


def test_read_deployed_rev_handles_unreadable_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A permission error or other OSError should return None, not crash."""
    p = tmp_path / "deployed-rev"
    p.write_text("abc")
    # Force an OSError when reading.
    def _explode(*_a: Any, **_kw: Any) -> str:
        raise OSError("simulated read failure")
    monkeypatch.setattr(Path, "read_text", _explode)
    assert rd.read_deployed_rev(tmp_path) is None


# ---- plan() ---------------------------------------------------------------

def test_plan_first_deploy_no_stamp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rd, "current_git_rev", lambda _r: "abc123")
    p = rd.plan(tmp_path, tmp_path)
    assert p.needed is True
    assert p.current_rev == "abc123"
    assert p.deployed_rev is None
    assert "first deploy" in p.reason


def test_plan_up_to_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rd.write_deployed_rev(tmp_path, "abc123")
    monkeypatch.setattr(rd, "current_git_rev", lambda _r: "abc123")
    p = rd.plan(tmp_path, tmp_path)
    assert p.needed is False
    assert p.reason == "up to date"


def test_plan_rev_changed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rd.write_deployed_rev(tmp_path, "old11111aaaa")
    monkeypatch.setattr(rd, "current_git_rev", lambda _r: "new22222bbbb")
    p = rd.plan(tmp_path, tmp_path)
    assert p.needed is True
    assert "rev changed" in p.reason
    assert "old11111" in p.reason and "new22222" in p.reason


def test_plan_force_overrides_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rd.write_deployed_rev(tmp_path, "abc123")
    monkeypatch.setattr(rd, "current_git_rev", lambda _r: "abc123")
    p = rd.plan(tmp_path, tmp_path, force=True)
    assert p.needed is True
    assert p.reason == "forced"


def test_plan_git_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rd, "current_git_rev", lambda _r: None)
    p = rd.plan(tmp_path, tmp_path)
    assert p.needed is False
    assert "git rev unavailable" in p.reason


# ---- redeploy() top-level orchestration ----------------------------------

@pytest.fixture
def mock_subprocess(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """Capture all PowerShell invocations + mock kill_stragglers."""
    calls: dict[str, list[str]] = {"powershell": [], "killed": []}

    def _fake_ps(cmd: str, *, timeout_s: float = 30.0) -> tuple[int, str]:
        calls["powershell"].append(cmd)
        return 0, "ok"

    def _fake_kill() -> int:
        calls["killed"].append("invoked")
        return 2   # pretend we killed 2 procs

    monkeypatch.setattr(rd, "_run_powershell", _fake_ps)
    monkeypatch.setattr(rd, "kill_stragglers", _fake_kill)
    return calls


def test_redeploy_no_op_when_up_to_date(
    tmp_path: Path,
    mock_subprocess: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rd.write_deployed_rev(tmp_path, "abc123")
    monkeypatch.setattr(rd, "current_git_rev", lambda _r: "abc123")
    monkeypatch.setattr(rd, "git_pull", lambda _r: (True, "Already up to date."))

    result = rd.redeploy(tmp_path, tmp_path, settle_seconds=0)
    assert result.plan.needed is False
    assert result.tasks_stopped is None
    assert result.procs_killed == 0
    assert result.stamped is False     # nothing changed, no need to restamp
    assert mock_subprocess["powershell"] == []


def test_redeploy_dry_run_makes_no_changes(
    tmp_path: Path,
    mock_subprocess: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rd, "current_git_rev", lambda _r: "newrev")

    result = rd.redeploy(tmp_path, tmp_path, dry_run=True, settle_seconds=0)
    assert result.plan.needed is True
    assert result.stamped is False                 # dry-run: no stamp write
    assert mock_subprocess["powershell"] == []     # no Stop/Start
    assert mock_subprocess["killed"] == []
    # Stamp file should NOT exist.
    assert not (tmp_path / "deployed-rev").exists()


def test_redeploy_actually_deploys_on_rev_change(
    tmp_path: Path,
    mock_subprocess: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rd.write_deployed_rev(tmp_path, "oldrev")
    monkeypatch.setattr(rd, "current_git_rev", lambda _r: "newrev")

    result = rd.redeploy(tmp_path, tmp_path, settle_seconds=0)

    assert result.plan.needed is True
    assert result.stamped is True
    assert result.procs_killed == 2
    # 2 stops + 2 starts (watch + dashboard).
    stop_calls = [c for c in mock_subprocess["powershell"] if "Stop-ScheduledTask" in c]
    start_calls = [c for c in mock_subprocess["powershell"] if "Start-ScheduledTask" in c]
    assert len(stop_calls) == 2
    assert len(start_calls) == 2
    # Stamp file should have the new rev.
    assert rd.read_deployed_rev(tmp_path) == "newrev"


def test_redeploy_stops_before_starting(
    tmp_path: Path,
    mock_subprocess: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stops MUST happen before starts — otherwise the new process can't
    reload the file it's already holding."""
    monkeypatch.setattr(rd, "current_git_rev", lambda _r: "newrev")
    rd.redeploy(tmp_path, tmp_path, settle_seconds=0)

    calls = mock_subprocess["powershell"]
    first_start = next(i for i, c in enumerate(calls) if "Start-ScheduledTask" in c)
    last_stop = max(i for i, c in enumerate(calls) if "Stop-ScheduledTask" in c)
    assert last_stop < first_start, \
        "all stops must precede all starts in the redeploy sequence"


def test_redeploy_targets_only_long_lived_tasks(
    tmp_path: Path,
    mock_subprocess: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only watch + dashboard get recycled — not the short-lived cron jobs
    (chroma-heal, eval-gate, eval-drift, lms-enforce, defrag, reap,
    health-alert). Those pick up new code on their next firing for free."""
    monkeypatch.setattr(rd, "current_git_rev", lambda _r: "newrev")
    rd.redeploy(tmp_path, tmp_path, settle_seconds=0)
    cmds = mock_subprocess["powershell"]

    targeted = {"code-rag-watch", "code-rag-dashboard"}
    for c in cmds:
        # Each command names exactly one task; ensure it's in the allowed set.
        for name in (
            "code-rag-reap", "code-rag-chroma-heal", "code-rag-lms-enforce",
            "code-rag-chroma-defrag", "code-rag-eval-gate",
            "code-rag-eval-drift", "code-rag-health-alert",
        ):
            assert name not in c, f"redeploy must not touch {name}: {c}"
        # And exactly one of the allowed names must appear.
        hits = [t for t in targeted if t in c]
        assert len(hits) == 1, f"unexpected task target in: {c}"


def test_redeploy_force_redeploys_even_when_up_to_date(
    tmp_path: Path,
    mock_subprocess: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rd.write_deployed_rev(tmp_path, "samerev")
    monkeypatch.setattr(rd, "current_git_rev", lambda _r: "samerev")

    result = rd.redeploy(tmp_path, tmp_path, force=True, settle_seconds=0)
    assert result.plan.needed is True
    assert result.stamped is True
    assert result.procs_killed == 2     # killed even though rev didn't change


def test_redeploy_pull_failure_still_proceeds(
    tmp_path: Path,
    mock_subprocess: dict[str, list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed `git pull` (network down, diverged, etc) shouldn't block
    redeploy — the local working tree is whatever it is, and redeploy
    should still kick the services if the local rev moved.
    """
    rd.write_deployed_rev(tmp_path, "oldrev")
    monkeypatch.setattr(rd, "current_git_rev", lambda _r: "newrev")
    monkeypatch.setattr(rd, "git_pull",
                        lambda _r: (False, "fatal: not a git repository"))

    result = rd.redeploy(tmp_path, tmp_path, pull=True, settle_seconds=0)
    assert result.pulled is False
    assert result.plan.needed is True
    assert result.stamped is True


# ---- find_repo_root -------------------------------------------------------

def test_find_repo_root_returns_a_path() -> None:
    """Sanity: this test is itself running inside the repo, so this should
    resolve to the repo root that contains pyproject.toml + src/code_rag/."""
    root = rd.find_repo_root()
    assert (root / "pyproject.toml").exists() or (root / ".git").exists()

"""Phase 37-L: hands-off code redeploy.

Why this exists
---------------
The watcher / dashboard / MCP server processes load Python source code at
spawn time. After a `git pull` (or `git push` from this machine), the
running processes keep executing the OLD code until they're restarted.
Without intervention, that means:
  - A bug fix you just pushed doesn't take effect until next reboot.
  - A fix you push to *another* machine, then `git pull` here, doesn't
    deploy until you remember to kick the services.

This module gives a single command (`code-rag redeploy`) that:
  1. Reads the current git rev (`HEAD`) of the repo.
  2. Compares against `data/deployed-rev` — a stamp file written after
     the last successful redeploy.
  3. If different (or `--force`), gracefully stops watcher + dashboard
     scheduled tasks, kills any straggling Python processes that loaded
     old code, restarts the tasks, and updates the stamp.
  4. Optional `--pull`: `git pull` from origin first so a daily cron can
     fully self-deploy without any manual touch.

Scope decisions
---------------
- We DON'T touch MCP server processes spawned by Claude Code. Those have
  no scheduled task to respawn them; killing them risks dropping live
  Claude Code sessions. They naturally recycle via the reaper or the
  next Claude Code restart. (Phase 36-G's embedder-retry handles cold
  startup gracefully.)
- We DON'T touch LM Studio. It's a separate service the user manages.
- We always re-read git rev AFTER pull, so the stamp matches the code
  actually deployed.

Idempotency
-----------
Running `redeploy` twice in a row is a no-op the second time: the
stamp matches the rev, so nothing is killed. `--force` overrides this.
"""
from __future__ import annotations

import contextlib
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from code_rag.logging import get

log = get(__name__)


_STAMP_FILENAME = "deployed-rev"


# Scheduled-task names this redeploy module is allowed to stop+restart.
# We DELIBERATELY exclude the ones that don't run a long-lived process
# (chroma-heal, lms-enforce, eval-gate, eval-drift, health-alert, defrag,
# reap) — those are short-lived cron jobs that pick up new code on their
# next firing without any kick.
_TASKS_TO_RECYCLE = ("code-rag-watch", "code-rag-dashboard")


@dataclass(frozen=True)
class RedeployPlan:
    """Describes what redeploy() *would* do. Returned by plan() so callers
    (including tests) can inspect the decision without side effects."""
    needed: bool
    current_rev: str | None
    deployed_rev: str | None
    reason: str       # human-readable: "first deploy" / "rev changed" / "up to date" / etc.


def _stamp_path(data_dir: Path) -> Path:
    return data_dir / _STAMP_FILENAME


def current_git_rev(repo_root: Path) -> str | None:
    """Return the current `git rev-parse HEAD`, or None on any error
    (not a git repo, git not installed, etc)."""
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if r.returncode != 0:
            return None
        rev = (r.stdout or "").strip()
        return rev or None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


def read_deployed_rev(data_dir: Path) -> str | None:
    """Return the SHA stamped after the last successful redeploy, or None
    if no stamp exists."""
    p = _stamp_path(data_dir)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def write_deployed_rev(data_dir: Path, rev: str) -> bool:
    """Persist the current rev as the deployed marker. Returns True iff
    the write succeeded. Phase 38: atomic write via temp + os.replace so
    a crash mid-write can't leave a truncated SHA that read_deployed_rev
    would later mis-compare against current."""
    p = _stamp_path(data_dir)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.warning("redeploy.stamp_mkdir_fail", path=str(p.parent), err=str(e))
        return False
    import os as _os
    import tempfile as _tempfile
    try:
        with _tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False,
            dir=str(p.parent), prefix=".deployed-rev.",
            suffix=".tmp",
        ) as tf:
            tf.write(rev.strip() + "\n")
            tf.flush()
            tmp_name = tf.name
        _os.replace(tmp_name, p)
        return True
    except OSError as e:
        log.warning("redeploy.stamp_write_fail", path=str(p), err=str(e))
        return False


def plan(repo_root: Path, data_dir: Path, *, force: bool = False) -> RedeployPlan:
    """Decide whether a redeploy is needed. Pure function — no side effects.

    Returns a RedeployPlan with `needed`, the two revs, and a reason.
    """
    current = current_git_rev(repo_root)
    deployed = read_deployed_rev(data_dir)

    if force:
        return RedeployPlan(
            needed=True, current_rev=current, deployed_rev=deployed,
            reason="forced",
        )
    if current is None:
        return RedeployPlan(
            needed=False, current_rev=None, deployed_rev=deployed,
            reason="git rev unavailable (not a repo? git missing?)",
        )
    if deployed is None:
        return RedeployPlan(
            needed=True, current_rev=current, deployed_rev=None,
            reason="first deploy (no stamp file)",
        )
    if current != deployed:
        return RedeployPlan(
            needed=True, current_rev=current, deployed_rev=deployed,
            reason=f"rev changed: {deployed[:8]} -> {current[:8]}",
        )
    return RedeployPlan(
        needed=False, current_rev=current, deployed_rev=deployed,
        reason="up to date",
    )


def _run_powershell(cmd: str, *, timeout_s: float = 30.0) -> tuple[int, str]:
    """Run a single PowerShell one-liner, return (returncode, stdout+stderr)."""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout_s, check=False,
        )
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return -1, f"{type(e).__name__}: {e}"


def stop_tasks(tasks: tuple[str, ...] = _TASKS_TO_RECYCLE) -> dict[str, str]:
    """Stop each scheduled task (best-effort). Returns name -> outcome."""
    results: dict[str, str] = {}
    for name in tasks:
        rc, _out = _run_powershell(
            f"Stop-ScheduledTask -TaskName '{name}' -ErrorAction SilentlyContinue",
        )
        results[name] = "stopped" if rc == 0 else f"err({rc})"
    return results


def start_tasks(tasks: tuple[str, ...] = _TASKS_TO_RECYCLE) -> dict[str, str]:
    """Start each scheduled task. Returns name -> outcome."""
    results: dict[str, str] = {}
    for name in tasks:
        rc, _out = _run_powershell(
            f"Start-ScheduledTask -TaskName '{name}' -ErrorAction SilentlyContinue",
        )
        results[name] = "started" if rc == 0 else f"err({rc})"
    return results


def kill_stragglers() -> int:
    """Kill any code_rag autostart_bootstrap / dashboard / mcp Python procs
    still running. The scheduled tasks restart cleanly on Start-ScheduledTask
    even if a previous instance lingered. Returns count killed."""
    try:
        # Lazy-import so tests can monkey-patch without dragging in win32 stuff
        # at module import time on non-Windows.
        from code_rag.util.proc_hygiene import (
            kill_pid,
            list_code_rag_processes,
        )
    except Exception as e:  # pragma: no cover — defensive
        log.warning("redeploy.proc_hygiene_unavailable", err=str(e))
        return 0

    killed = 0
    for p in list_code_rag_processes():
        # Don't touch MCP servers — they're spawned by Claude Code and we
        # don't have a scheduled task to respawn them.
        if p.kind in ("watcher", "dashboard", "watch_cli"):
            with contextlib.suppress(Exception):
                kill_pid(p.pid)
                killed += 1
    return killed


def git_pull(repo_root: Path) -> tuple[bool, str]:
    """`git -C <repo> pull --ff-only`. Returns (ok, output). FF-only so a
    diverged local branch fails loudly instead of silently merging."""
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_root), "pull", "--ff-only"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=120, check=False,
        )
        out = (r.stdout or "") + (r.stderr or "")
        return r.returncode == 0, out
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return False, f"{type(e).__name__}: {e}"


@dataclass
class RedeployResult:
    plan: RedeployPlan
    pulled: bool = False
    pull_output: str = ""
    tasks_stopped: dict[str, str] | None = None
    procs_killed: int = 0
    tasks_started: dict[str, str] | None = None
    stamped: bool = False
    elapsed_s: float = 0.0


def redeploy(
    repo_root: Path,
    data_dir: Path,
    *,
    pull: bool = False,
    force: bool = False,
    dry_run: bool = False,
    settle_seconds: float = 3.0,
) -> RedeployResult:
    """Top-level orchestrator. See module docstring for scope.

    `settle_seconds` is the gap between Stop-ScheduledTask + kill and
    Start-ScheduledTask. 3s is enough on this machine for the scheduled
    task to clear its previous-instance state.
    """
    t0 = time.monotonic()

    pre_pull_rev = current_git_rev(repo_root)
    pulled = False
    pull_output = ""
    if pull:
        ok, pull_output = git_pull(repo_root)
        pulled = ok
        if not ok:
            log.warning("redeploy.pull_failed", out=pull_output[:500])
        # Re-read the rev after pull so the stamp reflects what's actually deployed.

    p = plan(repo_root, data_dir, force=force)
    result = RedeployResult(plan=p, pulled=pulled, pull_output=pull_output)

    # Phase 39: if the user has explicitly hit Stop All, do NOT kick the
    # watcher/dashboard tasks back to life — that would defeat the
    # intentional stop. Still perform the rev compare so the next regular
    # redeploy (post-Start-All or post-reboot) sees the up-to-date stamp
    # and doesn't re-fire on stale state.
    from code_rag.util.stop_marker import is_intentionally_stopped
    if is_intentionally_stopped(data_dir) and not force:
        log.info("redeploy.skip_intentionally_stopped",
                 reason="data/.stopped present — user said stay stopped")
        # Stamp the rev anyway so future redeploys don't keep claiming
        # "rev changed" against an old marker — only stamp if rev moved.
        if p.needed and p.current_rev:
            result.stamped = write_deployed_rev(data_dir, p.current_rev)
        result.elapsed_s = time.monotonic() - t0
        return result

    if not p.needed or dry_run:
        result.elapsed_s = time.monotonic() - t0
        if dry_run:
            log.info("redeploy.dry_run", reason=p.reason,
                     current=p.current_rev, deployed=p.deployed_rev)
        else:
            log.info("redeploy.skip", reason=p.reason)
        return result

    # 1) Stop scheduled tasks.
    result.tasks_stopped = stop_tasks()
    # 2) Kill straggler watcher/dashboard procs that didn't exit cleanly.
    result.procs_killed = kill_stragglers()
    # 3) Settle.
    if settle_seconds > 0:
        time.sleep(settle_seconds)
    # 4) Start scheduled tasks again.
    result.tasks_started = start_tasks()
    # 5) Stamp the new rev. Only mark as stamped if the write actually
    # succeeded — a disk-full or permission failure used to be silently
    # swallowed, claiming a deploy succeeded when the stamp was missing,
    # so the next cron iteration re-killed the watcher.
    if p.current_rev:
        result.stamped = write_deployed_rev(data_dir, p.current_rev)
    log.info(
        "redeploy.done",
        from_=p.deployed_rev, to=p.current_rev,
        pre_pull=pre_pull_rev, pulled=pulled,
        procs_killed=result.procs_killed,
        elapsed_s=round(time.monotonic() - t0, 2),
    )
    result.elapsed_s = time.monotonic() - t0
    return result


def find_repo_root() -> Path:
    """Find the repo root that contains this module. The redeploy CLI uses
    this when not invoked from inside the tree."""
    here = Path(__file__).resolve()
    for ancestor in (here.parent, *here.parents):
        if (ancestor / ".git").exists():
            return ancestor
        if (ancestor / "pyproject.toml").exists() and (ancestor / "src" / "code_rag").exists():
            return ancestor
    # Fall back to the working tree root assumed by the install layout.
    return here.parents[3]   # src/code_rag/util/redeploy.py -> repo


def main_for_test_entry() -> int:  # pragma: no cover — exercised by CLI
    """Allow `python -m code_rag.util.redeploy --pull` for ops debugging
    when click isn't on PATH for some reason."""
    pull = "--pull" in sys.argv
    force = "--force" in sys.argv
    dry = "--dry-run" in sys.argv
    repo = find_repo_root()
    from code_rag.config import load_settings
    s = load_settings()
    r = redeploy(repo, s.paths.data_dir, pull=pull, force=force, dry_run=dry)
    print(f"reason={r.plan.reason} stamped={r.stamped} killed={r.procs_killed} "
          f"elapsed_s={r.elapsed_s:.2f}")
    return 0 if not r.plan.needed or r.stamped else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main_for_test_entry())

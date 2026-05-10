"""Phase 60-I: idle-watcher loop that auto-stops vLLM when code-rag is idle.

Why this exists
---------------
Code-rag's vLLM stack holds ~13 GB VRAM steady-state, which prevents
YouTubeBot's ComfyUI FLUX (~14 GB VRAM) from coexisting on the 22.5 GB
4090. Pattern B (manual Stop button) only handles "I'm gaming" — but
YouTubeBot's `run_pipeline.py --watch` daemon fires renders unattended.
This loop is the autonomy hook: when code-rag has been idle for >N
minutes, write the auto-stop marker and pkill the WSL vLLM servers.

Idle signal
-----------
We use `data/chroma/chroma.sqlite3` mtime as the proxy. Chroma is
written by:

  * The indexer (one-shot bulk reindex via `code_rag index`).
  * The watcher (per-file events from filesystem changes).

Reads (MCP search queries that don't add new chunks) don't update
chroma. That's deliberate — interactive queries shouldn't be the only
thing keeping vLLM warm. If you're truly using code-rag (Claude is
making search calls), the indexer will likely be running too because
you're touching files; if you've gone fully idle for 2 h, pay the
30 s cold-start.

Wake conditions (NOT in this module — wired in watcher.live._apply)
-------------------------------------------------------------------
A file event arriving at the watcher will:
  1. Notice the auto-stop marker.
  2. Clear it.
  3. Spawn `scripts/resume_code_rag.ps1`.
  4. Block up to 90 s for vLLM /v1/models to respond before processing.

Stop-marker priority
--------------------
The auto-stop marker NEVER triggers when the user-stop marker
(`data/.stopped`) is present. User intent always wins.
"""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from code_rag.logging import get
from code_rag.util.auto_stop_marker import is_auto_stopped, mark_auto_stopped
from code_rag.util.stop_marker import is_intentionally_stopped

if TYPE_CHECKING:
    from code_rag.config import Settings

log = get(__name__)


# Default: 2 hours of no chroma writes before auto-stopping. Long enough
# that a user stepping away from Claude Code for 30-60 min doesn't
# trigger; short enough that overnight YouTubeBot rendering reclaims
# the GPU. Override via `auto_stop.idle_seconds` in config.toml.
_DEFAULT_IDLE_SECONDS = 7200


def _chroma_mtime(settings: "Settings") -> float | None:
    """Return chroma.sqlite3's last-modified time in unix seconds, or
    None if the file doesn't exist (no index built yet)."""
    p = settings.paths.data_dir / "chroma" / "chroma.sqlite3"
    try:
        return p.stat().st_mtime
    except OSError:
        return None


def is_idle(settings: "Settings", *, idle_seconds: float) -> bool:
    """Return True iff chroma hasn't been written in `idle_seconds`."""
    mtime = _chroma_mtime(settings)
    if mtime is None:
        # No index file → we're not "idle", we're "uninitialized". Don't
        # auto-stop; let the user finish setting up.
        return False
    import time
    return (time.time() - mtime) > idle_seconds


def _kill_vllm_in_wsl_blocking() -> bool:
    """Synchronous helper: pkill all vLLM-family processes in WSL.
    Mirrors `dashboard.operations._kill_vllm_in_wsl` but inlined here
    so the watcher loop doesn't import the dashboard module (would pull
    in starlette + httpx for nothing). Returns True iff the pkill ran.
    """
    cmd = [
        "wsl.exe", "-d", "Ubuntu", "-e", "bash", "-c",
        "pkill -TERM -f 'vllm serve' 2>/dev/null; "
        "pkill -TERM -f 'EngineCore' 2>/dev/null; "
        "pkill -TERM -f 'multiprocessing.resource_tracker' 2>/dev/null; "
        "pkill -TERM -f 'multiprocessing.spawn' 2>/dev/null; "
        "sleep 1; "
        "pkill -KILL -f 'vllm serve' 2>/dev/null; "
        "pkill -KILL -f 'EngineCore' 2>/dev/null; "
        "pkill -KILL -f 'multiprocessing.resource_tracker' 2>/dev/null; "
        "pkill -KILL -f 'multiprocessing.spawn' 2>/dev/null; "
        "echo done",
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15.0,
            creationflags=0x08000000,  # CREATE_NO_WINDOW
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


async def auto_stop_loop(
    settings: "Settings",
    *,
    interval_s: float = 60.0,
    idle_seconds: float = _DEFAULT_IDLE_SECONDS,
) -> None:
    """Long-lived: every `interval_s`, check idle and auto-stop if
    conditions met. Designed to be `asyncio.create_task`-ed alongside
    the file watcher in `autostart_bootstrap`. Never raises — defensive
    swallow at the loop boundary so a glitch can't kill the watcher.
    """
    log.info("auto_stop_loop.started",
             interval_s=interval_s, idle_seconds=idle_seconds)
    while True:
        try:
            data_dir = settings.paths.data_dir

            # Skip if user has already pressed Stop OR we've already
            # auto-stopped. Idempotent in both directions.
            if is_intentionally_stopped(data_dir):
                pass  # user-stop is sticky; do nothing.
            elif is_auto_stopped(data_dir):
                pass  # already auto-stopped; wait for wake event.
            elif is_idle(settings, idle_seconds=idle_seconds):
                log.info("auto_stop_loop.triggering",
                         idle_seconds=idle_seconds,
                         reason=f"chroma idle >{idle_seconds:.0f}s")
                # Order matters: write marker FIRST so a concurrent watcher
                # event sees the marker and triggers wake (rather than
                # racing to embed against a vLLM that's about to die).
                mark_auto_stopped(data_dir, reason=f"watcher.idle.{idle_seconds:.0f}s")
                killed = await asyncio.to_thread(_kill_vllm_in_wsl_blocking)
                log.info("auto_stop_loop.stopped", killed=killed)
        except Exception as e:  # pragma: no cover — defensive
            log.warning("auto_stop_loop.cycle_error",
                        err=f"{type(e).__name__}: {e}")
        await asyncio.sleep(interval_s)


# ---- demand-signal wake-up helper (called from watcher.live) ----


def wake_if_auto_stopped(data_dir: Path) -> bool:
    """Called by the watcher when a file event arrives. If the auto-stop
    marker is present, clear it and spawn `scripts/resume_code_rag.ps1`
    to bring vLLM back. Returns True iff a wake was triggered (caller
    should then wait for vLLM to be reachable before processing the event).
    """
    from code_rag.util.auto_stop_marker import clear_auto_stopped

    if is_intentionally_stopped(data_dir):
        # User-stop wins; don't wake.
        return False
    if not is_auto_stopped(data_dir):
        return False

    clear_auto_stopped(data_dir, reason="watcher.demand_signal")

    # Find the resume script relative to this module.
    ps_script = (
        Path(__file__).parent.parent.parent.parent
        / "scripts" / "resume_code_rag.ps1"
    )
    if not ps_script.exists():
        log.warning("auto_stop_wake.script_missing", path=str(ps_script))
        return False

    try:
        subprocess.Popen(
            ["powershell.exe", "-NoProfile", "-ExecutionPolicy",
             "Bypass", "-File", str(ps_script)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, close_fds=True,
            creationflags=0x08000000,  # CREATE_NO_WINDOW
        )
        log.info("auto_stop_wake.spawned", script=str(ps_script))
        return True
    except OSError as e:
        log.warning("auto_stop_wake.spawn_fail", err=str(e))
        return False

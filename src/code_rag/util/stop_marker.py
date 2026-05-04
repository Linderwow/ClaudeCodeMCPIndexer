"""Phase 39: intentional-stop marker.

Why this exists
---------------
Stop All in the dashboard kills the watcher + dashboard tasks. But the
Phase 36-C reaper (running every 10 min) was designed to self-heal from
crashes — it sees `is_watcher_alive() == False` and respawns the watcher
via `Start-ScheduledTask`. So Stop All used to come back within 10 min,
which the user never wanted: they pressed Stop, they meant Stop.

Reapers and the daily redeploy task now check this module's
`is_intentionally_stopped()` BEFORE respawning anything. Crash-vs-intent
is distinguished by the presence of `data/.stopped`:

    crash:    no .stopped file → reaper respawns (normal self-heal)
    intent:   .stopped file present → reaper leaves it alone

Lifetime semantics
------------------
The marker persists across machine reboots... almost. The autostart
bootstrap deletes the marker on logon, so a PC restart implicitly clears
the user's stop intent (matches their stated expectation: "stay stopped
until I hit Start All OR restart the PC"). To resume mid-session without
restarting, hit Start All.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

from code_rag.logging import get

log = get(__name__)


_MARKER_FILENAME = ".stopped"


def marker_path(data_dir: Path) -> Path:
    """Canonical location of the intent marker."""
    return data_dir / _MARKER_FILENAME


def is_intentionally_stopped(data_dir: Path) -> bool:
    """True iff the user has explicitly hit Stop All and not yet hit Start
    All (or rebooted)."""
    return marker_path(data_dir).exists()


def _system_boot_time_unix() -> float | None:
    """Return wall-clock unix timestamp of the last system boot, or None
    if it can't be determined on this platform.

    Windows: uses GetTickCount64() from kernel32 — returns milliseconds
    since boot. Wall-clock boot time = now - tickcount/1000. Zero deps,
    no subprocess, microsecond cost.

    Linux: reads `btime` from /proc/stat.

    Other platforms: returns None — caller falls back to "stay stopped"
    (safe default vs. resuming against user intent).
    """
    import sys
    import time
    if sys.platform == "win32":
        try:
            import ctypes
            ms = ctypes.windll.kernel32.GetTickCount64()  # type: ignore[attr-defined]
            return time.time() - (ms / 1000.0)
        except (OSError, AttributeError):
            return None
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/stat") as f:
                for line in f:
                    if line.startswith("btime "):
                        return float(line.split()[1])
        except OSError:
            return None
    return None


def marker_predates_last_boot(data_dir: Path) -> bool | None:
    """Return True iff the marker exists AND was created BEFORE the last
    system boot — i.e. the user pressed Stop All in some PRIOR session and
    the PC has since been rebooted, so it's safe to clear and resume.

    Returns False iff the marker exists and was created in the current
    boot session (Stop All happened after the last reboot — user intent
    is "stay stopped").

    Returns None iff the marker doesn't exist (caller can short-circuit).

    Phase 42: this lets `autostart_bootstrap` distinguish "AtLogon fire
    after reboot" (clear-and-proceed) from "Task Scheduler auto-restart
    after Stop All terminated the process" (bail-and-respect-intent).
    The previous implementation cleared the marker on every fire, which
    broke the Stop All contract any time TaskScheduler's RestartCount
    policy kicked in.
    """
    p = marker_path(data_dir)
    if not p.exists():
        return None
    try:
        marker_mtime = p.stat().st_mtime
    except OSError:
        # If we can't stat the marker, fall back to "stay stopped" — that's
        # the safer default vs. resuming when the user said don't.
        return False
    boot_time = _system_boot_time_unix()
    if boot_time is None:
        # Can't determine boot time → safer default is "stay stopped".
        return False
    return marker_mtime < boot_time


def mark_intentionally_stopped(data_dir: Path, *, reason: str = "dashboard.stop_all") -> bool:
    """Create the marker. Returns True iff the file is on disk after the
    call (covers both the new-write and already-exists cases)."""
    p = marker_path(data_dir)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.warning("stop_marker.mkdir_fail", path=str(p.parent), err=str(e))
        return False
    try:
        # Plain text body — useful for debugging if a user finds the file
        # and wonders what put it there.
        from datetime import UTC, datetime
        body = f"intentionally stopped at {datetime.now(UTC).isoformat()} via {reason}\n"
        p.write_text(body, encoding="utf-8")
        return True
    except OSError as e:
        log.warning("stop_marker.write_fail", path=str(p), err=str(e))
        return p.exists()


def clear_intentionally_stopped(data_dir: Path, *, reason: str = "start_all") -> bool:
    """Delete the marker. Returns True iff the file is gone after the call
    (covers both the just-deleted and never-existed cases)."""
    p = marker_path(data_dir)
    if not p.exists():
        return True
    with contextlib.suppress(OSError):
        p.unlink()
    if p.exists():
        log.warning("stop_marker.delete_fail", path=str(p), reason=reason)
        return False
    log.info("stop_marker.cleared", reason=reason)
    return True

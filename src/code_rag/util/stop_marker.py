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

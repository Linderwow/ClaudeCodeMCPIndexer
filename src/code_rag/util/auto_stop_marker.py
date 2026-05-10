"""Phase 60-I: auto-stop marker (transient, distinct from user-stop).

Why this exists separately from `stop_marker.py`
------------------------------------------------
The Phase 39 stop marker (`data/.stopped`) is STICKY across logon
because the user explicitly clicked Stop. The auto-stop marker
(`data/.auto_stopped`) is TRANSIENT — set by the watcher when code-rag
has been idle for >30 min so YouTubeBot's ComfyUI render can claim
the GPU. It clears automatically on the next demand signal (a watcher
file event, or a dashboard Start click).

Coexistence rules
-----------------
The two markers serve different intents and don't interact:

  * `.stopped`        present  → user said stay stopped. Don't auto-resume.
  * `.auto_stopped`   present  → idle savings. Auto-resume on demand.
  * BOTH present                → user wins. Treat as user-stop.
  * NEITHER present              → run normally.

The watcher only writes `.auto_stopped` when `.stopped` is NOT present.
The dashboard's Stop button writes `.stopped` (regardless of auto state).
The dashboard's Start button clears BOTH.

Demand-signal wakeup
--------------------
When a watcher file event arrives and `.auto_stopped` is present (but
`.stopped` is not), the watcher:

  1. Clears `.auto_stopped` (it'll be re-set after another idle stretch).
  2. Spawns `scripts/resume_code_rag.ps1` to bring vLLM back.
  3. Waits up to 90s for vLLM /v1/models to respond before processing
     the event (so the embedding call doesn't fail).

This is "lightweight on-demand" without the full cold-start coordination
of Pattern C — vLLM cold start is ~30s, which is acceptable for a file-
event-driven workflow but would be too slow for interactive MCP queries.
"""
from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from pathlib import Path

from code_rag.logging import get

log = get(__name__)


_MARKER_FILENAME = ".auto_stopped"


def marker_path(data_dir: Path) -> Path:
    return data_dir / _MARKER_FILENAME


def is_auto_stopped(data_dir: Path) -> bool:
    """True iff the watcher has auto-stopped vLLM due to idle."""
    return marker_path(data_dir).exists()


def mark_auto_stopped(data_dir: Path, *, reason: str = "watcher.idle") -> bool:
    """Create the auto-stop marker. Best-effort; returns True iff the
    file exists after the call.
    """
    p = marker_path(data_dir)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.warning("auto_stop_marker.mkdir_fail", path=str(p.parent), err=str(e))
        return False
    try:
        body = (
            f"auto-stopped at {datetime.now(UTC).isoformat()} via {reason}\n"
            "(transient — clears on next demand signal)\n"
        )
        p.write_text(body, encoding="utf-8")
        return True
    except OSError as e:
        log.warning("auto_stop_marker.write_fail", path=str(p), err=str(e))
        return p.exists()


def clear_auto_stopped(data_dir: Path, *, reason: str = "demand.signal") -> bool:
    """Delete the auto-stop marker. Returns True iff the file is gone
    after the call (covers the never-existed and just-deleted cases).
    """
    p = marker_path(data_dir)
    if not p.exists():
        return True
    with contextlib.suppress(OSError):
        p.unlink()
    if p.exists():
        log.warning("auto_stop_marker.delete_fail", path=str(p), reason=reason)
        return False
    log.info("auto_stop_marker.cleared", reason=reason)
    return True

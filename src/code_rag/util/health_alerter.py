"""Phase 37-J: dashboard degraded-state alerter.

Polls the dashboard's `/api/health` (which probes LM Studio, watcher
heartbeat, Chroma, VRAM, LMS duplicates) and writes the latest status
to `data/health-state.json`. On every transition (`ok` -> `degraded` or
`degraded` -> `critical`), appends a row to `data/alerts.jsonl` AND
optionally fires a Windows toast notification so the user sees the
degradation without manually opening the dashboard.

Why an external poller, not push-from-dashboard
-----------------------------------------------
The dashboard /api/health endpoint is on-demand: it only computes
state when a client hits it. To get continuous visibility we need a
scheduled task that polls. That keeps the dashboard's request handler
lightweight while giving us a tick-by-tick alert log.

Design choices
--------------
- **Stateless** between runs. State lives in the JSON file on disk, so
  the poller can be killed/restarted at any moment without losing
  alerting history.
- **Best-effort toasts.** PowerShell `New-BurntToastNotification` is
  the cleanest path on Windows, but it's a third-party module. Fall
  back to `msg.exe` (built-in but ugly modal) only when the user has
  explicitly asked for noisy alerts.
- **Idempotent.** A `degraded` state that persists for hours triggers
  exactly ONE alert (on first detection) — not one per poll cycle.
"""
from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from code_rag.logging import get

log = get(__name__)


HEALTH_STATE_FILENAME = "health-state.json"
ALERTS_LOG_FILENAME = "alerts.jsonl"


@dataclass
class AlerterPaths:
    """Resolved file locations the alerter writes to."""
    state: Path
    alerts: Path

    @classmethod
    def for_data_dir(cls, data_dir: Path) -> AlerterPaths:
        return cls(
            state=data_dir / HEALTH_STATE_FILENAME,
            alerts=data_dir / ALERTS_LOG_FILENAME,
        )


@dataclass
class HealthSnapshot:
    """A single /api/health probe result + transition info."""
    ts: str
    overall: str        # "ok" | "degraded" | "critical" | "unreachable"
    checks: dict[str, Any] = field(default_factory=dict)
    previous_overall: str | None = None
    transition: bool = False    # True iff overall != previous_overall


def _read_previous(state_path: Path) -> str | None:
    if not state_path.exists():
        return None
    try:
        data = json.loads(state_path.read_text("utf-8"))
        prev = data.get("overall")
        return str(prev) if isinstance(prev, str) else None
    except (json.JSONDecodeError, OSError):
        return None


def _write_state(state_path: Path, snap: HealthSnapshot) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": snap.ts,
        "overall": snap.overall,
        "previous_overall": snap.previous_overall,
        "transition": snap.transition,
        "checks": snap.checks,
    }
    try:
        state_path.write_text(
            json.dumps(payload, indent=2), encoding="utf-8",
        )
    except OSError as e:
        log.warning("alerter.state_write_fail",
                    path=str(state_path), err=str(e))


def _append_alert(alerts_path: Path, snap: HealthSnapshot) -> None:
    """Append one JSON line to alerts.jsonl on a transition."""
    alerts_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": snap.ts,
        "from": snap.previous_overall,
        "to": snap.overall,
        "checks": snap.checks,
    }
    try:
        with alerts_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")
    except OSError as e:
        log.warning("alerter.alert_write_fail",
                    path=str(alerts_path), err=str(e))


def _summarize_failed_checks(checks: dict[str, Any]) -> str:
    """Return a short human-readable list of failed-check names for toasts.
    Toasts have a hard ~250 char limit so we keep this terse."""
    failed: list[str] = []
    for name, value in checks.items():
        if isinstance(value, dict) and value.get("ok") is False:
            detail = value.get("detail", "")
            failed.append(f"{name}: {detail}" if detail else name)
    if not failed:
        return "(no specific check details)"
    joined = "; ".join(failed)
    return joined if len(joined) < 200 else (joined[:197] + "...")


def maybe_show_toast(snap: HealthSnapshot) -> bool:
    """Best-effort Windows toast on degradation. Returns True if a toast
    was actually shown (or attempted). No-ops on non-Windows.
    BurntToast is the preferred path; we don't fall back to msg.exe
    because a system-modal popup on background degradation is hostile."""
    import platform
    import subprocess
    if platform.system() != "Windows":
        return False
    if snap.overall not in ("degraded", "critical", "unreachable"):
        return False
    title = f"code-rag {snap.overall}"
    body = _summarize_failed_checks(snap.checks)
    # Use the same `New-BurntToastNotification` pattern other Windows tools
    # use. If BurntToast isn't installed, the call fails silently — we log
    # and move on.
    ps_cmd = (
        "if (Get-Module -ListAvailable -Name BurntToast) { "
        "Import-Module BurntToast -ErrorAction Stop; "
        f"New-BurntToastNotification -Text \"{title}\", \"{body}\" "
        "}"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
            capture_output=True, timeout=10, check=False,
        )
        return True
    except Exception as e:  # pragma: no cover — defensive
        log.debug("alerter.toast_fail", err=f"{type(e).__name__}: {e}")
        return False


def probe_health(
    base_url: str = "http://127.0.0.1:7321",
    *,
    timeout_s: float = 5.0,
) -> tuple[str, dict[str, Any]]:
    """Hit the dashboard's /api/health. Returns (overall, checks).

    On HTTP failure (dashboard down, port closed) returns
    `("unreachable", {})` — that's itself a meaningful state worth
    alerting on, not an error condition that should crash the poller.
    """
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/api/health", timeout=timeout_s)
        if r.status_code != 200:
            return "unreachable", {"http_status": r.status_code}
        body = r.json()
    except (httpx.HTTPError, OSError, ValueError) as e:
        return "unreachable", {"err": f"{type(e).__name__}: {e}"}
    overall = body.get("overall", "unreachable")
    checks = body.get("checks", {}) if isinstance(body.get("checks"), dict) else {}
    return str(overall), checks


def check_once(
    paths: AlerterPaths,
    *,
    base_url: str = "http://127.0.0.1:7321",
    timeout_s: float = 5.0,
    show_toast: bool = True,
) -> HealthSnapshot:
    """Run one probe + state-transition check. Returns the snapshot so
    callers can log it / inspect it / write tests against it."""
    overall, checks = probe_health(base_url, timeout_s=timeout_s)
    previous = _read_previous(paths.state)
    snap = HealthSnapshot(
        ts=datetime.now(UTC).isoformat(),
        overall=overall,
        checks=checks,
        previous_overall=previous,
        transition=(previous is not None and overall != previous),
    )
    _write_state(paths.state, snap)
    if snap.transition:
        _append_alert(paths.alerts, snap)
        if show_toast:
            with contextlib.suppress(Exception):
                maybe_show_toast(snap)
        log.info("alerter.transition",
                 from_=previous, to=overall,
                 failed=_summarize_failed_checks(checks))
    return snap


def poll_forever(
    paths: AlerterPaths,
    *,
    base_url: str = "http://127.0.0.1:7321",
    interval_s: float = 60.0,
    show_toast: bool = True,
) -> None:  # pragma: no cover — long-running loop
    """Run check_once every `interval_s` until killed. Designed for the
    Task Scheduler installer; tests cover `check_once` directly."""
    log.info("alerter.started", base_url=base_url, interval_s=interval_s)
    while True:
        try:
            check_once(paths, base_url=base_url, show_toast=show_toast)
        except Exception as e:
            # Defensive: an alerter crash can't be allowed to silence
            # subsequent alerts. Log and continue.
            log.warning("alerter.cycle_error",
                        err=f"{type(e).__name__}: {e}")
        time.sleep(interval_s)

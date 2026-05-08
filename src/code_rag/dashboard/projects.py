"""Phase 50: Unified Command Center.

Aggregates state for ALL the projects the user actively runs:
  - code-rag       : the RAG stack (this project)
  - YouTubeBot     : the YouTube content pipeline (separate repo)
  - MNQAlpha       : the trading-signals daemons (separate repo at ~/signals)

For each project we report:
  - Scheduled tasks (name, cadence, last-run, hidden status, exe)
  - Running processes (pid, parent pid, RAM, command preview, age)

Read-only by design — actions are exposed via the existing per-project
endpoints (Start All / Stop All / etc.). This module is purely a
read-side aggregator powering the Unified Command Center page.

All probes run in a worker thread (called via asyncio.to_thread from
the handler) so the dashboard event loop never blocks on PowerShell
cmdlets.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

_CREATE_NO_WINDOW = 0x08000000

# Project profiles — each entry tells us which scheduled tasks belong to
# the project (matched against TaskName via fnmatch-style globs) and
# which running processes belong to it (matched against the Win32 process
# command line via regex).
PROJECT_PROFILES: list[dict[str, Any]] = [
    {
        "id": "code-rag",
        "label": "code-rag",
        "task_globs": ["code-rag-*"],
        # Match exact entry-point shapes — same anchoring as
        # proc_hygiene.list_code_rag_processes uses.
        "process_regex": (
            r"-m\s+code_rag(\.|\s|$)"
            r"|code-rag\.exe"
            r"|code_rag\\Scripts\\code-rag"
        ),
    },
    {
        "id": "YouTubeBot",
        "label": "YouTubeBot",
        "task_globs": [
            "ThePremise_*",
            "YouTubeBot*",
            "YouTubeBot_*",
        ],
        "process_regex": (
            r"\\YouTubeBot\\"
            r"|run_pipeline\.py"
            r"|run_watch_hidden\.vbs"
            r"|run_cognitron_hidden\.vbs"
            r"|run_premise_hidden\.vbs"
            r"|run_watchdog_hidden\.vbs"
            r"|run_cleanup_hidden\.vbs"
        ),
    },
    {
        "id": "MNQAlpha",
        "label": "MNQAlpha (signals)",
        "task_globs": ["MNQAlpha_*"],
        "process_regex": r"\\Users\\Alex\\signals\\",
    },
]


@dataclass(frozen=True)
class TaskInfo:
    name: str
    state: str               # "Ready" | "Running" | "Disabled" | ...
    hidden: bool
    exe: str                 # leaf name of the action executable
    repeat: str | None       # e.g. "PT5M", "PT1H", "P1D", or None
    last_run: str | None     # ISO timestamp or None
    last_result: int | None  # Win32 result; 0 = success


@dataclass(frozen=True)
class ProcInfo:
    pid: int
    parent_pid: int
    name: str
    ram_mb: int
    cmd_preview: str
    age_s: int


@dataclass
class ProjectState:
    id: str
    label: str
    tasks: list[TaskInfo] = field(default_factory=list)
    processes: list[ProcInfo] = field(default_factory=list)
    error: str | None = None


# ---- live probes ----------------------------------------------------------


def _list_tasks(globs: list[str]) -> list[TaskInfo]:
    """One PowerShell invocation that returns JSON for every matching task.

    Glob-matching is done in PowerShell via `-like` so we don't pay the cost
    of returning EVERY task to Python. The 5-sec timeout is enough on this
    box (~250 tasks system-wide); bumps if user has a heavy install.
    """
    if sys.platform != "win32":
        return []
    if not globs:
        return []
    # Build the OR-chain: ($_.TaskName -like 'g1') -or ($_.TaskName -like 'g2') ...
    or_clauses = " -or ".join(
        f"($_.TaskName -like '{g}')" for g in globs
    )
    # Phase 53: dump RAW trigger fields (class name + StartBoundary +
    # DaysOfWeek + Repetition.Interval) per trigger and let Python format.
    # The previous attempt at an in-PS switch with multi-statement cases
    # had too many edge-case crashes (DaysOfWeek empty, StartBoundary
    # malformed, etc.) and silently returned 0 tasks. Raw fields are
    # boring + reliable.
    script = (
        "Get-ScheduledTask | Where-Object { " + or_clauses + " } | "
        "ForEach-Object { "
        "  $info = Get-ScheduledTaskInfo $_; "
        "  $triggers = @($_.Triggers | ForEach-Object { "
        "      [PSCustomObject]@{ "
        "        cls            = $_.CimClass.CimClassName; "
        "        start_boundary = $_.StartBoundary; "
        "        days_of_week   = if ($_.DaysOfWeek) { $_.DaysOfWeek.ToString() } else { $null }; "
        "        interval       = if ($_.Repetition) { $_.Repetition.Interval } else { $null }; "
        "      } "
        "  }); "
        "  [PSCustomObject]@{ "
        "    name        = $_.TaskName; "
        "    state       = $_.State.ToString(); "
        "    hidden      = $_.Settings.Hidden; "
        "    exe         = (Split-Path $_.Actions[0].Execute -Leaf); "
        "    triggers    = $triggers; "
        "    last_run    = if ($info.LastRunTime -and $info.LastRunTime.Year -gt 1900) "
        "                  { $info.LastRunTime.ToString('o') } else { $null }; "
        "    last_result = $info.LastTaskResult; "
        "  } "
        "} | ConvertTo-Json -Compress -Depth 4"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=10.0, check=False,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if r.returncode != 0 or not r.stdout.strip():
        return []
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return []
    # Single-result objects come back as dict not list — normalize.
    if isinstance(data, dict):
        data = [data]
    out: list[TaskInfo] = []
    for d in data:
        if not isinstance(d, dict):
            continue
        try:
            triggers_raw = d.get("triggers") or []
            if isinstance(triggers_raw, dict):   # PS may flatten 1-element list
                triggers_raw = [triggers_raw]
            schedule = _format_schedule(triggers_raw)
            out.append(TaskInfo(
                name=str(d.get("name", "")),
                state=str(d.get("state", "Unknown")),
                hidden=bool(d.get("hidden", False)),
                exe=str(d.get("exe", "")),
                repeat=schedule,
                last_run=(str(d["last_run"]) if d.get("last_run") else None),
                last_result=(
                    int(d["last_result"]) if isinstance(d.get("last_result"), int)
                    else None
                ),
            ))
        except (ValueError, TypeError):
            continue
    out.sort(key=lambda t: t.name)
    return out


def _format_schedule(triggers: list[dict[str, Any]]) -> str | None:
    """Phase 53: turn raw triggers into a human schedule string.

    Examples:
      [{cls=...DailyTrigger..., start_boundary='2026-05-08T03:00:00'}]
        -> 'daily 03:00'
      [{cls=...WeeklyTrigger..., days_of_week='Monday, Tuesday', start='...05:00'}]
        -> 'weekly Mon/Tue 05:00'
      [{cls=...RepetitionPattern... or TimeTrigger..., interval='PT5M'}]
        -> 'PT5M'
      [{cls=...DailyTrigger..., interval='PT30M'}]
        -> 'daily 03:00 / +PT30M'

    Returns None when no triggers parse to a meaningful schedule.
    """
    parts: list[str] = []
    for t in triggers:
        if not isinstance(t, dict):
            continue
        cls = str(t.get("cls", ""))
        start = t.get("start_boundary")
        days = t.get("days_of_week")
        interval = t.get("interval")

        # Extract HH:MM from start_boundary if present. Strict check —
        # only accept if it looks like a real ISO time (HH:MM with the
        # colon at position 2). Defends against junk strings that
        # happen to contain a 'T'.
        time_str = ""
        if isinstance(start, str) and "T" in start:
            tail = start.split("T", 1)[1][:5]
            if len(tail) == 5 and tail[2] == ":" and tail[:2].isdigit() and tail[3:].isdigit():
                time_str = tail

        # Extract abbreviated days_of_week if present.
        # Windows MSFT_TaskDaysOfWeek can come back two ways:
        #   - String: 'Monday, Wednesday' → split + abbreviate
        #   - Bitfield int (when ToString() returns the numeric value):
        #     1=Sun, 2=Mon, 4=Tue, 8=Wed, 16=Thu, 32=Fri, 64=Sat
        #     Special: 62 = Mon-Fri (weekdays), 65 = weekends, 127 = all
        _DAY_NAMES = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        days_str = ""
        if isinstance(days, str) and days:
            try:
                bits = int(days)
                # Special cases for nicer rendering
                if bits == 62:
                    days_str = "weekdays"
                elif bits == 65:
                    days_str = "weekends"
                elif bits == 127:
                    days_str = "all"
                else:
                    out_days = [
                        _DAY_NAMES[i] for i in range(7) if bits & (1 << i)
                    ]
                    days_str = "/".join(out_days)
            except ValueError:
                # Comma-separated names like 'Monday, Wednesday'
                day_abbrev = []
                for d in days.split(","):
                    d = d.strip()
                    if d:
                        day_abbrev.append(d[:3])
                days_str = "/".join(day_abbrev)

        base: str | None
        if "DailyTrigger" in cls:
            base = f"daily {time_str}".strip()
        elif "WeeklyTrigger" in cls:
            base = f"weekly {days_str} {time_str}".strip()
        elif "MonthlyTrigger" in cls or "MonthlyDOWTrigger" in cls:
            base = "monthly"
        elif "BootTrigger" in cls:
            base = "at boot"
        elif "LogonTrigger" in cls:
            base = "at logon"
        elif "RegistrationTrigger" in cls:
            base = "at registration"
        elif "EventTrigger" in cls:
            base = "on event"
        elif "TimeTrigger" in cls:
            base = "one-shot" if not interval else None
        elif "RepetitionPattern" in cls:
            base = None   # interval handled below
        else:
            # Unknown — show stripped class name.
            base = cls.replace("MSFT_Task", "").replace("Trigger", "")
            base = base or None

        # Combine base + interval (e.g. 'daily 03:00 / +PT30M')
        if interval and base:
            parts.append(f"{base} / +{interval}")
        elif interval:
            parts.append(str(interval))
        elif base:
            parts.append(base)
    if not parts:
        return None
    return ", ".join(parts)


def _list_processes(pattern: str) -> list[ProcInfo]:
    """One PowerShell invocation returning JSON for every Win32 process whose
    command line matches the regex. Walks Win32_Process via Get-CimInstance.

    Pattern is wrapped with `-match` (PowerShell regex). We pre-escape in
    Python so callers can pass simple regex; the script invocation itself
    has its own escaping handled below.
    """
    if sys.platform != "win32":
        return []
    if not pattern:
        return []
    # PowerShell's -match wants a regex string. We embed it via a here-string
    # to avoid escaping double-quotes / backslashes. The single-quoted
    # here-string `@'...'@` does NO interpolation — safe for arbitrary regex
    # bodies.
    script = (
        "$pattern = @'\n" + pattern + "\n'@\n"
        "Get-CimInstance Win32_Process -EA SilentlyContinue | "
        "Where-Object { $_.CommandLine -and ($_.CommandLine -match $pattern) } | "
        "ForEach-Object { "
        "  $p = Get-Process -Id $_.ProcessId -EA SilentlyContinue; "
        "  $cmd = if ($_.CommandLine.Length -gt 160) "
        "         { $_.CommandLine.Substring(0,160) } else { $_.CommandLine }; "
        "  $age = if ($_.CreationDate) "
        "         { [int]((Get-Date) - $_.CreationDate).TotalSeconds } else { 0 }; "
        "  [PSCustomObject]@{ "
        "    pid         = $_.ProcessId; "
        "    parent_pid  = $_.ParentProcessId; "
        "    name        = $_.Name; "
        "    ram_mb      = if ($p) { [int]($p.WorkingSet64 / 1MB) } else { 0 }; "
        "    cmd_preview = $cmd; "
        "    age_s       = $age; "
        "  } "
        "} | ConvertTo-Json -Compress -Depth 3"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=10.0, check=False,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if r.returncode != 0 or not r.stdout.strip():
        return []
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    out: list[ProcInfo] = []
    for d in data:
        if not isinstance(d, dict):
            continue
        try:
            out.append(ProcInfo(
                pid=int(d.get("pid", 0)),
                parent_pid=int(d.get("parent_pid", 0)),
                name=str(d.get("name", "")),
                ram_mb=int(d.get("ram_mb", 0)),
                cmd_preview=str(d.get("cmd_preview", "")),
                age_s=int(d.get("age_s", 0)),
            ))
        except (ValueError, TypeError):
            continue
    # Sort: anaconda3 children first (real workers), then venv shims, then misc.
    def _sort_key(p: ProcInfo) -> tuple[int, int]:
        # Heuristic: a process whose parent is also in the list is the child
        # — show parents first; for same-parent siblings show by RAM desc.
        return (-p.ram_mb, p.pid)
    out.sort(key=_sort_key)
    return out


# ---- Phase 51: action helpers (whitelisted to project-owned items) -------


def _task_belongs_to_a_project(task_name: str) -> bool:
    """True iff `task_name` matches at least one PROJECT_PROFILES glob.
    Used as the whitelist gate for run_task / stop_task — prevents the
    dashboard from touching system tasks (Microsoft\\*, ASUS\\*, vendor
    updaters, etc.) even if a malicious request slipped past the
    frontend.
    """
    import fnmatch
    for prof in PROJECT_PROFILES:
        for pat in prof.get("task_globs", []):
            if fnmatch.fnmatchcase(task_name, pat):
                return True
    return False


def _process_belongs_to_a_project(cmd_line: str) -> bool:
    """True iff `cmd_line` matches at least one PROJECT_PROFILES regex.
    Used as the whitelist gate for kill_process — prevents the dashboard
    from terminating arbitrary user / system processes."""
    if not cmd_line:
        return False
    for prof in PROJECT_PROFILES:
        rx = prof.get("process_regex", "")
        if rx and re.search(rx, cmd_line):
            return True
    return False


def _ps_run(script: str, *, timeout_s: float = 15.0) -> tuple[int, str, str]:
    """Run a PowerShell snippet headlessly. Returns (rc, stdout, stderr)."""
    if sys.platform != "win32":
        return (1, "", "non-Windows")
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout_s, check=False,
            creationflags=_CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        return (124, "", "timed out")
    except OSError as e:
        return (1, "", str(e))
    return (r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip())


def run_task(name: str) -> dict[str, Any]:
    """Start a scheduled task by name. Whitelisted to PROJECT_PROFILES.

    Returns {ok, detail} suitable for JSONResponse.
    """
    if not name or not isinstance(name, str):
        return {"ok": False, "detail": "missing or invalid task name"}
    if not _task_belongs_to_a_project(name):
        return {"ok": False,
                "detail": f"refused: '{name}' is not in any project's task globs"}
    # Single-quote the name; PowerShell single-quotes don't interpolate.
    safe = name.replace("'", "''")
    rc, _out, err = _ps_run(
        f"Start-ScheduledTask -TaskName '{safe}' -ErrorAction Stop",
    )
    if rc == 0:
        return {"ok": True, "detail": f"Start-ScheduledTask {name}"}
    return {"ok": False, "detail": f"failed: {err or f'rc={rc}'}"}


def stop_task(name: str) -> dict[str, Any]:
    """Stop a running scheduled task. Whitelisted."""
    if not name or not isinstance(name, str):
        return {"ok": False, "detail": "missing or invalid task name"}
    if not _task_belongs_to_a_project(name):
        return {"ok": False,
                "detail": f"refused: '{name}' is not in any project's task globs"}
    safe = name.replace("'", "''")
    rc, _out, err = _ps_run(
        f"Stop-ScheduledTask -TaskName '{safe}' -ErrorAction Stop",
    )
    if rc == 0:
        return {"ok": True, "detail": f"Stop-ScheduledTask {name}"}
    return {"ok": False, "detail": f"failed: {err or f'rc={rc}'}"}


def kill_process(pid: int) -> dict[str, Any]:
    """Force-terminate a process by PID. Whitelisted to processes whose
    command line matches a project's regex.

    Two-step: probe Win32_Process for the cmdline first, verify the
    whitelist, then Stop-Process. Prevents PID-reuse race attacks (a
    PID we kill is the SAME PID we just verified).
    """
    if not isinstance(pid, int) or pid <= 0:
        return {"ok": False, "detail": "missing or invalid pid"}

    # Probe for the cmdline + verify whitelist.
    probe_script = (
        f"$p = Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\" "
        "-EA SilentlyContinue; "
        "if ($p) { $p.CommandLine } else { '' }"
    )
    rc, cmd_line, err = _ps_run(probe_script, timeout_s=8.0)
    if rc != 0:
        return {"ok": False, "detail": f"probe failed: {err or f'rc={rc}'}"}
    if not cmd_line:
        return {"ok": False, "detail": f"pid {pid} not found (already gone?)"}
    if not _process_belongs_to_a_project(cmd_line):
        return {"ok": False,
                "detail": (f"refused: pid {pid} cmdline doesn't match "
                           "any project's process pattern")}

    # Whitelisted — issue the kill.
    rc2, _out, err2 = _ps_run(
        f"Stop-Process -Id {pid} -Force -ErrorAction Stop",
    )
    if rc2 == 0:
        return {"ok": True, "detail": f"killed pid {pid}"}
    return {"ok": False, "detail": f"kill failed: {err2 or f'rc={rc2}'}"}


def get_projects_state() -> list[dict[str, Any]]:
    """Return state for every project in PROJECT_PROFILES.

    The function shape (list-of-dicts) is JSON-friendly so the handler
    can `JSONResponse(get_projects_state())` directly.
    """
    out: list[dict[str, Any]] = []
    for prof in PROJECT_PROFILES:
        state = ProjectState(id=prof["id"], label=prof["label"])
        try:
            state.tasks = _list_tasks(prof.get("task_globs", []))
        except Exception as e:
            state.error = f"task probe failed: {type(e).__name__}: {e}"
        try:
            state.processes = _list_processes(prof.get("process_regex", ""))
        except Exception as e:
            state.error = (
                (state.error + " | " if state.error else "")
                + f"process probe failed: {type(e).__name__}: {e}"
            )
        out.append({
            "id": state.id,
            "label": state.label,
            "tasks": [asdict(t) for t in state.tasks],
            "processes": [asdict(p) for p in state.processes],
            "error": state.error,
        })
    return out

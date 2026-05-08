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
    script = (
        "Get-ScheduledTask | Where-Object { " + or_clauses + " } | "
        "ForEach-Object { "
        "  $info = Get-ScheduledTaskInfo $_; "
        "  $repeat = ($_.Triggers | ForEach-Object { "
        "      if ($_.Repetition -and $_.Repetition.Interval) { "
        "          $_.Repetition.Interval } "
        "  }) | Select-Object -First 1; "
        "  [PSCustomObject]@{ "
        "    name        = $_.TaskName; "
        "    state       = $_.State.ToString(); "
        "    hidden      = $_.Settings.Hidden; "
        "    exe         = (Split-Path $_.Actions[0].Execute -Leaf); "
        "    repeat      = $repeat; "
        "    last_run    = if ($info.LastRunTime -and $info.LastRunTime.Year -gt 1900) "
        "                  { $info.LastRunTime.ToString('o') } else { $null }; "
        "    last_result = $info.LastTaskResult; "
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
    # Single-result objects come back as dict not list — normalize.
    if isinstance(data, dict):
        data = [data]
    out: list[TaskInfo] = []
    for d in data:
        if not isinstance(d, dict):
            continue
        try:
            out.append(TaskInfo(
                name=str(d.get("name", "")),
                state=str(d.get("state", "Unknown")),
                hidden=bool(d.get("hidden", False)),
                exe=str(d.get("exe", "")),
                repeat=(str(d["repeat"]) if d.get("repeat") else None),
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

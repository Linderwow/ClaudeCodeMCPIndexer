"""Phase 32: process hygiene — keep the system free of orphaned children.

Three problems we solve here:

1. **Orphaned MCP servers.** When Claude Code crashes or exits ungracefully,
   the `code-rag mcp` subprocess it launched may not get a clean stdin EOF.
   On Windows, the subprocess can outlive its parent indefinitely, slowly
   accumulating across restarts and consuming RAM (~600 MB each on a
   loaded index). The fix is a parent-death watchdog: the MCP server polls
   its parent PID periodically and self-exits when the parent goes away.

2. **Duplicate dashboard / watcher instances.** Task Scheduler is
   configured with `MultipleInstancesPolicy=IgnoreNew`, but the user can
   still trigger duplicates by running the task manually, or via venv
   launcher quirks. A singleton lockfile lets each component refuse to
   start a second copy.

3. **Identifying orphans for cleanup.** A reaper utility scans live
   `code_rag` processes and flags those whose ancestor is no longer
   `claude.exe` (for MCP) or no longer Task Scheduler / a real launcher
   (for dashboard / watcher).

All functionality is stdlib-only — no psutil dependency. We use Windows
Win32 APIs via ctypes to check process aliveness (works on Windows; on
other platforms `os.kill(pid, 0)` is the standard approach).
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from code_rag.logging import get

log = get(__name__)


# ---- process aliveness probe ----------------------------------------------


def is_process_alive(pid: int) -> bool:
    """True iff a process with the given PID is currently running.

    Cross-platform via two strategies:
      * Windows: OpenProcess + GetExitCodeProcess via ctypes.
      * POSIX:   `os.kill(pid, 0)` — signal 0 just probes.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        return _is_alive_windows(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it — still alive.
        return True
    return True


def _is_alive_windows(pid: int) -> bool:
    """Win32 OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION).

    Returns True iff the process exists. Uses
    `PROCESS_QUERY_LIMITED_INFORMATION` (0x1000) which is the lowest-priv
    handle that still works against most processes.

    When OpenProcess returns NULL we MUST distinguish "process doesn't
    exist" from "process exists but we lack rights to query it".
    Examples of the latter: SYSTEM-owned services like svchost.exe
    (Task Scheduler runs under svchost) — without this check, the reaper
    incorrectly flags every Task-Scheduler-launched process as orphaned
    because it can't see the live svchost ancestor. The discriminator is
    GetLastError():
      * ERROR_INVALID_PARAMETER (87) → no such PID → return False.
      * ERROR_ACCESS_DENIED (5)      → exists, can't access → return True.
      * Other errors                 → conservatively return True
                                        (avoid false-positive "dead" classification).
    """
    import ctypes
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    ERROR_INVALID_PARAMETER = 87
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        err = kernel32.GetLastError()
        if err == ERROR_INVALID_PARAMETER:
            return False
        # ACCESS_DENIED or anything else → process exists, just opaque to us.
        return True
    try:
        exit_code = ctypes.c_ulong(0)
        ok = kernel32.GetExitCodeProcess(h, ctypes.byref(exit_code))
        if not ok:
            return True   # we have a handle but can't read exit code → alive
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(h)


# ---- parent-death watchdog ------------------------------------------------


def start_parent_death_watchdog(
    *,
    parent_pid: int | None = None,
    poll_interval_s: float = 10.0,
    grace_s: float = 1.0,
) -> threading.Thread:
    """Spawn a daemon thread that exits the process when our parent dies.

    Designed for the MCP server: Claude Code is the parent; if it dies,
    we should die too. Without this, the MCP server can outlive Claude
    Code restarts and accumulate as orphans.

    Args:
        parent_pid: PID to watch. Defaults to `os.getppid()` at call time —
            i.e. whoever launched us right now. On Windows venv stubs, the
            real parent is the venv stub which itself dies when claude.exe
            dies, so this still gives the right behavior (chain death).
        poll_interval_s: How often to check. 10s is fine — we don't need
            instant termination, just bounded staleness.
        grace_s: Wait this many seconds after detecting parent death before
            exiting, in case the parent is being replaced by another
            (e.g. Claude Code restart with handoff). Practically this
            doesn't happen for stdio-attached MCP servers, but it's cheap
            insurance against false positives.

    Returns the watchdog thread (already started, daemon=True).

    The watchdog calls `os._exit(0)` rather than a normal sys.exit() to
    bypass `atexit` handlers — at orphan-detection time we want to
    disappear immediately, not run a 10s graceful shutdown that will
    just be killed anyway.
    """
    target_pid = parent_pid if parent_pid is not None else os.getppid()

    def _watch() -> None:
        # Initial wait — don't trip on a slow startup race.
        time.sleep(poll_interval_s)
        while True:
            if not is_process_alive(target_pid):
                # Re-check after grace period to avoid false positives.
                time.sleep(grace_s)
                if not is_process_alive(target_pid):
                    log.info("proc_hygiene.parent_died",
                             parent_pid=target_pid)
                    os._exit(0)
            time.sleep(poll_interval_s)

    t = threading.Thread(
        target=_watch, daemon=True, name="parent-death-watchdog",
    )
    t.start()
    log.info("proc_hygiene.watchdog_started",
             parent_pid=target_pid, poll_s=poll_interval_s)
    return t


# ---- singleton lockfile ---------------------------------------------------


class SingletonLock:
    """A PID-based lockfile to enforce one running instance of a component.

    Used by the dashboard and watcher to prevent duplicate spawns when
    Task Scheduler retries, the user manually invokes, or venv launcher
    quirks duplicate the process.

    Usage:
        with SingletonLock(path) as lock:
            if not lock.acquired:
                sys.exit(0)   # another instance is alive; bow out
            ...run main loop...

    Acquire is non-blocking: if the lockfile exists with a live PID, we
    return False immediately. If it exists with a dead PID (stale from a
    previous crash), we steal the lock.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fd: int | None = None
        self.acquired: bool = False

    def __enter__(self) -> "SingletonLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Try to read an existing lock.
        if self._path.exists():
            try:
                existing = int(self._path.read_text("utf-8").strip())
            except (ValueError, OSError):
                existing = -1
            if existing > 0 and is_process_alive(existing):
                # Live owner — don't steal.
                log.info("proc_hygiene.singleton_busy",
                         path=str(self._path), holder_pid=existing)
                self.acquired = False
                return self
            # Stale — overwrite below.
            log.info("proc_hygiene.singleton_stale",
                     path=str(self._path), stale_pid=existing)

        self._path.write_text(str(os.getpid()), encoding="utf-8")
        self.acquired = True
        log.info("proc_hygiene.singleton_acquired",
                 path=str(self._path), pid=os.getpid())
        return self

    def __exit__(self, *_exc: object) -> None:
        if not self.acquired:
            return
        # Only delete if our PID still owns the file (defensive: if a
        # second instance somehow stole the lock, don't delete its file).
        try:
            owner = int(self._path.read_text("utf-8").strip())
            if owner == os.getpid():
                self._path.unlink(missing_ok=True)
        except (OSError, ValueError):
            pass


# ---- reaper: find + kill orphaned code_rag processes ---------------------


# Kinds of code_rag subprocess we know about and the parent process names
# that legitimately spawn them. A process is "orphan" iff its ancestor
# chain doesn't include any of these expected parents.
_KIND_PATTERNS: tuple[tuple[str, str], ...] = (
    ("mcp",       "code_rag mcp"),
    ("dashboard", "code_rag dashboard"),
    ("watcher",   "code_rag.autostart_bootstrap"),
    ("watch_cli", "code_rag watch"),
)

# Parent process names that legitimately host each kind. If the ancestor
# chain doesn't contain ANY of these, the process is orphaned.
_EXPECTED_PARENTS: dict[str, frozenset[str]] = {
    "mcp":       frozenset({"claude.exe", "claude"}),
    # Task Scheduler launches via svchost.exe → conhost.exe → our exe; the
    # immediate parent is svchost.exe in some configs. We accept both
    # taskeng.exe (older) and svchost.exe (newer Windows).
    "dashboard": frozenset({"svchost.exe", "taskeng.exe", "explorer.exe",
                            "claude.exe", "claude"}),
    "watcher":   frozenset({"svchost.exe", "taskeng.exe", "explorer.exe"}),
    "watch_cli": frozenset({"svchost.exe", "taskeng.exe", "explorer.exe",
                            "cmd.exe", "powershell.exe", "pwsh.exe", "bash.exe"}),
}


class ProcessInfo:
    """Lightweight subset of psutil.Process — what the reaper needs to
    decide whether to kill a code_rag process."""

    __slots__ = ("pid", "ppid", "name", "cmdline", "create_time", "kind",
                 "ancestors_alive", "ancestor_names", "is_orphan", "reason")

    def __init__(
        self, pid: int, ppid: int, name: str, cmdline: str,
        create_time: float, kind: str | None,
    ) -> None:
        self.pid = pid
        self.ppid = ppid
        self.name = name
        self.cmdline = cmdline
        self.create_time = create_time
        self.kind = kind            # "mcp" | "dashboard" | "watcher" | "watch_cli" | None
        self.ancestors_alive = True
        self.ancestor_names: list[str] = []
        self.is_orphan = False
        self.reason: str | None = None

    def __repr__(self) -> str:
        return (f"<ProcessInfo pid={self.pid} kind={self.kind} "
                f"orphan={self.is_orphan} reason={self.reason!r}>")


def _classify_kind(cmdline: str) -> str | None:
    for kind, pattern in _KIND_PATTERNS:
        if pattern in cmdline:
            return kind
    return None


def list_code_rag_processes() -> list[ProcessInfo]:
    """Enumerate all live code_rag-related processes via WMI on Windows.

    Uses `Win32_Process` so we get the command line (not just the image
    name) and parent PID. Falls back to an empty list on non-Windows for
    now — the orphan problem is Windows-specific in practice (the user's
    setup is Windows-only).
    """
    if sys.platform != "win32":
        return []
    import subprocess
    # WMIC is deprecated; use PowerShell + Get-CimInstance which is
    # supported on every modern Windows.
    script = (
        "Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe' OR Name='python.exe'\" "
        "-EA SilentlyContinue | "
        "Where-Object { $_.CommandLine -and $_.CommandLine -match 'code_rag' } | "
        "Select-Object ProcessId, ParentProcessId, Name, CommandLine, CreationDate | "
        "ConvertTo-Json -Compress"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=15.0, check=False,
            creationflags=0x08000000,  # CREATE_NO_WINDOW
        )
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return []
    if r.returncode != 0 or not r.stdout.strip():
        return []
    import json
    try:
        data = json.loads(r.stdout)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    out: list[ProcessInfo] = []
    for d in data:
        cmdline = str(d.get("CommandLine") or "")
        kind = _classify_kind(cmdline)
        out.append(ProcessInfo(
            pid=int(d.get("ProcessId") or 0),
            ppid=int(d.get("ParentProcessId") or 0),
            name=str(d.get("Name") or ""),
            cmdline=cmdline,
            create_time=0.0,        # not populated; not used for orphan check
            kind=kind,
        ))
    return out


def _process_name(pid: int) -> str:
    """Return the process image name for a PID, or '' if not found."""
    if sys.platform != "win32":
        return ""
    import subprocess
    script = (
        f"$p = Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\" -EA SilentlyContinue; "
        "if ($p) { $p.Name } else { '' }"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=5.0, check=False,
            creationflags=0x08000000,
        )
    except (subprocess.TimeoutExpired, OSError):
        return ""
    return r.stdout.strip()


def classify_orphans(procs: list[ProcessInfo]) -> list[ProcessInfo]:
    """Stamp `is_orphan` + `reason` on each input process.

    Walks each process's ancestor chain (PID → PPID → grandparent's PPID …)
    until we either:
      * Find a process whose name is in `_EXPECTED_PARENTS[kind]` → not orphan
      * Hit a dead PID → orphan (the legitimate ancestor died)
      * Hit System / Idle (PID 0/4) → orphan (chain reached top without match)
      * Hit our own .venv launcher stub → keep walking (it's our intermediate)

    Returns the same list, with fields populated.
    """
    expected_per_proc = {
        p.pid: _EXPECTED_PARENTS.get(p.kind or "", frozenset()) for p in procs
    }
    own_pids = {p.pid for p in procs}
    for p in procs:
        if p.kind is None:
            continue
        expected = expected_per_proc[p.pid]
        if not expected:
            continue
        # Walk up. Limit depth to 8 to defend against accidental cycles.
        cur_pid = p.ppid
        chain: list[str] = []
        for _ in range(8):
            if cur_pid <= 0 or cur_pid in (4,):  # 0=Idle, 4=System
                p.is_orphan = True
                p.reason = "ancestor chain reached System without finding a legitimate parent"
                break
            if not is_process_alive(cur_pid):
                p.is_orphan = True
                p.reason = f"ancestor pid={cur_pid} no longer alive"
                p.ancestors_alive = False
                break
            name = _process_name(cur_pid)
            chain.append(f"{cur_pid}:{name}")
            if name and name.lower() in {n.lower() for n in expected}:
                # Found a legitimate ancestor — keep this process.
                break
            # If the ancestor is ANOTHER code_rag process (the .venv stub
            # launcher), keep walking to find the real Claude/Task Scheduler
            # ancestor.
            if cur_pid in own_pids:
                # Find that process and continue with its PPID.
                parent = next((q for q in procs if q.pid == cur_pid), None)
                cur_pid = parent.ppid if parent else 0
                continue
            # Anything else (e.g. random process that isn't a known parent)
            # could be legitimate (we don't enumerate every possibility),
            # so continue walking up the chain.
            cur_pid = _ppid_of(cur_pid)
        p.ancestor_names = chain
    return procs


def _ppid_of(pid: int) -> int:
    """Return the parent PID of a process, or 0 if unavailable."""
    if sys.platform != "win32":
        return 0
    import subprocess
    script = (
        f"$p = Get-CimInstance Win32_Process -Filter \"ProcessId={pid}\" -EA SilentlyContinue; "
        "if ($p) { $p.ParentProcessId } else { 0 }"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=5.0, check=False,
            creationflags=0x08000000,
        )
    except (subprocess.TimeoutExpired, OSError):
        return 0
    try:
        return int(r.stdout.strip() or 0)
    except ValueError:
        return 0


def kill_pid(pid: int) -> bool:
    """Best-effort terminate a process. Returns True if the process is no
    longer alive after the call (whether we killed it or it was already
    gone)."""
    if not is_process_alive(pid):
        return True
    if sys.platform == "win32":
        import ctypes
        PROCESS_TERMINATE = 0x0001
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        h = kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
        if not h:
            return not is_process_alive(pid)
        try:
            kernel32.TerminateProcess(h, 1)
        finally:
            kernel32.CloseHandle(h)
    else:
        try:
            os.kill(pid, 9)
        except (ProcessLookupError, PermissionError):
            return not is_process_alive(pid)
    # Brief wait to confirm termination.
    time.sleep(0.2)
    return not is_process_alive(pid)


def reap_orphans(*, kill: bool = False) -> dict[str, list[dict[str, object]]]:
    """Find (and optionally kill) orphaned code_rag processes.

    Returns a structured report:
        {
            "alive":   [...legitimate processes...],
            "orphans": [...orphaned processes...],
            "killed":  [...PIDs successfully terminated...],
        }
    """
    procs = classify_orphans(list_code_rag_processes())
    alive = [_proc_to_dict(p) for p in procs if not p.is_orphan]
    orphans = [_proc_to_dict(p) for p in procs if p.is_orphan]
    killed: list[int] = []
    if kill:
        for p in procs:
            if not p.is_orphan:
                continue
            if kill_pid(p.pid):
                killed.append(p.pid)
    return {"alive": alive, "orphans": orphans, "killed": killed}


def _proc_to_dict(p: ProcessInfo) -> dict[str, object]:
    return {
        "pid": p.pid,
        "ppid": p.ppid,
        "name": p.name,
        "kind": p.kind,
        "cmdline_preview": (p.cmdline[:200] + "…") if len(p.cmdline) > 200 else p.cmdline,
        "is_orphan": p.is_orphan,
        "reason": p.reason,
        "ancestor_chain": p.ancestor_names,
    }

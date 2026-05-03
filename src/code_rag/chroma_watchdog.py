"""Phase 36-A: Chroma self-healing watchdog.

Today's failure mode: the chroma.sqlite3 grew to 1.8 GB and HNSW
data_level0.bin to 2.5 GB after multiple days of partial writes from
crashes/restarts. `client.count()` deadlocked indefinitely. Every
process trying to attach Chroma — MCP servers, dashboard, watcher —
hung forever.

Detection: probe the Chroma store with `count()` under a hard subprocess
timeout. We MUST run the probe in a subprocess (not just `asyncio.wait_for`)
because a hang inside Chroma's C++ code holds the GIL and ignores asyncio
cancellation. Subprocess gives us the option to actually kill it.

Recovery: if the probe times out, the store is unrecoverable in-place.
Wipe the chroma/ subdirectory, restart the watcher, let it re-embed
against the FTS source-of-truth. Re-embed time on this corpus is
~1-2h with the Phase 35 streaming indexer; that's the best we can do
without keeping a redundant vector dump.

Not run by the MCP server itself (which is read-only and can't trigger
a wipe). Wired as a periodic task — see scripts/install-watchdog-autostart.ps1.
"""
from __future__ import annotations

import contextlib
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from code_rag.logging import get

log = get(__name__)


# ---- the actual probe ------------------------------------------------------


_PROBE_SCRIPT = r'''
import sys, os, time, json, threading
sys.path.insert(0, r"{src}")

# Phase 38 (audit fix): if our parent dies (taskkill / Stop-ScheduledTask /
# OS shutdown) WHILE we are stuck inside Chroma's C++ code, Windows does
# NOT cascade-kill us — we orphan, hold a Chroma file lock, and the next
# watchdog cycle ALSO times out because the directory is still locked.
# A daemon thread watching the parent PID self-terminates this probe
# within `parent_check_s` seconds of the parent's death.
_PARENT_PID = {ppid}
_PARENT_CHECK_S = 2.0


def _watch_parent():
    while True:
        try:
            if sys.platform == "win32":
                # On Windows, signaling 0 to a dead PID raises OSError.
                import ctypes
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                STILL_ACTIVE = 259
                h = ctypes.windll.kernel32.OpenProcess(
                    PROCESS_QUERY_LIMITED_INFORMATION, 0, _PARENT_PID,
                )
                if not h:
                    os._exit(2)
                code = ctypes.c_ulong()
                ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
                ctypes.windll.kernel32.CloseHandle(h)
                if code.value != STILL_ACTIVE:
                    os._exit(2)
            else:
                os.kill(_PARENT_PID, 0)
        except (OSError, ProcessLookupError):
            os._exit(2)
        time.sleep(_PARENT_CHECK_S)


threading.Thread(target=_watch_parent, daemon=True).start()

import chromadb
t0 = time.monotonic()
try:
    client = chromadb.PersistentClient(path=r"{path}")
    colls = client.list_collections()
    counts = {{}}
    for c in colls:
        counts[c.name] = c.count()
    out = {{"ok": True, "elapsed_s": time.monotonic()-t0, "counts": counts}}
except Exception as e:
    out = {{"ok": False, "elapsed_s": time.monotonic()-t0, "error": f"{{type(e).__name__}}: {{e}}"}}
print(json.dumps(out))
'''


def probe_chroma(
    chroma_dir: Path,
    *,
    timeout_s: float = 30.0,
    python_exe: str | None = None,
) -> dict[str, Any]:
    """Subprocess-isolated Chroma probe with hard timeout.

    Returns: {
        "ok":        True iff probe completed cleanly
        "elapsed_s": probe wall time (-1 if timed out)
        "counts":    {collection_name: int} on success
        "error":     error message on failure
        "timed_out": True iff we hit the wall-clock kill
    }

    Uses a fresh subprocess for two reasons:
      1. Hangs inside Chroma's C++ extension hold the GIL and ignore
         asyncio.wait_for cancellation. Process kill is the only escape.
      2. Even successful probes leave background mmap/lock state that
         interferes with subsequent in-process Chroma usage on Windows.
    """
    py = python_exe or sys.executable
    src_dir = Path(__file__).resolve().parents[2]   # → src/
    import os as _os
    script = _PROBE_SCRIPT.format(
        src=str(src_dir),
        path=str(chroma_dir),
        ppid=_os.getpid(),
    )
    t0 = time.monotonic()
    try:
        r = subprocess.run(
            [py, "-c", script],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=timeout_s, check=False,
            creationflags=0x08000000 if sys.platform == "win32" else 0,
        )
    except subprocess.TimeoutExpired:
        log.warning("chroma_watchdog.probe_timeout",
                    chroma_dir=str(chroma_dir), timeout_s=timeout_s)
        return {
            "ok": False,
            "elapsed_s": time.monotonic() - t0,
            "error": f"probe subprocess hung past {timeout_s}s",
            "timed_out": True,
        }
    if r.returncode != 0:
        return {
            "ok": False,
            "elapsed_s": time.monotonic() - t0,
            "error": f"probe rc={r.returncode}: {r.stderr[:300]}",
            "timed_out": False,
        }
    try:
        import json
        out = json.loads((r.stdout or "").strip().splitlines()[-1])
    except (ValueError, IndexError, KeyError) as e:
        return {
            "ok": False,
            "elapsed_s": time.monotonic() - t0,
            "error": f"probe output unparseable: {e}",
            "timed_out": False,
        }
    out["timed_out"] = False
    return out


# ---- the recovery action ---------------------------------------------------


def wipe_chroma(chroma_dir: Path, *, also_index_meta: Path | None = None) -> bool:
    """Best-effort wipe of the chroma/ directory.

    Returns True iff the directory is gone (or never existed) after
    the call. Defensive: failures during shutil.rmtree are swallowed
    and reported via return value, since this is recovery code that
    must not raise into the watchdog loop.

    The caller is responsible for ensuring no process is currently
    using chroma — typically by killing the watcher first.
    """
    if not chroma_dir.exists():
        return True
    try:
        shutil.rmtree(chroma_dir, ignore_errors=False)
    except OSError as e:
        log.error("chroma_watchdog.wipe_failed",
                  chroma_dir=str(chroma_dir), err=str(e))
        # Try ignore_errors=True as a fallback so a single locked file
        # doesn't block the whole recovery.
        with contextlib.suppress(OSError):
            shutil.rmtree(chroma_dir, ignore_errors=True)
    if also_index_meta is not None and also_index_meta.exists():
        with contextlib.suppress(OSError):
            also_index_meta.unlink()
    return not chroma_dir.exists()


# ---- top-level orchestration ----------------------------------------------


def heal_if_unhealthy(
    chroma_dir: Path,
    *,
    probe_timeout_s: float = 30.0,
    index_meta_path: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Probe Chroma; if unhealthy, kill the watcher and wipe the store.

    Returns a dict report:
        {
            "probed":   probe_chroma() result
            "healed":   True iff we triggered a wipe (or dry_run==True
                        and we WOULD have wiped)
            "wipe_ok":  True iff the wipe succeeded (only set if healed)
        }

    Caller responsibility: schedule this periodically (e.g. via Task
    Scheduler every 10 min) so corruption is caught within bounded time.
    """
    probe = probe_chroma(chroma_dir, timeout_s=probe_timeout_s)
    report: dict[str, Any] = {"probed": probe, "healed": False}
    if probe.get("ok"):
        return report
    log.warning("chroma_watchdog.unhealthy",
                error=probe.get("error"), timed_out=probe.get("timed_out"))
    if dry_run:
        report["healed"] = True
        report["wipe_ok"] = None
        return report
    # Kill watcher + MCP servers BEFORE wiping so they don't grab a
    # fresh lock on the file we're trying to delete.
    _kill_chroma_holders()
    wipe_ok = wipe_chroma(chroma_dir, also_index_meta=index_meta_path)
    report["healed"] = True
    report["wipe_ok"] = wipe_ok
    log.info("chroma_watchdog.healed", wipe_ok=wipe_ok)
    return report


def _kill_chroma_holders() -> None:
    """Kill processes likely to hold Chroma open: watcher + MCP servers.

    The dashboard also opens Chroma indirectly (for status) but it's
    quick and idempotent — we don't kill it to keep the UI alive.
    """
    if sys.platform != "win32":
        return
    script = (
        "Get-CimInstance Win32_Process -EA SilentlyContinue | "
        "Where-Object { $_.CommandLine -and ($_.CommandLine -match "
        "'autostart_bootstrap|code_rag watch|code_rag mcp') -and "
        "$_.CommandLine -notmatch 'tasks/b' } | "
        "ForEach-Object { Stop-Process -Id $_.ProcessId -Force -EA SilentlyContinue }"
    )
    with contextlib.suppress(subprocess.TimeoutExpired, OSError):
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True,
            timeout=15.0, check=False,
            creationflags=0x08000000,
        )

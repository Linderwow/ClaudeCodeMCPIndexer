"""Pure orchestration logic for the dashboard.

All Windows-y subprocess + HTTP shelling lives here so the Starlette layer is
thin. Each operation returns a JSON-serializable dict so the UI can render
it without inventing schemas.

Design rules
------------
- Every operation is idempotent. "Start" on something already-running is a no-op
  with `ok=True`. "Stop" on something already-stopped is the same.
- We NEVER touch Claude's MCP subprocesses. Their lifecycle belongs to Claude.
  We only manage: LM Studio server, loaded models, the `code-rag-watch`
  scheduled task.
- Errors are returned in the result, not raised. The UI surfaces them.
- All shellouts have a hard timeout so a hung subprocess can't lock the
  dashboard.
"""
from __future__ import annotations

import contextlib
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

from code_rag.config import Settings
from code_rag.lms_ctl import find_lms

WATCHER_TASK_NAME = "code-rag-watch"

# Hard caps on how long a single subprocess call can run.
_LMS_FAST_TIMEOUT = 10.0          # ps, unload, server status
_LMS_LOAD_TIMEOUT = 300.0         # initial model load can take minutes
_LMS_START_TIMEOUT = 30.0         # server startup
_TASK_OP_TIMEOUT = 15.0           # Start/Stop-ScheduledTask + Get-* queries

# CREATE_NO_WINDOW so PowerShell/lms shellouts don't flash a console.
_CREATE_NO_WINDOW = 0x08000000


# ---- result envelope --------------------------------------------------------


@dataclass
class StepResult:
    """One discrete operation's outcome."""
    name: str
    ok: bool
    detail: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "ok": self.ok,
            "detail": self.detail, "duration_ms": round(self.duration_ms, 1),
        }


@dataclass
class CompositeResult:
    """A sequence of StepResults from a multi-step operation."""
    steps: list[StepResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps)

    def add(self, step: StepResult) -> None:
        self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "steps": [s.to_dict() for s in self.steps],
        }


# ---- status -----------------------------------------------------------------


def get_status(settings: Settings) -> dict[str, Any]:
    """Snapshot of every component the dashboard cares about.

    Each component lives behind a separate subprocess / HTTP call. PowerShell
    process startup is ~1.5s on Windows, so 5 sequential queries bust the
    UI's 2s poll cadence. We do two things to fit the budget:

      1. Combine the four PowerShell-driven queries (watcher state + info,
         pythonw PIDs, RAM) into ONE PowerShell invocation that emits JSON.
      2. Run the four top-level fetches (LM Studio probe, the combined PS
         query, lms ps, nvidia-smi) concurrently in a thread pool.

    Net wall time: ~1.5s instead of ~6s.
    """
    from concurrent.futures import ThreadPoolExecutor

    # The PS-batch result feeds both _watcher_status and the RAM portion of
    # _resources_status, so we compute it once and split.
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_lms      = ex.submit(_lms_status, settings)
        f_psbatch  = ex.submit(_powershell_batch)
        f_index    = ex.submit(_index_status, settings)
        f_gpu      = ex.submit(_gpu_status_via_nvidia_smi)

    ps = f_psbatch.result()
    return {
        "ts":         _now_iso(),
        "lm_studio":  f_lms.result(),
        "watcher":    _watcher_from_batch(ps),
        "index":      f_index.result(),
        "resources":  {"ram": _ram_from_batch(ps), "gpu": f_gpu.result()},
    }


def _powershell_batch() -> dict[str, Any]:
    """One PowerShell invocation -> watcher state + info + pythonw PIDs + RAM.

    Returns {} on any failure; callers handle missing keys.
    """
    # Match WATCHER processes specifically by command line, not by RAM size.
    # The naive "any pythonw > 200 MB" filter swept up Claude Code's MCP
    # subprocesses (each ~1.3 GB) and labeled them as watcher PIDs in the UI.
    # Win32_Process gives us the CommandLine reliably.
    script = (
        f"$task = Get-ScheduledTask -TaskName '{WATCHER_TASK_NAME}' -EA SilentlyContinue;\n"
        f"$info = Get-ScheduledTaskInfo -TaskName '{WATCHER_TASK_NAME}' -EA SilentlyContinue;\n"
        "$pids = @(Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe'\" -EA SilentlyContinue |\n"
        "  Where-Object { $_.CommandLine -and ($_.CommandLine -like '*code_rag.autostart_bootstrap*' "
        "                                  -or $_.CommandLine -like '*code_rag watch*') } |\n"
        "  Sort-Object CreationDate |\n"
        "  Select-Object -ExpandProperty ProcessId);\n"
        "$os = Get-CimInstance Win32_OperatingSystem;\n"
        "[PSCustomObject]@{\n"
        "  taskState          = if ($task) { $task.State.ToString() } else { 'NotRegistered' }\n"
        "  lastRunTime        = if ($info) { $info.LastRunTime.ToString('o') } else { $null }\n"
        "  lastTaskResult     = if ($info) { $info.LastTaskResult } else { $null }\n"
        "  numberOfMissedRuns = if ($info) { $info.NumberOfMissedRuns } else { $null }\n"
        "  pids               = $pids\n"
        "  ramTotalKB         = $os.TotalVisibleMemorySize\n"
        "  ramFreeKB          = $os.FreePhysicalMemory\n"
        "} | ConvertTo-Json -Compress"
    )
    raw = _ps_query(script)
    with contextlib.suppress(json.JSONDecodeError, ValueError):
        return json.loads(raw or "{}") or {}
    return {}


def _watcher_from_batch(b: dict[str, Any]) -> dict[str, Any]:
    pids = b.get("pids") or []
    if not isinstance(pids, list):  # PS emits a single int when there's exactly one
        pids = [pids]
    return {
        "task_state":            (b.get("taskState") or "Unknown"),
        "last_run":              b.get("lastRunTime"),
        "last_result":           b.get("lastTaskResult"),
        "number_of_missed_runs": b.get("numberOfMissedRuns"),
        "pythonw_pids":          [int(x) for x in pids if x is not None],
    }


def _ram_from_batch(b: dict[str, Any]) -> dict[str, Any]:
    try:
        total_kb = int(b.get("ramTotalKB", 0))
        free_kb  = int(b.get("ramFreeKB", 0))
    except (TypeError, ValueError):
        return {}
    if total_kb <= 0:
        return {}
    return {
        "total_gb": round(total_kb / 1024 / 1024, 2),
        "used_gb":  round((total_kb - free_kb) / 1024 / 1024, 2),
        "free_gb":  round(free_kb / 1024 / 1024, 2),
    }


def _gpu_status_via_nvidia_smi() -> dict[str, Any] | None:
    """nvidia-smi is fast (~150ms) and tolerant of being absent; we run it
    directly rather than through PowerShell so its startup cost is minimal."""
    try:
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4.0, check=False,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return None
    if r.returncode != 0 or not r.stdout.strip():
        return None
    parts = [p.strip() for p in r.stdout.splitlines()[0].split(",")]
    if len(parts) < 5:
        return None
    try:
        return {
            "name":          parts[0],
            "vram_used_gb":  round(int(parts[1]) / 1024, 2),
            "vram_total_gb": round(int(parts[2]) / 1024, 2),
            "util_pct":      int(parts[3]),
            "temp_c":        int(parts[4]),
        }
    except (ValueError, IndexError):
        return None


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


# Cache for /v1/models — when LM Studio's worker is busy serving inference,
# /v1/models can take 2 s to respond, busting our 2 s poll cadence. The list
# of downloaded models only changes when the user runs `lms get`, so a 5 s
# TTL is plenty fresh and keeps polls snappy.
_LMS_AVAIL_CACHE: tuple[float, bool, list[str]] = (0.0, False, [])
_LMS_AVAIL_TTL_S = 5.0


def _lms_probe(base_url: str) -> tuple[bool, list[str]]:
    """Return (server_up, available_model_ids). 5s TTL cache."""
    global _LMS_AVAIL_CACHE
    now = time.monotonic()
    ts, server_up, ids = _LMS_AVAIL_CACHE
    if now - ts < _LMS_AVAIL_TTL_S and ts > 0:
        return server_up, ids
    server_up = False
    ids = []
    try:
        r = httpx.get(f"{base_url}/models", timeout=2.0)
        if r.status_code == 200:
            server_up = True
            data = r.json().get("data", []) or []
            ids = [str(m.get("id", "")) for m in data if m.get("id")]
    except (httpx.HTTPError, OSError):
        pass
    _LMS_AVAIL_CACHE = (now, server_up, ids)
    return server_up, ids


def _lms_status(settings: Settings) -> dict[str, Any]:
    base_url = settings.embedder.base_url.rstrip("/")
    server_up, available_ids = _lms_probe(base_url)
    loaded = _lms_ps()
    return {
        "server_up": server_up,
        "base_url":  base_url,
        "models_loaded":    loaded,
        "models_available": available_ids,
        "configured": {
            "embedder": settings.embedder.model,
            "reranker": settings.reranker.model if settings.reranker.model else None,
            "reranker_kind": settings.reranker.kind,
        },
    }


def _lms_ps() -> list[dict[str, Any]]:
    """Parse `lms ps` text output. Fixed-width columns; resilient to extra
    whitespace. Returns one dict per loaded model."""
    loc = find_lms()
    if loc.path is None:
        return []
    try:
        r = subprocess.run(
            [str(loc.path), "ps"],
            capture_output=True, text=True,
            timeout=_LMS_FAST_TIMEOUT, check=False,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
    if r.returncode != 0:
        return []
    return _parse_lms_ps(r.stdout)


def _parse_lms_ps(text: str) -> list[dict[str, Any]]:
    """Parse `lms ps` columnar output.

    Sample header: IDENTIFIER  MODEL  STATUS  SIZE  CONTEXT  PARALLEL  DEVICE  TTL
    Each row is whitespace-aligned. We don't try to be clever with column
    positions — split on 2+ spaces and zip with the header tokens.
    """
    out: list[dict[str, Any]] = []
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith("IDENTIFIER")),
        None,
    )
    if header_idx is None:
        return out
    headers = [h.strip().lower() for h in re.split(r"\s{2,}", lines[header_idx])]
    for ln in lines[header_idx + 1 :]:
        cells = re.split(r"\s{2,}", ln)
        if len(cells) < 4:
            continue
        # Right-pad short rows so we always have len(headers) cells.
        cells = (cells + [""] * len(headers))[: len(headers)]
        row = dict(zip(headers, cells, strict=False))
        size_mb = _parse_size_mb(row.get("size", ""))
        out.append({
            "id":      row.get("identifier", ""),
            "status":  row.get("status", ""),
            "size_mb": size_mb,
            "context": row.get("context", ""),
            "ttl":     row.get("ttl", "") or None,
            "device":  row.get("device", ""),
        })
    return out


def _parse_size_mb(s: str) -> float:
    """Parse '2.41 GB' / '512 MB' to MB float. Returns 0 if unparseable."""
    m = re.match(r"\s*([\d.]+)\s*(MB|GB|KB)\s*$", s.strip(), re.IGNORECASE)
    if not m:
        return 0.0
    val, unit = float(m.group(1)), m.group(2).upper()
    return {"KB": val / 1024, "MB": val, "GB": val * 1024}[unit]


def _index_status(settings: Settings) -> dict[str, Any]:
    p = settings.index_meta_path
    if not p.exists():
        return {"present": False}
    try:
        meta = json.loads(p.read_text("utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        return {"present": True, "error": str(e)}

    chunk_count: int | None = None
    fts_path = settings.fts_path
    if fts_path.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(fts_path, isolation_level=None)
            conn.execute("PRAGMA query_only = 1")
            row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
            chunk_count = int(row[0]) if row else 0
            conn.close()
        except sqlite3.Error:
            pass

    return {
        "present":         True,
        "chunks":          chunk_count,
        "embedder_kind":   meta.get("embedder_kind"),
        "embedder_model":  meta.get("embedder_model"),
        "embedder_dim":    meta.get("embedder_dim"),
        "schema_version": meta.get("schema_version"),
        "created_at":      meta.get("created_at"),
        "updated_at":      meta.get("updated_at"),
    }


# ---- start operations -------------------------------------------------------


def start_all(settings: Settings) -> CompositeResult:
    """Bring the whole stack up.

    Order matters: LM Studio first (the watcher needs the embedder), then
    pre-load models, then the Task Scheduler watcher.
    """
    res = CompositeResult()
    res.add(start_lms_server(settings))
    if not res.steps[-1].ok:
        return res

    res.add(load_model(settings.embedder.model, _LMS_LOAD_TIMEOUT))
    if settings.reranker.kind == "lm_chat" and settings.reranker.model:
        res.add(load_model(settings.reranker.model, _LMS_LOAD_TIMEOUT))

    res.add(start_watcher_task())
    return res


def start_lms_server(settings: Settings) -> StepResult:
    t0 = time.monotonic()
    base_url = settings.embedder.base_url.rstrip("/")
    if _server_reachable(base_url):
        return StepResult("start_lms_server", True, "already running",
                          (time.monotonic() - t0) * 1000)
    loc = find_lms()
    if loc.path is None:
        return StepResult("start_lms_server", False,
                          "lms.exe not found; is LM Studio installed?",
                          (time.monotonic() - t0) * 1000)
    try:
        subprocess.Popen(
            [str(loc.path), "server", "start"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, creationflags=_CREATE_NO_WINDOW,
            close_fds=True,
        )
    except OSError as e:
        return StepResult("start_lms_server", False, f"spawn failed: {e}",
                          (time.monotonic() - t0) * 1000)

    deadline = time.monotonic() + _LMS_START_TIMEOUT
    while time.monotonic() < deadline:
        if _server_reachable(base_url):
            return StepResult("start_lms_server", True, "ready",
                              (time.monotonic() - t0) * 1000)
        time.sleep(0.5)
    return StepResult("start_lms_server", False,
                      f"server did not respond within {_LMS_START_TIMEOUT:.0f}s",
                      (time.monotonic() - t0) * 1000)


def load_model(model: str, timeout_s: float) -> StepResult:
    t0 = time.monotonic()
    loc = find_lms()
    if loc.path is None:
        return StepResult(f"load_model({model})", False, "lms.exe not found",
                          (time.monotonic() - t0) * 1000)
    try:
        r = subprocess.run(
            [str(loc.path), "load", model],
            capture_output=True, text=True,
            timeout=timeout_s, check=False,
            creationflags=_CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        return StepResult(f"load_model({model})", False,
                          f"timed out after {timeout_s:.0f}s",
                          (time.monotonic() - t0) * 1000)
    if r.returncode == 0:
        return StepResult(f"load_model({model})", True, "loaded",
                          (time.monotonic() - t0) * 1000)
    return StepResult(f"load_model({model})", False,
                      (r.stderr or r.stdout).strip()[:240],
                      (time.monotonic() - t0) * 1000)


def start_watcher_task() -> StepResult:
    t0 = time.monotonic()
    out = _ps_query(
        f"try {{ Start-ScheduledTask -TaskName '{WATCHER_TASK_NAME}' -ErrorAction Stop; 'OK' }} "
        f"catch {{ 'ERR: ' + $_.Exception.Message }}"
    )
    detail = (out or "").strip()
    ok = detail == "OK"
    return StepResult("start_watcher_task", ok, detail or "no output",
                      (time.monotonic() - t0) * 1000)


# ---- stop operations --------------------------------------------------------


def stop_all(settings: Settings, *, stop_lm_studio: bool = True) -> CompositeResult:
    """Bring the stack down COMPLETELY.

    Stops watcher, unloads models, and stops the LM Studio server. The server
    has to go too — if it stays up, ANY request from a Claude Code MCP
    subprocess (or anything else pointing at /v1/embeddings) will JIT-reload
    the embedder, defeating the user's intent. The watcher autostart task is
    only stopped for THIS session — it'll fire again on next logon (the
    user's hands-off-on-reboot requirement).

    Pass `stop_lm_studio=False` to keep the server up (e.g. for the
    granular-stop case where the dashboard's per-card buttons did the work).
    """
    _ = settings
    res = CompositeResult()
    res.add(stop_watcher_task())
    res.add(unload_all_models())
    if stop_lm_studio:
        res.add(stop_lms_server())
    return res


def stop_watcher_task() -> StepResult:
    t0 = time.monotonic()
    state = _ps_query(
        f"$t = Get-ScheduledTask -TaskName '{WATCHER_TASK_NAME}' -ErrorAction SilentlyContinue; "
        "if ($t) { $t.State.ToString() } else { 'NotRegistered' }"
    )
    state = (state or "").strip()
    if state == "NotRegistered":
        return StepResult("stop_watcher_task", True,
                          "task not registered (nothing to stop)",
                          (time.monotonic() - t0) * 1000)
    if state == "Ready":
        return StepResult("stop_watcher_task", True, "already stopped",
                          (time.monotonic() - t0) * 1000)
    out = _ps_query(
        f"try {{ Stop-ScheduledTask -TaskName '{WATCHER_TASK_NAME}' -ErrorAction Stop; 'OK' }} "
        f"catch {{ 'ERR: ' + $_.Exception.Message }}"
    )
    detail = (out or "").strip()
    return StepResult("stop_watcher_task", detail == "OK", detail or "no output",
                      (time.monotonic() - t0) * 1000)


def unload_all_models() -> StepResult:
    t0 = time.monotonic()
    loc = find_lms()
    if loc.path is None:
        return StepResult("unload_all_models", False, "lms.exe not found",
                          (time.monotonic() - t0) * 1000)
    try:
        r = subprocess.run(
            [str(loc.path), "unload", "--all"],
            capture_output=True, text=True,
            timeout=_LMS_FAST_TIMEOUT, check=False,
            creationflags=_CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        return StepResult("unload_all_models", False, "timed out",
                          (time.monotonic() - t0) * 1000)
    if r.returncode == 0:
        return StepResult("unload_all_models", True,
                          (r.stdout or "").strip()[:240] or "all models unloaded",
                          (time.monotonic() - t0) * 1000)
    return StepResult("unload_all_models", False,
                      (r.stderr or r.stdout).strip()[:240],
                      (time.monotonic() - t0) * 1000)


def unload_model(model: str) -> StepResult:
    t0 = time.monotonic()
    loc = find_lms()
    if loc.path is None:
        return StepResult(f"unload_model({model})", False, "lms.exe not found",
                          (time.monotonic() - t0) * 1000)
    try:
        r = subprocess.run(
            [str(loc.path), "unload", model],
            capture_output=True, text=True,
            timeout=_LMS_FAST_TIMEOUT, check=False,
            creationflags=_CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        return StepResult(f"unload_model({model})", False, "timed out",
                          (time.monotonic() - t0) * 1000)
    return StepResult(
        f"unload_model({model})", r.returncode == 0,
        ((r.stdout if r.returncode == 0 else r.stderr) or "").strip()[:240],
        (time.monotonic() - t0) * 1000,
    )


def stop_lms_server() -> StepResult:
    t0 = time.monotonic()
    loc = find_lms()
    if loc.path is None:
        return StepResult("stop_lms_server", False, "lms.exe not found",
                          (time.monotonic() - t0) * 1000)
    try:
        r = subprocess.run(
            [str(loc.path), "server", "stop"],
            capture_output=True, text=True,
            timeout=_LMS_FAST_TIMEOUT, check=False,
            creationflags=_CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        return StepResult("stop_lms_server", False, "timed out",
                          (time.monotonic() - t0) * 1000)
    return StepResult(
        "stop_lms_server", r.returncode == 0,
        ((r.stdout if r.returncode == 0 else r.stderr) or "").strip()[:240],
        (time.monotonic() - t0) * 1000,
    )


# ---- internal helpers -------------------------------------------------------


def _server_reachable(base_url: str) -> bool:
    try:
        r = httpx.get(f"{base_url}/models", timeout=1.5)
        return r.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def _ps_query(script: str) -> str:
    """Run a small PowerShell expression and return stdout. On failure: ''."""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True,
            timeout=_TASK_OP_TIMEOUT, check=False,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return ""
    return r.stdout if r.returncode == 0 else ""

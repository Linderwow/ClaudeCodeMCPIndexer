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
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=4.0, check=False,
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


# LM Studio's `/api/v0/models` endpoint (their richer admin API, alongside
# the OpenAI-compatible /v1/models) returns BOTH the available model list
# AND each model's load state in ONE pure-HTTP call. By using it we avoid
# `lms ps` entirely -- the subprocess fans out into 7+ lms-cli RPC calls per
# invocation that flood LM Studio's logs at our 2s poll cadence.
#
# Schema we care about:
#   {"data": [
#     {"id": "...", "type": "embeddings"|"llm", "state": "loaded"|"not-loaded",
#      "quantization": "Q4_K_M", "max_context_length": 4096, ...}
#   ]}
#
# 3 s TTL is plenty fresh: model state only changes on load/unload events,
# which are user-initiated (Start all / Stop all / per-model unload buttons)
# and which we trigger ourselves -- so we know to invalidate the cache after.
_LMS_API_CACHE: tuple[float, dict[str, Any]] = (0.0, {})
_LMS_API_TTL_S = 3.0


def _lms_fetch_state(base_url: str) -> dict[str, Any]:
    """Hit /api/v0/models and return a parsed view, cached briefly. Falls back
    to a server_up=False shape on any error so callers can render gracefully.

    Pre-gated on _server_reachable() so when LM Studio is DOWN we exit in
    ~50ms instead of paying the httpx IPv6→IPv4 fallback timeout (~4s on
    Windows). Critical for the dashboard's 2s poll cadence — without this
    gate, `get_status` blocks past the cadence whenever LM Studio is off,
    making the UI feel unresponsive after Stop All.
    """
    global _LMS_API_CACHE
    now = time.monotonic()
    ts, cached = _LMS_API_CACHE
    if now - ts < _LMS_API_TTL_S and ts > 0:
        return cached

    parsed: dict[str, Any] = {"server_up": False, "available": [], "loaded": []}

    # Fast TCP probe before HTTP — see _server_reachable docstring.
    if not _server_reachable(base_url):
        _LMS_API_CACHE = (now, parsed)
        return parsed

    # /api/v0/models lives at the LM Studio HTTP server (same host:port as
    # /v1/...) but on the admin path. Strip "/v1" if base_url has it.
    api_base = base_url
    if api_base.endswith("/v1"):
        api_base = api_base[: -len("/v1")]
    elif api_base.endswith("/v1/"):
        api_base = api_base[: -len("/v1/")]
    api_base = api_base.rstrip("/")

    try:
        r = httpx.get(f"{api_base}/api/v0/models", timeout=2.0)
        if r.status_code == 200:
            parsed["server_up"] = True
            data = r.json().get("data", []) or []
            for m in data:
                mid = str(m.get("id", ""))
                if not mid:
                    continue
                parsed["available"].append(mid)
                if str(m.get("state", "")).lower() == "loaded":
                    parsed["loaded"].append({
                        "id":      mid,
                        "status":  "IDLE",  # /api/v0 doesn't expose IDLE/COMPUTING
                        "size_mb": 0.0,     # not returned by /api/v0
                        "context": str(m.get("loaded_context_length",
                                             m.get("max_context_length", ""))),
                        "ttl":     None,    # not exposed by /api/v0
                        "device":  m.get("type", ""),  # repurpose as type label
                        "quantization": m.get("quantization", ""),
                    })
    except (httpx.HTTPError, OSError):
        pass

    _LMS_API_CACHE = (now, parsed)
    return parsed


def _lms_status(settings: Settings) -> dict[str, Any]:
    base_url = settings.embedder.base_url.rstrip("/")
    state = _lms_fetch_state(base_url)
    return {
        "server_up": state["server_up"],
        "base_url":  base_url,
        "models_loaded":    state["loaded"],
        "models_available": state["available"],
        "configured": {
            "embedder": settings.embedder.model,
            "reranker": settings.reranker.model if settings.reranker.model else None,
            "reranker_kind": settings.reranker.kind,
        },
    }


def invalidate_lms_cache() -> None:
    """Drop the cached /api/v0/models snapshot so the next status poll
    refreshes immediately. Call this after operations that change load state
    (load_model / unload_model / unload_all_models / start_lms_server /
    stop_lms_server) so the UI reflects the new state on the very next
    poll instead of waiting up to TTL."""
    global _LMS_API_CACHE
    _LMS_API_CACHE = (0.0, {})


# Note: We deliberately do NOT call `lms ps` from get_status anymore. Each
# invocation fans into ~7 lms-cli RPC sub-calls that flood LM Studio's logs.
# `_parse_lms_ps` and `_parse_size_mb` are kept for tests + ops that touch
# the CLI directly (load/unload), but the dashboard's polling path now uses
# the lighter /api/v0/models HTTP endpoint via `_lms_fetch_state`.


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

    Phase 39: also clears the `data/.stopped` intent marker so the reaper
    resumes its normal auto-respawn-on-crash behavior.
    """
    res = CompositeResult()
    # Phase 39: clear stop intent BEFORE starting anything else, so the
    # reaper (or daily redeploy) doesn't see a transient state where
    # the watcher is alive but the marker still says "stay stopped".
    from code_rag.util.stop_marker import clear_intentionally_stopped
    cleared = clear_intentionally_stopped(settings.paths.data_dir,
                                          reason="dashboard.start_all")
    res.add(StepResult(
        "clear_stop_marker", cleared,
        "stop intent cleared" if cleared else "could not clear marker",
        duration_ms=0.0,
    ))

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
            invalidate_lms_cache()
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
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout_s, check=False,
            creationflags=_CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        return StepResult(f"load_model({model})", False,
                          f"timed out after {timeout_s:.0f}s",
                          (time.monotonic() - t0) * 1000)
    if r.returncode == 0:
        invalidate_lms_cache()
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

    Phase 39: also writes a `data/.stopped` intent marker that the
    Phase 36-C reaper and Phase 37-L daily redeploy task respect. With
    the marker in place, the watcher STAYS dead until either the user
    hits Start All (which deletes the marker) or the PC reboots (the
    autostart bootstrap deletes the marker on logon). Without this, the
    reaper would see `is_watcher_alive() == False` and respawn within
    10 minutes, defeating Stop All's intent.

    Pass `stop_lm_studio=False` to keep the server up (e.g. for the
    granular-stop case where the dashboard's per-card buttons did the work).
    """
    res = CompositeResult()
    # Phase 39: write the intent marker FIRST so any concurrent reaper
    # that's mid-cycle sees it before we kill the watcher (eliminating a
    # tiny race where the reaper could observe "watcher dead" before the
    # marker exists and respawn).
    from code_rag.util.stop_marker import mark_intentionally_stopped
    marked = mark_intentionally_stopped(settings.paths.data_dir,
                                        reason="dashboard.stop_all")
    res.add(StepResult(
        "set_stop_marker", marked,
        "stop intent recorded (reaper + redeploy will leave watcher alone)"
        if marked
        else "WARNING: marker write failed; reaper may respawn watcher",
        duration_ms=0.0,
    ))
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
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_LMS_FAST_TIMEOUT, check=False,
            creationflags=_CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        return StepResult("unload_all_models", False, "timed out",
                          (time.monotonic() - t0) * 1000)
    if r.returncode == 0:
        invalidate_lms_cache()
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
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_LMS_FAST_TIMEOUT, check=False,
            creationflags=_CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        return StepResult(f"unload_model({model})", False, "timed out",
                          (time.monotonic() - t0) * 1000)
    if r.returncode == 0:
        invalidate_lms_cache()
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
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_LMS_FAST_TIMEOUT, check=False,
            creationflags=_CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired:
        return StepResult("stop_lms_server", False, "timed out",
                          (time.monotonic() - t0) * 1000)
    if r.returncode == 0:
        invalidate_lms_cache()
    return StepResult(
        "stop_lms_server", r.returncode == 0,
        ((r.stdout if r.returncode == 0 else r.stderr) or "").strip()[:240],
        (time.monotonic() - t0) * 1000,
    )


# ---- internal helpers -------------------------------------------------------


def _server_reachable(base_url: str) -> bool:
    """True iff the LM Studio HTTP server is accepting connections.

    We do a raw TCP probe (IPv4) instead of an HTTP GET because:
      1. httpx routes `localhost` through getaddrinfo, which on Windows
         returns BOTH `::1` (IPv6) AND `127.0.0.1` (IPv4). httpx tries the
         IPv6 first, eats the full timeout on connection-refused, then
         falls back to IPv4 — paying the timeout TWICE per probe. With a
         1.5s timeout that's 3s, and `get_status` is supposed to fit
         under the dashboard's 2s poll cadence.
      2. We don't actually need the HTTP roundtrip to know the server's up;
         a TCP handshake is sufficient. /api/v0/models will provide the
         richer state when reachable, gated by this probe.
    Result: <50ms when down, <50ms when up.
    """
    import socket
    import urllib.parse
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    # Force IPv4 — `localhost` would otherwise try ::1 first.
    if host in ("localhost", "::1"):
        host = "127.0.0.1"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # 250ms is generous for loopback — when LM Studio's listening
            # the handshake completes in < 5ms; when it isn't, Windows
            # firewall sometimes silently drops instead of REFUSE-ing, so
            # we still want a budget. Net: 250ms when down, ~5ms when up.
            s.settimeout(0.25)
            return s.connect_ex((host, port)) == 0
    except OSError:
        return False


def _ps_query(script: str) -> str:
    """Run a small PowerShell expression and return stdout. On failure: ''."""
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=_TASK_OP_TIMEOUT, check=False,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (subprocess.TimeoutExpired, OSError, FileNotFoundError):
        return ""
    return r.stdout if r.returncode == 0 else ""

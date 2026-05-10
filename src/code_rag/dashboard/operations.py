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
# Phase 60-O (audit P3): bumped from 3.0 -> 5.0. Dashboard polls every 2s,
# so 3s TTL meant every other poll missed cache and paid the full
# /api/v0/models round-trip. 5s TTL is invalidated explicitly via
# invalidate_lms_cache() on user-initiated state changes (Start/Stop), so
# stale data isn't a concern. Cuts LMS HTTP traffic by ~60%.
_LMS_API_TTL_S = 5.0


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

    # Phase 60: support both LM Studio (/api/v0/models, richer admin info)
    # AND vLLM / TEI / sglang (/v1/models, OpenAI standard, no `state` field
    # because every model returned IS loaded). Try the rich endpoint first;
    # on 404 fall back to /v1/models. This makes the dashboard truthful about
    # an embedder server being up regardless of which backend the user runs.
    rich_ok = False
    try:
        r = httpx.get(f"{api_base}/api/v0/models", timeout=2.0)
        if r.status_code == 200:
            rich_ok = True
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

    # OpenAI-standard fallback: /v1/models. vLLM, TEI, sglang, etc all serve
    # this. No `state` — we treat every returned model as loaded (these
    # backends only list models actually in memory).
    if not rich_ok:
        try:
            r = httpx.get(f"{api_base}/v1/models", timeout=2.0)
            if r.status_code == 200:
                parsed["server_up"] = True
                data = r.json().get("data", []) or []
                for m in data:
                    mid = str(m.get("id", ""))
                    if not mid:
                        continue
                    parsed["available"].append(mid)
                    parsed["loaded"].append({
                        "id":      mid,
                        "status":  "IDLE",
                        "size_mb": 0.0,
                        "context": "",
                        "ttl":     None,
                        "device":  "",
                        "quantization": "",
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

    # Phase 60-A2: surface dense (chroma) vector count alongside the FTS
    # chunk count. During a reindex these diverge — chunker fills FTS fast,
    # embedder catches up later — and the divergence IS the user-visible
    # reindex-progress signal. `vectors / chunks` is the % done.
    vector_count: int | None = None
    try:
        # Read the chroma sqlite directly. Cheap (a single COUNT) and avoids
        # spinning up the chromadb client (which probes the embedder).
        import sqlite3
        chroma_db = settings.paths.data_dir / "chroma" / "chroma.sqlite3"
        if chroma_db.exists():
            conn = sqlite3.connect(chroma_db, isolation_level=None)
            conn.execute("PRAGMA query_only = 1")
            # Find the active collection sqlite table. Chroma stores per-
            # collection vector counts in `embeddings` keyed by collection_id;
            # the simplest cross-version count is just total rows in
            # `embeddings` for the matching collection segment, but for a
            # quick "is the index growing?" status the total count across
            # all collections is fine.
            row = conn.execute(
                "SELECT COUNT(*) FROM embeddings"
            ).fetchone()
            vector_count = int(row[0]) if row else 0
            conn.close()
    except (OSError, sqlite3.Error):
        pass

    # Reindex-in-progress detection: vectors lag chunks by more than 1%.
    reindex_pct: float | None = None
    if chunk_count and vector_count is not None and chunk_count > 0:
        reindex_pct = round(100.0 * vector_count / chunk_count, 1)

    # Phase 60-K: classify the reindex state so the dashboard can show
    # WHY catch-up is running. Three states:
    #   - "wipe_recovery": vectors=0 but chunks>0 (Phase 60-K detector
    #     fired). Shows up after an overnight chroma corruption reset.
    #   - "catching_up":   0 < vectors < chunks (regular catch-up).
    #   - "live":          vectors >= chunks (steady state).
    #
    # Phase 60-M: the original `vector_count == 0` test races -- vectors
    # transitions 0 -> N+ within seconds of the wipe-detector clearing
    # file_hashes, so a 5s dashboard poll almost never catches the zero
    # window. autostart_bootstrap now writes a sticky `.wipe_recovery`
    # marker when it fires. We read the marker and keep declaring
    # "wipe_recovery" until vectors climb back to >= 95% of fts (at which
    # point we delete the marker so it doesn't linger past the recovery).
    wipe_marker = settings.paths.data_dir / ".wipe_recovery"
    wipe_marker_active = wipe_marker.exists()
    # Phase 60-O (audit IPC#10): max-age safety on the wipe_recovery
    # marker. If a stalled reindex pinned the marker, we don't want it to
    # show "wipe recovery" forever -- after 24h, treat it as stale and
    # tear down.
    if wipe_marker_active:
        try:
            import time as _t
            age_s = _t.time() - wipe_marker.stat().st_mtime
            if age_s > 24 * 3600:
                wipe_marker.unlink(missing_ok=True)
                wipe_marker_active = False
        except OSError:
            pass
    state = None
    if chunk_count and vector_count is not None:
        # 95%-of-chunks recovery threshold: anything above and we declare
        # recovery complete and tear down the marker.
        recovered = (
            reindex_pct is not None and reindex_pct >= 95.0
        )
        if wipe_marker_active and not recovered:
            state = "wipe_recovery"
        elif vector_count == 0 and chunk_count > 100:
            # Defensive: marker missing (e.g. wiped manually) but state still
            # matches -- still paint wipe_recovery rather than catching_up.
            state = "wipe_recovery"
        elif reindex_pct is not None and reindex_pct < 99.0:
            state = "catching_up"
        else:
            state = "live"
        # Tear down the marker once we're past the recovery threshold so
        # subsequent polls return to the regular catching_up/live ladder.
        if wipe_marker_active and recovered:
            try:
                wipe_marker.unlink(missing_ok=True)
            except OSError:
                pass

    return {
        "present":         True,
        "chunks":          chunk_count,
        "vectors":         vector_count,        # Phase 60-A2
        "reindex_pct":     reindex_pct,         # Phase 60-A2: vectors/chunks
        "state":           state,                # Phase 60-K
        "embedder_kind":   meta.get("embedder_kind"),
        "embedder_model":  meta.get("embedder_model"),
        "embedder_dim":    meta.get("embedder_dim"),
        "schema_version": meta.get("schema_version"),
        "created_at":      meta.get("created_at"),
        "updated_at":      meta.get("updated_at"),
    }


# ---- start operations -------------------------------------------------------


def _budget_verdict_for_code_rag():
    """Phase 52: return a BudgetVerdict for the code-rag project given
    current resource state, or None if probes failed (treat-as-pass on
    no-data so we don't block on a flaky probe).
    """
    from code_rag.dashboard.resource_budget import (
        CurrentResources, can_start_project,
    )
    gpu = _gpu_status_via_nvidia_smi()
    if gpu is None:
        return None   # no GPU detected → don't block CPU-only setups

    # RAM via Win32_OperatingSystem.
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "$o=Get-CimInstance Win32_OperatingSystem; "
             "[PSCustomObject]@{tot=$o.TotalVisibleMemorySize;"
             "free=$o.FreePhysicalMemory} | ConvertTo-Json -Compress"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
            timeout=5.0, check=False,
            creationflags=_CREATE_NO_WINDOW,
        )
        import json as _j
        m = _j.loads(r.stdout) if r.returncode == 0 and r.stdout.strip() else {}
        ram_total_gb = float(m.get("tot", 0)) / 1024 / 1024
        ram_used_gb = (float(m.get("tot", 0)) - float(m.get("free", 0))) / 1024 / 1024
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return None

    cur = CurrentResources(
        ram_used_gb=ram_used_gb,
        ram_total_gb=ram_total_gb,
        vram_used_gb=float(gpu.get("vram_used_gb", 0)),
        vram_total_gb=float(gpu.get("vram_total_gb", 0)),
    )
    return can_start_project("code-rag", cur)


def start_all(settings: Settings, *, force: bool = False) -> CompositeResult:
    """Bring the whole stack up.

    Phase 60-G: post LM-Studio era. The embedder is now vLLM-in-WSL
    (port 8000) plus an optional rerank vLLM (port 8001). When the
    configured base_url points at :8000 we take the vLLM path:

        clear stop marker -> start vLLM-embed in WSL -> start vLLM-rerank
        (best-effort) -> wait for /v1/models on both -> start watcher.

    Legacy LM Studio path (base_url :1234) is preserved verbatim for
    rollback compatibility.

    Phase 47 -> 52: budget-aware refuse. Uses resource_budget.PROJECT_COSTS
    to refuse before OOM. Override with force=True (dashboard `?force=1`).
    """
    res = CompositeResult()

    # Phase 52: budget-aware safety gate.
    if not force:
        verdict = _budget_verdict_for_code_rag()
        if verdict is not None:
            if not verdict.ok:
                res.add(StepResult(
                    "budget_guard", False, verdict.suggestion,
                    duration_ms=0.0,
                ))
                return res
            res.add(StepResult(
                "budget_guard", True,
                f"OK: code-rag needs {verdict.cost_ram_gb:.1f} GB RAM + "
                f"{verdict.cost_vram_gb:.1f} GB VRAM; "
                f"{verdict.available_ram_gb:.1f} GB RAM + "
                f"{verdict.available_vram_gb:.1f} GB VRAM available",
                duration_ms=0.0,
            ))

    # Phase 39: clear stop intent BEFORE starting.
    from code_rag.util.stop_marker import clear_intentionally_stopped
    cleared = clear_intentionally_stopped(settings.paths.data_dir,
                                          reason="dashboard.start_all")
    res.add(StepResult(
        "clear_stop_marker", cleared,
        "stop intent cleared" if cleared else "could not clear marker",
        duration_ms=0.0,
    ))
    # Phase 60-I: also clear the auto-stop marker so a Start click
    # supersedes any pending idle-stop state.
    from code_rag.util.auto_stop_marker import clear_auto_stopped
    auto_cleared = clear_auto_stopped(settings.paths.data_dir,
                                      reason="dashboard.start_all")
    if not auto_cleared:
        res.add(StepResult(
            "clear_auto_stop_marker", False,
            "could not clear auto-stop marker (non-fatal)",
            duration_ms=0.0,
        ))

    # Phase 60-G: dispatch on base_url. vLLM is on :8000 by convention; LM
    # Studio on :1234. Anything else falls through to the LM Studio path
    # for backward compat (will fail cleanly if the binary isn't there).
    # Use getattr+default so test stubs without `base_url` keep working.
    # Audit fix: parse the URL properly so `http://localhost:18000/v1`
    # doesn't false-match the substring ":8000".
    from urllib.parse import urlparse
    base_url = getattr(settings.embedder, "base_url", "").rstrip("/")
    try:
        is_vllm = urlparse(base_url).port == 8000
    except (ValueError, AttributeError):
        is_vllm = False

    if is_vllm:
        res.add(_start_vllm_in_wsl(settings))
        if not res.steps[-1].ok:
            return res
    else:
        # Legacy LM Studio path.
        res.add(start_lms_server(settings))
        if not res.steps[-1].ok:
            return res
        res.add(load_model(settings.embedder.model, _LMS_LOAD_TIMEOUT))
        if settings.reranker.kind == "lm_chat" and settings.reranker.model:
            res.add(load_model(settings.reranker.model, _LMS_LOAD_TIMEOUT))

    res.add(start_watcher_task())
    return res


def _start_vllm_in_wsl(settings: Settings) -> StepResult:
    """Phase 60-G: launch vLLM-embed (port 8000) and vLLM-rerank (port
    8001) inside the Ubuntu WSL distro. Mirrors what
    scripts/resume_code_rag.ps1 does on the scheduled-task path so the
    dashboard Start button has parity with the cold-boot resume.

    Behavior:
      - If both endpoints already 200 on /v1/models, return OK ("already up").
      - Otherwise launch the missing one(s) via setsid+nohup so they
        survive the dashboard's subprocess context.
      - Wait up to 180 s per server for /v1/models to respond.
      - Rerank failure is best-effort (non-fatal): the cross-encoder
        rerank step in queries will degrade to no-op if its endpoint is
        down. Embed failure IS fatal because nothing else can run.

    Phase 60-N: takes the shared `data/vllm-launch.lock` to serialize
    against scripts/resume_code_rag.ps1. Without this, the dashboard's
    Start button could spawn a second pair of vLLM servers while the
    autostart bootstrap was racing the same launch -- duplicates burn
    ~10 GB VRAM each. Concurrent callers exit clean ("another launch in
    progress") instead of stacking spawns.
    """
    from code_rag.util.proc_hygiene import SingletonLock
    t0 = time.monotonic()

    launch_lock_path = settings.paths.data_dir / "vllm-launch.lock"
    launch_lock = SingletonLock(launch_lock_path).__enter__()
    if not launch_lock.acquired:
        return StepResult(
            "start_vllm_in_wsl", True,
            f"another vLLM launch in progress (vllm-launch.lock held by "
            f"live PID); skipping.",
            (time.monotonic() - t0) * 1000,
        )

    try:
        return _start_vllm_in_wsl_locked(settings, t0)
    finally:
        launch_lock.__exit__(None, None, None)


def _start_vllm_in_wsl_locked(settings: Settings, t0: float) -> StepResult:
    """Phase 60-N: actual launch routine, runs only once the
    vllm-launch.lock has been acquired. Split out so the lock-release
    finally has a single clean exit point.
    """

    def _server_up(port: int) -> bool:
        try:
            r = httpx.get(f"http://localhost:{port}/v1/models", timeout=2.0)
            return r.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    def _launch(script_name: str) -> None:
        # setsid + nohup + redirected fds so the server outlives the parent
        # WSL bash session. </dev/null prevents stdin EOF kills.
        # Audit-2 round-3 fix: append `disown; sleep 3` so the foreground
        # bash stays alive long enough for the backgrounded process to
        # actually detach. Without this, when invoked from PowerShell-
        # mediated wsl.exe (Task Scheduler context), the launched process
        # gets killed before it opens the redirect target — the launch
        # silently no-ops and downstream waits hit their full 180s timeout.
        cmd = [
            "wsl.exe", "-d", "Ubuntu", "-e", "bash", "-c",
            f"rm -f /tmp/{script_name}.log; "
            f"setsid nohup bash $HOME/bin/{script_name} "
            f"> /tmp/{script_name}.log 2>&1 < /dev/null & disown; sleep 3",
        ]
        subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, creationflags=_CREATE_NO_WINDOW,
            close_fds=True,
        )

    def _wait_for(port: int, timeout_s: float = 180.0) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if _server_up(port):
                return True
            time.sleep(2.0)
        return False

    # Audit fix: SEQUENTIAL launch (embed → wait until UP → rerank). Original
    # draft launched both in parallel; their CUDA-graph-capture VRAM spikes
    # overlap and embed (which requests more) crashes with "Free memory <
    # desired GPU memory utilization" because rerank already snatched its
    # share. Sequential start guarantees each server sees the full free
    # VRAM at its mem-util computation time.
    embed_already_up = _server_up(8000)
    if not embed_already_up:
        try:
            _launch("serve-qwen3-embed-8b.sh")
        except Exception as e:
            return StepResult(
                "start_vllm_in_wsl", False,
                f"failed to launch embed via wsl.exe: {e}",
                (time.monotonic() - t0) * 1000,
            )

    if not _wait_for(8000):
        return StepResult(
            "start_vllm_in_wsl", False,
            "vLLM-embed never came up on :8000 within 180s. Check "
            "/tmp/vllm-embed.log inside WSL.",
            (time.monotonic() - t0) * 1000,
        )

    # Now embed is settled — launch rerank (best-effort).
    rerank_already_up = _server_up(8001)
    if not rerank_already_up:
        try:
            _launch("serve-rerank-bge-m3.sh")
        except Exception:
            # Don't fail the whole start on rerank launch issues.
            pass

    rerank_up = _wait_for(8001, timeout_s=60.0)  # short wait — best-effort
    detail = (
        "vLLM-embed up on :8000"
        + (" (already running)" if embed_already_up else " (launched)")
        + "; vLLM-rerank "
        + ("up on :8001" if rerank_up else "DOWN (degraded; rerank will no-op)")
    )
    return StepResult(
        "start_vllm_in_wsl", True, detail,
        (time.monotonic() - t0) * 1000,
    )


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
    """Issue `lms load <model>`. Phase 45: idempotent fast-path skip
    when the model is already loaded with the correct Phase 33 ctx.

    Why this matters: LM Studio's `lms load <name>` is NOT idempotent
    when the model is already in any loaded state. Instead of being a
    no-op, LM Studio creates a `<name>:N` duplicate ALIAS that occupies
    its own VRAM (~3 GB for the 4B embedder). Two cascades observed:

      1. Dashboard "Start All" → ensure_lm_studio_ready had reported
         the model already loaded → start_all called load_model anyway
         to "make sure" → 103-second LM Studio timeout + duplicate :2.

      2. Watcher autostart_bootstrap → fast path's ctx-check returned
         False on a missing-from-`lms ps` HyDE model → slow path issued
         `lms load` for the embedder too → duplicate :2.

    Phase 45 fix: short-circuit when ctx matches Phase 33 prefs. If
    we don't have a pref OR the model isn't currently in `lms ps`,
    fall through to the actual `lms load` (preserves the load-on-cold
    path for first-boot).
    """
    t0 = time.monotonic()
    loc = find_lms()
    if loc.path is None:
        return StepResult(f"load_model({model})", False, "lms.exe not found",
                          (time.monotonic() - t0) * 1000)
    # Phase 45: skip the redundant `lms load` if the model is already
    # loaded with the correct Phase 33 ctx — `lms load` would create a
    # duplicate :N alias instead of being a no-op.
    from code_rag.lms_ctl import _LMS_LOAD_SETTINGS, loaded_context_length
    _, want_ctx = _LMS_LOAD_SETTINGS.get(model, (None, None))
    if want_ctx is not None:
        actual_ctx = loaded_context_length(loc.path, model)
        if actual_ctx == want_ctx:
            return StepResult(
                f"load_model({model})", True,
                f"already loaded with ctx={actual_ctx} — skipped redundant load",
                (time.monotonic() - t0) * 1000,
            )
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


def stop_all(
    settings: Settings,
    *,
    stop_lm_studio: bool = True,
    kill_mcp_servers: bool = True,
) -> CompositeResult:
    """Bring the stack down COMPLETELY.

    Stops watcher, unloads models, kills MCP servers, and stops the LM
    Studio server. The server has to go too — if it stays up, ANY request
    from a Claude Code MCP subprocess (or anything else pointing at
    /v1/embeddings) will JIT-reload the embedder, defeating the user's
    intent. The watcher autostart task is only stopped for THIS session —
    it'll fire again on next logon (the user's hands-off-on-reboot
    requirement).

    Phase 39: also writes a `data/.stopped` intent marker that the
    Phase 36-C reaper and Phase 37-L daily redeploy task respect. With
    the marker in place, the watcher STAYS dead until either the user
    hits Start All (which deletes the marker) or the PC reboots (the
    autostart bootstrap deletes the marker on logon). Without this, the
    reaper would see `is_watcher_alive() == False` and respawn within
    10 minutes, defeating Stop All's intent.

    Phase 39b: ALSO kills the `code-rag mcp` server processes Claude Code
    spawned. Each MCP server holds a cross-encoder loaded into CUDA
    (~2 GB VRAM); leaving them alive means Stop All frees LM Studio's
    VRAM but the GPU still shows ~10-12 GB resident. Killing them is
    safe: Claude Code reconnects on next request, paying a 5-10s
    cross-encoder reload. Pass `kill_mcp_servers=False` to preserve
    the older "soft stop" semantics (Stop All without MCP kill).

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
    # Phase 60-A4: also kill any in-flight `code_rag index` bulk reindexer.
    # Pre-Phase-60 the Stop button left these running because they weren't
    # classified by proc_hygiene. From the user's POV the stack appeared
    # stopped while a 4-hour reindex kept hammering the GPU.
    res.add(_kill_indexer_processes())
    if kill_mcp_servers:
        res.add(_kill_mcp_servers())
    # Phase 60-A4: stop the WSL vLLM servers (embed on :8000, rerank on :8001)
    # since those replaced LM Studio as the embedder. Without this, Stop All
    # left vLLM holding ~14 GB VRAM and serving requests indefinitely.
    res.add(_kill_vllm_in_wsl())
    res.add(unload_all_models())
    if stop_lm_studio:
        res.add(stop_lms_server())
    return res


def _kill_indexer_processes() -> StepResult:
    """Phase 60-A4: kill every running `code_rag index` (bulk one-shot).

    The watcher (auto-incremental) is handled by stop_watcher_task; the MCP
    servers by _kill_mcp_servers. This helper covers the third type — the
    operator-launched bulk reindex. Without this, Stop All would leave a
    multi-hour `code_rag index` chewing through the codebase.
    """
    t0 = time.monotonic()
    try:
        from code_rag.util.proc_hygiene import kill_pid, list_code_rag_processes
    except Exception as e:  # pragma: no cover — defensive
        return StepResult(
            "kill_indexer", False, f"helper import failed: {e}",
            (time.monotonic() - t0) * 1000,
        )
    procs = [p for p in list_code_rag_processes() if p.kind == "index"]
    if not procs:
        return StepResult(
            "kill_indexer", True, "no indexer running",
            (time.monotonic() - t0) * 1000,
        )
    killed = 0
    for p in procs:
        try:
            # Phase 60-A4 audit fix: kill_pid takes only `pid`, no timeout_s
            # kwarg. Original draft had `kill_pid(p.pid, timeout_s=5.0)` which
            # raised TypeError (caught silently below) and never actually
            # killed anything — the very bug Phase 60-A4 was meant to fix.
            if kill_pid(p.pid):
                killed += 1
        except Exception:
            pass
    return StepResult(
        "kill_indexer", killed > 0,
        f"killed {killed}/{len(procs)} indexer process(es)",
        (time.monotonic() - t0) * 1000,
    )


def _kill_vllm_in_wsl() -> StepResult:
    """Phase 60-A4: stop vLLM servers running inside WSL.

    Phase 60 swapped the embedder from LM Studio (port 1234) to vLLM running
    inside Ubuntu WSL on port 8000 (embed) and 8001 (rerank). The classic
    `stop_lms_server` only knew how to stop LM Studio on Windows. Without
    this helper, Stop All left ~14 GB of VRAM resident in WSL even after
    "stopping" the stack.

    Uses `wsl.exe -d Ubuntu -e bash -c 'pkill -f ...'` which is best-effort:
    if WSL itself is down or vLLM isn't running, returns OK with a "nothing
    to stop" detail.
    """
    t0 = time.monotonic()
    try:
        # pkill -f matches against the full command line — `vllm serve` is
        # distinctive enough that we won't accidentally kill other Python
        # processes. SIGTERM first, then SIGKILL on stragglers.
        # Audit fix: also reap `VLLM::EngineCore` worker processes. vLLM
        # multiprocessing renames its workers, so they don't match
        # `vllm serve` — they survive a naive pkill and keep holding ~7-21
        # GB VRAM. `EngineCore` matches them by their renamed argv[0].
        cmd = [
            "wsl.exe", "-d", "Ubuntu", "-e", "bash", "-c",
            # vLLM forks several distinctly-named workers. Reap all of them
            # so VRAM is fully freed:
            #   - 'vllm serve'  : the API server parent
            #   - 'EngineCore'  : the renamed CUDA worker (holds the model)
            #   - 'multiprocessing.resource_tracker' : leftover from vLLM
            #     mp pool, can survive even after EngineCore dies
            #   - 'multiprocessing.spawn'            : same family
            "pkill -TERM -f 'vllm serve' 2>/dev/null; "
            "pkill -TERM -f 'EngineCore' 2>/dev/null; "
            "pkill -TERM -f 'multiprocessing.resource_tracker' 2>/dev/null; "
            "pkill -TERM -f 'multiprocessing.spawn' 2>/dev/null; "
            "sleep 1; "
            "pkill -KILL -f 'vllm serve' 2>/dev/null; "
            "pkill -KILL -f 'EngineCore' 2>/dev/null; "
            "pkill -KILL -f 'multiprocessing.resource_tracker' 2>/dev/null; "
            "pkill -KILL -f 'multiprocessing.spawn' 2>/dev/null; "
            "echo done",
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15.0,
            creationflags=_CREATE_NO_WINDOW,
        )
        ms = (time.monotonic() - t0) * 1000
        if proc.returncode == 0:
            return StepResult(
                "kill_vllm_wsl", True, "vLLM servers in WSL stopped (or were already stopped)",
                ms,
            )
        return StepResult(
            "kill_vllm_wsl", False,
            f"wsl pkill returned rc={proc.returncode}: {(proc.stderr or '')[:200]}",
            ms,
        )
    except subprocess.TimeoutExpired:
        return StepResult(
            "kill_vllm_wsl", False,
            "wsl pkill timed out (WSL may be hung; try Restart-Service LxssManager)",
            (time.monotonic() - t0) * 1000,
        )
    except Exception as e:  # pragma: no cover — defensive
        return StepResult(
            "kill_vllm_wsl", False, f"unexpected error: {e}",
            (time.monotonic() - t0) * 1000,
        )


def _kill_mcp_servers() -> StepResult:
    """Phase 39b: kill every `code-rag mcp` server process.

    These are spawned by Claude Code (one per workspace) and each loads
    the cross-encoder into CUDA. Stop All used to leave them alive
    "to preserve sessions", but the consequence is ~10-12 GB of VRAM
    sitting resident even after the user clicked Stop. Claude Code
    naturally respawns its MCP child on the next request with a fresh
    cross-encoder load (5-10s), so killing them here just delays the
    next request, not the session itself.

    Returns a StepResult counting how many we killed. If zero are alive,
    returns OK with a "no MCP servers running" detail.
    """
    t0 = time.monotonic()
    try:
        from code_rag.util.proc_hygiene import kill_pid, list_code_rag_processes
    except Exception as e:  # pragma: no cover — defensive
        return StepResult(
            "kill_mcp_servers", False, f"helper import failed: {e}",
            (time.monotonic() - t0) * 1000,
        )
    procs = [p for p in list_code_rag_processes() if p.kind == "mcp"]
    if not procs:
        return StepResult(
            "kill_mcp_servers", True, "no MCP servers running",
            (time.monotonic() - t0) * 1000,
        )
    killed = 0
    for p in procs:
        try:
            kill_pid(p.pid)
            killed += 1
        except Exception:
            # Best-effort kill: a single bad PID shouldn't abort the whole pass.
            continue
    detail = f"killed {killed}/{len(procs)} MCP server(s)"
    return StepResult(
        "kill_mcp_servers", killed == len(procs), detail,
        (time.monotonic() - t0) * 1000,
    )


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

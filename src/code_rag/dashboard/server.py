"""Local web dashboard.

Endpoints (localhost-only by default):
    GET  /                          static index.html
    GET  /static/*                  static assets (JS/CSS)
    GET  /api/status                composite status snapshot
    POST /api/start/all             start LM Studio + load models + start watcher
    POST /api/stop/all              stop watcher + unload models (LM Studio stays)
    POST /api/start/lms             start LM Studio server only
    POST /api/stop/lms              stop LM Studio server (kills models)
    POST /api/start/watcher         start the Task Scheduler watcher only
    POST /api/stop/watcher          stop the Task Scheduler watcher only
    POST /api/models/load           {"model": "<id>"}  load one model
    POST /api/models/unload         {"model": "<id>"}  unload one model (or "" for all)

Security: binds to 127.0.0.1. No auth — same-machine trust. Don't expose
to the LAN.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from code_rag.config import Settings, load_settings
from code_rag.dashboard import operations as ops
from code_rag.logging import get

log = get(__name__)


_STATIC_DIR = Path(__file__).resolve().parent / "static"


def _settings_or_500() -> Settings | JSONResponse:
    try:
        return load_settings()
    except Exception as e:
        log.exception("dashboard.settings_load_failed")
        return JSONResponse(
            {"error": f"failed to load config: {type(e).__name__}: {e}"},
            status_code=500,
        )


# ---- handlers ---------------------------------------------------------------


async def root(_req: Request) -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


async def status(_req: Request) -> JSONResponse:
    s = _settings_or_500()
    if isinstance(s, JSONResponse):
        return s
    return JSONResponse(ops.get_status(s))


async def health(_req: Request) -> JSONResponse:
    """Phase 36-E: real health probe (not just "process running").

    Returns a structured score with per-component status:

        {
          "overall":  "ok" | "degraded" | "critical"
          "checks": {
              "lm_studio":         {"ok": bool, "detail": "..."}
              "watcher_alive":     {"ok": bool, "detail": "pid=N age_min=M"}
              "watcher_heartbeat": {"ok": bool, "detail": "age_s=N"}
              "chroma":            {"ok": bool, "detail": "count=N elapsed_ms=M"}
              "vram":              {"ok": bool, "detail": "X / Y GB (Z%)"}
              "lms_duplicates":    {"ok": bool, "detail": "no aliases" | ":N alias..."}
          }
        }

    Each check is fast (<2s). Use this from the dashboard for true
    yellow/red badges instead of just "process running" green.

    "critical" iff any of: lm_studio down, watcher dead, chroma stuck.
    "degraded" iff: heartbeat stale, vram > 90%, duplicate aliases.
    """
    s = _settings_or_500()
    if isinstance(s, JSONResponse):
        return s

    from concurrent.futures import ThreadPoolExecutor

    from code_rag.chroma_watchdog import probe_chroma
    from code_rag.util.proc_hygiene import (
        is_watcher_alive,
        list_code_rag_processes,
        watcher_heartbeat_age_s,
    )

    checks: dict[str, dict[str, Any]] = {}

    # --- LM Studio ---
    base = s.embedder.base_url
    try:
        r = ops._lms_status(s)
        lms_ok = bool(r.get("server_up"))
        loaded = [m["id"] for m in r.get("models_loaded", [])]
        checks["lm_studio"] = {
            "ok": lms_ok,
            "detail": (f"up; {len(loaded)} model(s) loaded" if lms_ok
                       else f"down at {base}"),
        }
    except Exception as e:
        checks["lm_studio"] = {"ok": False, "detail": f"probe error: {e}"}

    # --- Watcher process + heartbeat ---
    procs = list_code_rag_processes()
    watcher_procs = [p for p in procs if p.kind in ("watcher", "watch_cli")]
    alive = bool(watcher_procs)
    checks["watcher_alive"] = {
        "ok": alive,
        "detail": (
            f"pid={watcher_procs[0].pid}" if alive else "no watcher process"
        ),
    }

    hb_age = watcher_heartbeat_age_s(s.paths.data_dir / "watcher.heartbeat")
    if hb_age is None:
        hb_ok = False
        hb_detail = "no heartbeat file"
    elif hb_age > 120:
        hb_ok = False
        hb_detail = f"stale ({hb_age:.0f}s)"
    else:
        hb_ok = True
        hb_detail = f"fresh ({hb_age:.0f}s)"
    checks["watcher_heartbeat"] = {"ok": hb_ok, "detail": hb_detail}

    # --- Chroma probe (subprocess-isolated, 5s timeout for quick health) ---
    def _chroma_probe() -> dict[str, Any]:
        return probe_chroma(s.chroma_dir, timeout_s=5.0)

    with ThreadPoolExecutor(max_workers=1) as ex:
        chroma_res = ex.submit(_chroma_probe).result()
    if chroma_res.get("ok"):
        total = sum(chroma_res.get("counts", {}).values())
        checks["chroma"] = {
            "ok": True,
            "detail": f"count={total} probe_ms={chroma_res.get('elapsed_s', 0)*1000:.0f}",
        }
    else:
        checks["chroma"] = {
            "ok": False,
            "detail": ("DEADLOCKED — run `code-rag chroma-heal --kill`"
                       if chroma_res.get("timed_out")
                       else f"error: {chroma_res.get('error', 'unknown')}"),
        }

    # --- VRAM pressure ---
    gpu = ops._gpu_status_via_nvidia_smi()
    if gpu is None:
        checks["vram"] = {"ok": True, "detail": "no GPU detected"}
    else:
        used, total = gpu.get("vram_used_gb", 0), gpu.get("vram_total_gb", 1)
        pct = (used / max(total, 1)) * 100
        checks["vram"] = {
            "ok": pct < 90,
            "detail": f"{used:.1f} / {total:.1f} GB ({pct:.0f}%)",
        }

    # --- LM Studio duplicate aliases ---
    try:
        loaded_ids = [m["id"] for m in r.get("models_loaded", [])]
        dup_aliases = [m for m in loaded_ids if ":" in m and m.split(":")[1].isdigit()]
        checks["lms_duplicates"] = {
            "ok": not dup_aliases,
            "detail": (f"{len(dup_aliases)} alias(es): {dup_aliases}"
                       if dup_aliases else "none"),
        }
    except Exception:
        checks["lms_duplicates"] = {"ok": True, "detail": "n/a"}

    # --- Overall ---
    critical_keys = {"lm_studio", "watcher_alive", "chroma"}
    critical_fail = any(not checks[k]["ok"] for k in critical_keys if k in checks)
    any_fail = any(not v["ok"] for v in checks.values())
    overall = "critical" if critical_fail else ("degraded" if any_fail else "ok")

    return JSONResponse({"overall": overall, "checks": checks})


async def start_all(_req: Request) -> JSONResponse:
    s = _settings_or_500()
    if isinstance(s, JSONResponse):
        return s
    return JSONResponse(ops.start_all(s).to_dict())


async def stop_all(req: Request) -> JSONResponse:
    s = _settings_or_500()
    if isinstance(s, JSONResponse):
        return s
    body = await _maybe_json(req)
    stop_lms = bool(body.get("stop_lm_studio", False))
    return JSONResponse(ops.stop_all(s, stop_lm_studio=stop_lms).to_dict())


async def start_lms(_req: Request) -> JSONResponse:
    s = _settings_or_500()
    if isinstance(s, JSONResponse):
        return s
    return JSONResponse(ops.start_lms_server(s).to_dict())


async def stop_lms(_req: Request) -> JSONResponse:
    return JSONResponse(ops.stop_lms_server().to_dict())


async def start_watcher(_req: Request) -> JSONResponse:
    return JSONResponse(ops.start_watcher_task().to_dict())


async def stop_watcher(_req: Request) -> JSONResponse:
    return JSONResponse(ops.stop_watcher_task().to_dict())


async def load_model(req: Request) -> JSONResponse:
    body = await _maybe_json(req)
    model = str(body.get("model", "")).strip()
    if not model:
        return JSONResponse({"error": "missing 'model' in body"}, status_code=400)
    return JSONResponse(ops.load_model(model, ops._LMS_LOAD_TIMEOUT).to_dict())


async def unload_model(req: Request) -> JSONResponse:
    body = await _maybe_json(req)
    model = str(body.get("model", "")).strip()
    result = ops.unload_all_models() if not model else ops.unload_model(model)
    return JSONResponse(result.to_dict())


async def _maybe_json(req: Request) -> dict[str, Any]:
    try:
        b = await req.body()
        if not b:
            return {}
        import json as _j
        parsed = _j.loads(b)
    except (ValueError, UnicodeDecodeError):
        return {}
    # Be defensive: a non-object body (e.g. a list) shouldn't crash the handler.
    return parsed if isinstance(parsed, dict) else {}


# ---- app factory ------------------------------------------------------------


class _NoCacheStaticMiddleware(BaseHTTPMiddleware):
    """Serve / and /static/* with no-cache so the browser always fetches the
    latest JS/CSS after a code-rag update. The dashboard payload is tiny
    (~10 KB total); the cost of skipping cache is negligible. Without this
    Chrome happily serves the stale app.js for hours after we ship a fix."""

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        resp: Response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            resp.headers["Pragma"] = "no-cache"
            resp.headers["Expires"] = "0"
        return resp


def build_app() -> Starlette:
    routes = [
        Route("/",                       endpoint=root,            methods=["GET"]),
        Route("/api/status",             endpoint=status,          methods=["GET"]),
        Route("/api/health",             endpoint=health,          methods=["GET"]),
        Route("/api/start/all",          endpoint=start_all,       methods=["POST"]),
        Route("/api/stop/all",           endpoint=stop_all,        methods=["POST"]),
        Route("/api/start/lms",          endpoint=start_lms,       methods=["POST"]),
        Route("/api/stop/lms",           endpoint=stop_lms,        methods=["POST"]),
        Route("/api/start/watcher",      endpoint=start_watcher,   methods=["POST"]),
        Route("/api/stop/watcher",       endpoint=stop_watcher,    methods=["POST"]),
        Route("/api/models/load",        endpoint=load_model,      methods=["POST"]),
        Route("/api/models/unload",      endpoint=unload_model,    methods=["POST"]),
        Mount("/static", app=StaticFiles(directory=str(_STATIC_DIR)), name="static"),
    ]
    middleware = [
        # Same-origin only by design — but allow the dashboard's own static
        # assets to fetch /api/* without browser CORS friction.
        Middleware(
            CORSMiddleware,
            allow_origins=["http://127.0.0.1", "http://localhost",
                           "http://127.0.0.1:7321", "http://localhost:7321"],
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        ),
        Middleware(_NoCacheStaticMiddleware),
    ]
    return Starlette(routes=routes, middleware=middleware)


def serve(host: str = "127.0.0.1", port: int = 7321, *, open_browser: bool = True) -> None:
    """Block in the foreground serving the dashboard."""
    import uvicorn
    if open_browser:
        import threading
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(f"http://{host}:{port}/")).start()
    uvicorn.run(build_app(), host=host, port=port, log_level="warning")

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
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
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


def build_app() -> Starlette:
    routes = [
        Route("/",                       endpoint=root,            methods=["GET"]),
        Route("/api/status",             endpoint=status,          methods=["GET"]),
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

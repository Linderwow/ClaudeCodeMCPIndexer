"""LM Studio CLI control — find `lms.exe`, start the local server, load a model,
and wait for `/v1/models` to report readiness.

Used by the autostart bootstrap so the whole stack comes up at logon without
a human touching LM Studio. All operations are idempotent: safe to call even
when the server is already running with the model loaded.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

from code_rag.logging import get

log = get(__name__)

# On Windows, `CREATE_NO_WINDOW` keeps subprocess.Popen from flashing a console.
# On other OSes this flag doesn't exist; we fall back to 0 (no extra flags).
_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

# Places LM Studio typically drops its CLI shim on Windows. We check in order
# of preference — `.lmstudio\bin\lms.exe` is the canonical shim that `lms
# bootstrap` adds to PATH, so it's tried first. The in-app paths are fallbacks
# for installs where bootstrap was never run.
_WIN_LMS_CANDIDATES: tuple[str, ...] = (
    r"%USERPROFILE%\.lmstudio\bin\lms.exe",
    r"%LOCALAPPDATA%\Programs\LM Studio\resources\app\.webpack\lms.exe",
    r"%LOCALAPPDATA%\Programs\LM Studio\resources\app\.webpack\main\bin\lms.exe",
    r"%LOCALAPPDATA%\LM Studio\resources\app\.webpack\main\bin\lms.exe",
)


@dataclass(frozen=True)
class LmsLocation:
    """Resolved lms CLI path + where we found it. `None` = not installed."""

    path: Path | None
    found_via: str     # "path" | "candidate" | "none"


def find_lms() -> LmsLocation:
    """Locate the `lms` CLI. Checks $PATH first, then common install roots."""
    import shutil
    on_path = shutil.which("lms")
    if on_path:
        return LmsLocation(Path(on_path), "path")
    if sys.platform == "win32":
        for raw in _WIN_LMS_CANDIDATES:
            p = Path(os.path.expandvars(raw))
            if p.exists():
                return LmsLocation(p, "candidate")
    return LmsLocation(None, "none")


# ---- server health ---------------------------------------------------------


def server_is_up(base_url: str, timeout_s: float = 2.0) -> bool:
    """Cheap reachability probe. True iff GET /v1/models returns 200."""
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout_s)
        return r.status_code == 200
    except (httpx.HTTPError, OSError):
        return False


def model_is_loaded(base_url: str, model: str, timeout_s: float = 5.0) -> bool:
    """True iff the given model id appears in the /v1/models list."""
    try:
        r = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout_s)
        if r.status_code != 200:
            return False
        data = r.json()
        ids = {m.get("id") for m in data.get("data", [])}
        return model in ids
    except (httpx.HTTPError, OSError, ValueError):
        return False


def wait_until_ready(base_url: str, model: str, *, timeout_s: float = 120.0) -> bool:
    """Block until the model is loaded or timeout elapses. Returns True on success."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if model_is_loaded(base_url, model):
            return True
        time.sleep(2.0)
    return False


# ---- lifecycle control -----------------------------------------------------


def start_server(lms: Path) -> subprocess.Popen[bytes]:
    """Launch `lms server start` as a background, windowless process.

    Returns the Popen handle so callers can attach a log if they want to, but
    in the autostart flow we deliberately let it orphan so the task scheduler
    entry can exit while the server keeps running.
    """
    return subprocess.Popen(
        [str(lms), "server", "start"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
        creationflags=_CREATE_NO_WINDOW,
        close_fds=True,
    )


def load_model(lms: Path, model: str, *, timeout_s: float = 300.0) -> subprocess.CompletedProcess[str]:
    """Invoke `lms load <model>` synchronously.

    Idempotent: if the model is already loaded, `lms load` exits ~instantly
    with a benign message. Large models can take minutes on first load, so
    the default timeout is generous.
    """
    return subprocess.run(
        [str(lms), "load", model],
        capture_output=True,
        text=True,
        # Decode lms.exe output as UTF-8; cp1252 (Windows default) chokes on
        # bytes lms emits for progress bars and unicode in model names. The
        # `errors="replace"` keeps the reader thread from raising and silently
        # spamming dashboard.log.
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
        creationflags=_CREATE_NO_WINDOW,
        check=False,
    )


# ---- high-level orchestration ----------------------------------------------


@dataclass
class BootstrapResult:
    ok: bool
    steps: list[str]                   # human-readable "did X" log lines
    error: str | None = None


def ensure_lm_studio_ready(
    base_url: str,
    embedder_model: str,
    extra_models: tuple[str, ...] = (),
    *,
    ready_timeout_s: float = 120.0,
) -> BootstrapResult:
    """Ensure LM Studio is running with `embedder_model` (and any `extra_models`) loaded.

    Steps (each idempotent):
      1. If the server already reports `embedder_model` ready → return immediately.
      2. Locate `lms`. If not installed, return a failure with actionable message.
      3. If the server isn't reachable, `lms server start` and wait.
      4. `lms load <embedder_model>` and wait for /v1/models to reflect it.
      5. Best-effort load of any extra models; missing ones don't fail the bootstrap.
    """
    steps: list[str] = []

    # Fast path: everything the caller asked for is already loaded.
    if model_is_loaded(base_url, embedder_model) and all(
        model_is_loaded(base_url, m) for m in extra_models
    ):
        steps.append(f"already-ready: {embedder_model}")
        for m in extra_models:
            steps.append(f"already-ready: {m}")
        return BootstrapResult(ok=True, steps=steps)

    loc = find_lms()
    if loc.path is None:
        return BootstrapResult(
            ok=False, steps=steps,
            error=("lms CLI not found. Install LM Studio, then run "
                   "`lms bootstrap` once so the CLI is on PATH."),
        )
    steps.append(f"lms found at {loc.path} (via {loc.found_via})")

    if not server_is_up(base_url):
        start_server(loc.path)
        steps.append("started LM Studio server")
        # Give the server a few seconds to bind its port before we hammer it.
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not server_is_up(base_url):
            time.sleep(0.5)
        if not server_is_up(base_url):
            return BootstrapResult(
                ok=False, steps=steps,
                error="LM Studio server failed to come up within 20s",
            )
        steps.append("server reachable")
    else:
        steps.append("server already running")

    r = load_model(loc.path, embedder_model)
    if r.returncode != 0:
        # Not automatically fatal — maybe the model IS loaded and lms just reported
        # a benign conflict. Let the readiness probe decide.
        log.warning("lms.load.nonzero", model=embedder_model,
                    rc=r.returncode, stderr=r.stderr[:200])
    steps.append(f"issued lms load {embedder_model}")

    if not wait_until_ready(base_url, embedder_model, timeout_s=ready_timeout_s):
        return BootstrapResult(
            ok=False, steps=steps,
            error=f"{embedder_model} did not become ready within {ready_timeout_s:.0f}s",
        )
    steps.append(f"{embedder_model} ready")

    for m in extra_models:
        if model_is_loaded(base_url, m):
            steps.append(f"already-ready: {m}")
            continue
        # Extras are best-effort. Tight timeout + swallow TimeoutExpired so a
        # missing/slow reranker never kills the whole bootstrap (the watcher
        # gracefully falls back to no-op rerank at query time).
        try:
            r = load_model(loc.path, m, timeout_s=60.0)
        except subprocess.TimeoutExpired:
            steps.append(f"{m} load timed out (skipped, non-fatal)")
            continue
        except Exception as e:  # pragma: no cover — defensive
            steps.append(f"{m} load errored: {type(e).__name__} (skipped)")
            continue
        if r.returncode == 0 and wait_until_ready(base_url, m, timeout_s=30.0):
            steps.append(f"{m} ready")
        else:
            steps.append(f"{m} load skipped (non-fatal)")

    return BootstrapResult(ok=True, steps=steps)

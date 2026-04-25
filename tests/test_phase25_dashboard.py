"""Phase 25: dashboard operations.

We exercise the pure operations module with a mock subprocess + httpx layer so
no LM Studio process is needed. The Starlette layer is a thin pass-through;
endpoint smoke is a separate live test.
"""
from __future__ import annotations

import subprocess
import types
from pathlib import Path
from typing import Any

import pytest

from code_rag import lms_ctl
from code_rag.config import load_settings
from code_rag.dashboard import operations as ops
from code_rag.dashboard.operations import (
    StepResult,
    _parse_lms_ps,
    _parse_size_mb,
    load_model,
    start_lms_server,
    stop_lms_server,
    stop_watcher_task,
    unload_all_models,
    unload_model,
)
from code_rag.lms_ctl import LmsLocation

# ---- helpers ---------------------------------------------------------------


def _make_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "repo"
    root.mkdir()
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f"""
[paths]
roots    = ["{root.as_posix()}"]
data_dir = "{(tmp_path / 'data').as_posix()}"
log_dir  = "{(tmp_path / 'logs').as_posix()}"

[embedder]
kind     = "lm_studio"
base_url = "http://localhost:1234/v1"
model    = "text-embedding-qwen3-embedding-4b"

[reranker]
kind     = "lm_chat"
base_url = "http://localhost:1234/v1"
model    = "qwen/qwen3-1.7b"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODE_RAG_CONFIG", str(cfg))
    return load_settings()


class _SubprocessRecorder:
    """Drop-in for subprocess.run that returns canned outputs and records calls."""
    def __init__(self, programmed: dict[str, types.SimpleNamespace]) -> None:
        self._programmed = programmed
        self.calls: list[list[str]] = []

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))
        # Find the first programmed key that appears anywhere in the args list.
        for key, ret in self._programmed.items():
            if any(key in str(a) for a in args):
                return ret
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")


def _stub_find_lms(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    fake = tmp_path / "lms.exe"
    fake.write_bytes(b"")
    monkeypatch.setattr(lms_ctl, "find_lms",
                        lambda: LmsLocation(fake, "path"))
    monkeypatch.setattr(ops, "find_lms",
                        lambda: LmsLocation(fake, "path"))
    return fake


# ---- pure parsers ----------------------------------------------------------


def test_parse_size_mb_handles_units() -> None:
    assert _parse_size_mb("2.41 GB") == pytest.approx(2.41 * 1024)
    assert _parse_size_mb("512 MB")  == pytest.approx(512)
    assert _parse_size_mb("1024 KB") == pytest.approx(1)
    assert _parse_size_mb("nonsense") == 0.0
    assert _parse_size_mb("") == 0.0


def test_parse_lms_ps_real_output() -> None:
    """Parse the actual `lms ps` output we observed earlier."""
    text = """
You have 2 models loaded.

IDENTIFIER                           MODEL                                STATUS    SIZE       CONTEXT    PARALLEL    DEVICE    TTL
qwen/qwen3-1.7b                      qwen/qwen3-1.7b                      IDLE      1.67 GB    32768      4           Local     60m / 1h
text-embedding-qwen3-embedding-4b    text-embedding-qwen3-embedding-4b    IDLE      2.50 GB    4096       -           Local
"""
    rows = _parse_lms_ps(text)
    assert len(rows) == 2
    assert rows[0]["id"] == "qwen/qwen3-1.7b"
    assert rows[0]["status"] == "IDLE"
    assert rows[0]["size_mb"] == pytest.approx(1.67 * 1024)
    assert rows[0]["ttl"] == "60m / 1h"
    assert rows[1]["id"] == "text-embedding-qwen3-embedding-4b"
    assert rows[1]["ttl"] is None  # the dash means no TTL


def test_parse_lms_ps_empty_or_missing_header() -> None:
    assert _parse_lms_ps("") == []
    assert _parse_lms_ps("nothing useful here") == []


# ---- start_lms_server ------------------------------------------------------


def test_start_lms_server_fast_path_when_already_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If /v1/models already responds 200, we should NOT spawn lms."""
    s = _make_settings(tmp_path, monkeypatch)
    monkeypatch.setattr(ops, "_server_reachable", lambda url: True)

    spawned: list[Any] = []
    monkeypatch.setattr(subprocess, "Popen",
                        lambda *a, **kw: spawned.append((a, kw)) or types.SimpleNamespace(pid=1))

    r = start_lms_server(s)
    assert r.ok
    assert "already running" in r.detail
    assert spawned == []


def test_start_lms_server_no_lms_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _make_settings(tmp_path, monkeypatch)
    monkeypatch.setattr(ops, "_server_reachable", lambda url: False)
    monkeypatch.setattr(ops, "find_lms",
                        lambda: LmsLocation(None, "none"))

    r = start_lms_server(s)
    assert not r.ok
    assert "lms.exe not found" in r.detail


def test_start_lms_server_spawns_then_waits_for_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server not up initially; after Popen we flip _server_reachable to True
    on the next call so the waiter exits cleanly."""
    s = _make_settings(tmp_path, monkeypatch)
    _stub_find_lms(monkeypatch, tmp_path)

    state = {"up": False}
    monkeypatch.setattr(ops, "_server_reachable", lambda url: state["up"])
    spawned: list[Any] = []

    def fake_popen(*args, **kwargs):
        spawned.append((args, kwargs))
        state["up"] = True   # next reachability probe succeeds
        return types.SimpleNamespace(pid=42)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(ops.time, "sleep", lambda _x: None)

    r = start_lms_server(s)
    assert r.ok, r
    assert "ready" in r.detail
    assert len(spawned) == 1


# ---- load_model ------------------------------------------------------------


def test_load_model_returns_ok_on_zero_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_find_lms(monkeypatch, tmp_path)
    rec = _SubprocessRecorder({"load": types.SimpleNamespace(
        returncode=0, stdout="loaded\n", stderr="",
    )})
    monkeypatch.setattr(subprocess, "run", rec)

    r = load_model("qwen/qwen3-1.7b", timeout_s=5.0)
    assert r.ok
    assert r.detail == "loaded"
    # The subprocess we ran must include `load <model>` in argv.
    assert any("load" in c and "qwen/qwen3-1.7b" in c for c in rec.calls)


def test_load_model_surfaces_failure_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_find_lms(monkeypatch, tmp_path)
    monkeypatch.setattr(subprocess, "run", _SubprocessRecorder({
        "load": types.SimpleNamespace(
            returncode=1, stdout="", stderr="oh no something went wrong",
        ),
    }))
    r = load_model("nope", timeout_s=5.0)
    assert not r.ok
    assert "oh no" in r.detail


def test_load_model_handles_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_find_lms(monkeypatch, tmp_path)

    def boom(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="lms load", timeout=5.0)
    monkeypatch.setattr(subprocess, "run", boom)

    r = load_model("slow-model", timeout_s=5.0)
    assert not r.ok
    assert "timed out" in r.detail


# ---- unload + stop ---------------------------------------------------------


def test_unload_all_models_calls_lms_unload_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_find_lms(monkeypatch, tmp_path)
    rec = _SubprocessRecorder({"unload": types.SimpleNamespace(
        returncode=0, stdout="all models unloaded", stderr="",
    )})
    monkeypatch.setattr(subprocess, "run", rec)
    r = unload_all_models()
    assert r.ok
    assert any("--all" in c for c in rec.calls), f"calls: {rec.calls}"


def test_unload_specific_model_uses_named_arg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_find_lms(monkeypatch, tmp_path)
    rec = _SubprocessRecorder({"unload": types.SimpleNamespace(
        returncode=0, stdout="ok", stderr="",
    )})
    monkeypatch.setattr(subprocess, "run", rec)
    r = unload_model("qwen3-reranker-4b")
    assert r.ok
    # name appears in the argv but `--all` does not.
    assert any("qwen3-reranker-4b" in c for c in rec.calls)
    assert not any("--all" in c for c in rec.calls)


def test_stop_lms_server(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_find_lms(monkeypatch, tmp_path)
    rec = _SubprocessRecorder({"server": types.SimpleNamespace(
        returncode=0, stdout="server stopped", stderr="",
    )})
    monkeypatch.setattr(subprocess, "run", rec)
    r = stop_lms_server()
    assert r.ok
    assert any("server" in c and "stop" in c for c in rec.calls)


# ---- watcher task ----------------------------------------------------------


def test_stop_watcher_task_fast_path_when_already_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ops, "_ps_query", lambda script: "Ready")
    r = stop_watcher_task()
    assert r.ok
    assert "already stopped" in r.detail


def test_stop_watcher_task_when_not_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ops, "_ps_query", lambda script: "NotRegistered")
    r = stop_watcher_task()
    assert r.ok
    assert "not registered" in r.detail


def test_stop_watcher_task_actually_stops_when_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_ps_query is called twice: once for state, once for the Stop verb. We
    program both responses by tracking the call count."""
    seq = ["Running", "OK"]
    def fake_ps(script: str) -> str:
        return seq.pop(0)
    monkeypatch.setattr(ops, "_ps_query", fake_ps)
    r = stop_watcher_task()
    assert r.ok
    assert r.detail == "OK"


# ---- composite start_all / stop_all ----------------------------------------


def test_start_all_runs_steps_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _make_settings(tmp_path, monkeypatch)

    calls: list[str] = []
    def rec_step(name: str):
        return lambda *a, **kw: (
            calls.append(name)
            or StepResult(name, True, "stub")
        )
    monkeypatch.setattr(ops, "start_lms_server",  rec_step("start_lms_server"))
    monkeypatch.setattr(ops, "load_model",        lambda model, t: (
        calls.append(f"load_model({model})") or StepResult(f"load_model({model})", True, "stub")
    ))
    monkeypatch.setattr(ops, "start_watcher_task", rec_step("start_watcher_task"))

    res = ops.start_all(s)
    assert res.ok
    # Embedder loaded first, then chat reranker, then watcher.
    assert calls == [
        "start_lms_server",
        f"load_model({s.embedder.model})",
        f"load_model({s.reranker.model})",
        "start_watcher_task",
    ]


def test_start_all_aborts_when_lms_server_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If LM Studio can't come up, don't even attempt loads or the watcher
    task — those would just compound the failure with confusing messages."""
    s = _make_settings(tmp_path, monkeypatch)
    monkeypatch.setattr(ops, "start_lms_server",
                        lambda _s: StepResult("start_lms_server", False, "down"))
    called: list[str] = []
    monkeypatch.setattr(ops, "load_model",
                        lambda *a, **kw: called.append("load") or StepResult("x", True))
    monkeypatch.setattr(ops, "start_watcher_task",
                        lambda: called.append("watcher") or StepResult("y", True))
    res = ops.start_all(s)
    assert not res.ok
    assert called == [], "must short-circuit on first failure"


def test_stop_all_keeps_lm_studio_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _make_settings(tmp_path, monkeypatch)
    seen: list[str] = []
    monkeypatch.setattr(ops, "stop_watcher_task",
                        lambda: seen.append("stop_watcher") or StepResult("a", True))
    monkeypatch.setattr(ops, "unload_all_models",
                        lambda: seen.append("unload_all") or StepResult("b", True))
    monkeypatch.setattr(ops, "stop_lms_server",
                        lambda: seen.append("stop_lms") or StepResult("c", True))
    res = ops.stop_all(s)
    assert res.ok
    assert seen == ["stop_watcher", "unload_all"]


def test_stop_all_with_stop_lm_studio_kills_server_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _make_settings(tmp_path, monkeypatch)
    seen: list[str] = []
    monkeypatch.setattr(ops, "stop_watcher_task",
                        lambda: seen.append("a") or StepResult("a", True))
    monkeypatch.setattr(ops, "unload_all_models",
                        lambda: seen.append("b") or StepResult("b", True))
    monkeypatch.setattr(ops, "stop_lms_server",
                        lambda: seen.append("c") or StepResult("c", True))
    res = ops.stop_all(s, stop_lm_studio=True)
    assert res.ok
    assert seen == ["a", "b", "c"]

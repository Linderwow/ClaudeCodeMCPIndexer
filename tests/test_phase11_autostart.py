"""Phase 11: full boot-chain autostart.

We test the lms_ctl orchestration with the real httpx (against a stubbed
server) and a patched `subprocess.run` so no actual LM Studio process is
spawned. The autostart entry point itself is covered by a smoke test that
mocks out the slow bits.
"""
from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import httpx
import pytest

from code_rag import lms_ctl
from code_rag.lms_ctl import (
    BootstrapResult,
    LmsLocation,
    ensure_lm_studio_ready,
    find_lms,
    model_is_loaded,
    server_is_up,
)

# ---- reachability probes ----------------------------------------------------


class _FakeHttpxOk:
    status_code = 200
    def json(self) -> dict[str, Any]:
        return {"data": [{"id": "qwen3-embedding-4b"}, {"id": "some-other"}]}


class _FakeHttpxMissing:
    status_code = 200
    def json(self) -> dict[str, Any]:
        return {"data": [{"id": "some-other"}]}


def test_server_is_up_true(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lms_ctl.httpx, "get", lambda *a, **kw: _FakeHttpxOk())
    assert server_is_up("http://localhost:1234/v1") is True


def test_server_is_up_false_on_connect_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_conn(*a: Any, **kw: Any) -> Any:
        raise httpx.ConnectError("refused")
    monkeypatch.setattr(lms_ctl.httpx, "get", raise_conn)
    assert server_is_up("http://localhost:1234/v1") is False


def test_model_is_loaded_finds_configured_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lms_ctl.httpx, "get", lambda *a, **kw: _FakeHttpxOk())
    assert model_is_loaded("http://localhost:1234/v1", "qwen3-embedding-4b")


def test_model_is_loaded_returns_false_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lms_ctl.httpx, "get", lambda *a, **kw: _FakeHttpxMissing())
    assert not model_is_loaded("http://localhost:1234/v1", "qwen3-embedding-4b")


# ---- lms finder -------------------------------------------------------------


def test_find_lms_hits_path_first(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_lms = tmp_path / "lms.exe"
    fake_lms.write_bytes(b"")
    monkeypatch.setattr("shutil.which", lambda name: str(fake_lms) if name == "lms" else None)
    loc = find_lms()
    assert loc.path == fake_lms
    assert loc.found_via == "path"


def test_find_lms_returns_none_when_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda name: None)
    # Also suppress any file-existence hits for the Windows candidates.
    monkeypatch.setattr(Path, "exists", lambda self: False)  # type: ignore[method-assign]
    loc = find_lms()
    assert loc.path is None
    assert loc.found_via == "none"


# ---- ensure_lm_studio_ready orchestration ----------------------------------


def _install_mock_server(monkeypatch: pytest.MonkeyPatch, loaded: list[str]) -> None:
    """Patch httpx.get so /v1/models reports `loaded` ids."""
    class _Resp:
        status_code = 200
        def json(self) -> dict[str, Any]:
            return {"data": [{"id": m} for m in loaded]}
    monkeypatch.setattr(lms_ctl.httpx, "get", lambda *a, **kw: _Resp())


def test_ensure_ready_fast_path_when_already_loaded(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_mock_server(monkeypatch, ["qwen3-embedding-4b"])
    # Phase 44: the fast path now ALSO verifies the loaded ctx matches
    # Phase 33 prefs. find_lms() is now expected (we need lms.exe to
    # run `lms ps`), but `start_server` and `load_model` must still be
    # no-ops — the fast path doesn't reload an already-correct model.
    monkeypatch.setattr(
        lms_ctl, "find_lms",
        lambda: lms_ctl.LmsLocation(path=Path("lms.exe"), found_via="test"),
    )
    def fail_if_called(*a: Any, **kw: Any) -> Any:
        raise AssertionError("must not load/start when model already loaded with correct ctx")
    monkeypatch.setattr(lms_ctl, "start_server", fail_if_called)
    monkeypatch.setattr(lms_ctl, "load_model", fail_if_called)
    # qwen3-embedding-4b has no entry in _LMS_LOAD_SETTINGS, so the
    # ctx-check returns True without calling `lms ps` — fast path holds.

    r = ensure_lm_studio_ready("http://localhost:1234/v1", "qwen3-embedding-4b")
    assert r.ok
    assert any("already-ready" in s for s in r.steps)


def test_ensure_ready_no_lms_installed_reports_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_mock_server(monkeypatch, [])
    monkeypatch.setattr(lms_ctl, "find_lms", lambda: LmsLocation(None, "none"))
    r = ensure_lm_studio_ready("http://localhost:1234/v1", "qwen3-embedding-4b")
    assert not r.ok
    assert "lms CLI not found" in (r.error or "")


def test_ensure_ready_happy_path_with_server_startup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """Server not up, then lms start -> server up, then lms load -> model ready."""
    # Mutable state: list of 'loaded' model ids; starts empty to simulate server down.
    loaded: list[str] = []
    server_state = {"up": False}

    class _Resp:
        def __init__(self, up: bool, ids: list[str]):
            self.status_code = 200 if up else 0
            self._up = up
            self._ids = ids
        def json(self) -> dict[str, Any]:
            return {"data": [{"id": m} for m in self._ids]}

    def fake_get(*a: Any, **kw: Any) -> Any:
        if not server_state["up"]:
            raise httpx.ConnectError("refused")
        return _Resp(True, loaded)

    monkeypatch.setattr(lms_ctl.httpx, "get", fake_get)

    fake_lms = tmp_path / "lms.exe"
    fake_lms.write_bytes(b"")
    monkeypatch.setattr(lms_ctl, "find_lms",
                        lambda: LmsLocation(fake_lms, "path"))

    def fake_start(path: Path):
        server_state["up"] = True
        return types.SimpleNamespace(pid=999)
    monkeypatch.setattr(lms_ctl, "start_server", fake_start)

    def fake_load(path: Path, model: str, **kw: Any):
        loaded.append(model)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(lms_ctl, "load_model", fake_load)

    # Short sleep so the readiness poll doesn't actually take 2s per iteration.
    monkeypatch.setattr(lms_ctl.time, "sleep", lambda _x: None)

    r = ensure_lm_studio_ready(
        "http://localhost:1234/v1",
        "qwen3-embedding-4b",
        extra_models=(),
        ready_timeout_s=5.0,
    )
    assert r.ok, r
    # Check the breadcrumb trail.
    joined = " | ".join(r.steps)
    assert "started LM Studio server" in joined
    assert "ready" in joined


def test_ensure_ready_extra_model_failure_is_non_fatal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """If the reranker fails to load, the embedder bootstrap still succeeds."""
    loaded: list[str] = ["qwen3-embedding-4b"]   # embedder already ready, reranker NOT

    class _Resp:
        status_code = 200
        def json(self) -> dict[str, Any]:
            return {"data": [{"id": m} for m in loaded]}
    monkeypatch.setattr(lms_ctl.httpx, "get", lambda *a, **kw: _Resp())

    fake_lms = tmp_path / "lms.exe"
    fake_lms.write_bytes(b"")
    monkeypatch.setattr(lms_ctl, "find_lms",
                        lambda: LmsLocation(fake_lms, "path"))
    monkeypatch.setattr(lms_ctl, "start_server",
                        lambda p: types.SimpleNamespace(pid=1))
    monkeypatch.setattr(lms_ctl, "load_model",
                        lambda p, m, **kw: types.SimpleNamespace(
                            returncode=1, stdout="", stderr="nope"))
    # Mock wait_until_ready so the extras loop doesn't burn a real 60s.
    monkeypatch.setattr(
        lms_ctl, "wait_until_ready",
        lambda base_url, model, timeout_s=60.0: model in loaded,
    )
    monkeypatch.setattr(lms_ctl.time, "sleep", lambda _x: None)

    r = ensure_lm_studio_ready(
        "http://localhost:1234/v1",
        "qwen3-embedding-4b",
        extra_models=("reranker-that-never-loads",),
    )
    assert r.ok
    # Extra model load is logged but doesn't fail the bootstrap.
    joined = " | ".join(r.steps)
    assert "reranker-that-never-loads" in joined


# ---- BootstrapResult shape -------------------------------------------------


def test_bootstrap_result_holds_error_when_not_ok() -> None:
    r = BootstrapResult(ok=False, steps=["tried x", "tried y"], error="nope")
    assert not r.ok
    assert r.error == "nope"
    assert len(r.steps) == 2

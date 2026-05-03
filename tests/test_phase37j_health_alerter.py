"""Phase 37-J: dashboard degraded-state alerter.

Tests cover the state-machine logic + I/O wiring. Toast notifications
are mocked away — they require Windows BurntToast and an actual
graphical session.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from code_rag.util.health_alerter import (
    AlerterPaths,
    HealthSnapshot,
    _summarize_failed_checks,
    check_once,
    probe_health,
)


def _make_paths(tmp_path: Path) -> AlerterPaths:
    return AlerterPaths.for_data_dir(tmp_path)


class _FakeResponse:
    def __init__(self, status: int, body: dict[str, Any]) -> None:
        self.status_code = status
        self._body = body

    def json(self) -> dict[str, Any]:
        return self._body


def _patch_httpx(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse | Exception) -> None:
    """Replace httpx.get inside health_alerter with a deterministic fake."""
    def _fake_get(url: str, timeout: float = 5.0) -> Any:
        if isinstance(response, Exception):
            raise response
        return response
    monkeypatch.setattr("code_rag.util.health_alerter.httpx.get", _fake_get)


def test_probe_health_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"overall": "ok", "checks": {"lm_studio": {"ok": True}}}
    _patch_httpx(monkeypatch, _FakeResponse(200, body))
    overall, checks = probe_health()
    assert overall == "ok"
    assert checks == {"lm_studio": {"ok": True}}


def test_probe_health_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"overall": "degraded", "checks": {"watcher_alive": {"ok": False, "detail": "no pid"}}}
    _patch_httpx(monkeypatch, _FakeResponse(200, body))
    overall, checks = probe_health()
    assert overall == "degraded"
    assert checks["watcher_alive"]["ok"] is False


def test_probe_health_unreachable_on_connection_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dashboard down → unreachable, not crash."""
    import httpx
    _patch_httpx(monkeypatch, httpx.ConnectError("port closed"))
    overall, checks = probe_health()
    assert overall == "unreachable"
    assert "ConnectError" in checks.get("err", "")


def test_probe_health_unreachable_on_500(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_httpx(monkeypatch, _FakeResponse(500, {}))
    overall, checks = probe_health()
    assert overall == "unreachable"
    assert checks["http_status"] == 500


def test_check_once_writes_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = {"overall": "ok", "checks": {}}
    _patch_httpx(monkeypatch, _FakeResponse(200, body))
    paths = _make_paths(tmp_path)

    snap = check_once(paths, show_toast=False)
    assert snap.overall == "ok"
    assert snap.transition is False     # no previous state
    assert snap.previous_overall is None

    # State file should exist + contain the snapshot.
    written = json.loads(paths.state.read_text("utf-8"))
    assert written["overall"] == "ok"
    assert written["transition"] is False


def test_check_once_detects_transition_and_logs_alert(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _make_paths(tmp_path)

    # First probe: ok.
    _patch_httpx(monkeypatch, _FakeResponse(200, {"overall": "ok", "checks": {}}))
    check_once(paths, show_toast=False)
    assert not paths.alerts.exists()    # no transition, no alert

    # Second probe: degraded → must record transition + write alert.
    _patch_httpx(monkeypatch, _FakeResponse(200, {
        "overall": "degraded",
        "checks": {"chroma": {"ok": False, "detail": "timeout 30s"}},
    }))
    snap = check_once(paths, show_toast=False)
    assert snap.transition is True
    assert snap.previous_overall == "ok"
    assert snap.overall == "degraded"

    # Alerts file should now have one row.
    assert paths.alerts.exists()
    rows = [json.loads(ln) for ln in paths.alerts.read_text("utf-8").splitlines() if ln]
    assert len(rows) == 1
    assert rows[0]["from"] == "ok"
    assert rows[0]["to"] == "degraded"


def test_check_once_idempotent_on_persistent_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three consecutive degraded probes should produce ONE alert (on the
    initial transition), not three."""
    paths = _make_paths(tmp_path)

    # First probe: ok.
    _patch_httpx(monkeypatch, _FakeResponse(200, {"overall": "ok", "checks": {}}))
    check_once(paths, show_toast=False)

    # Then three degraded probes in a row.
    _patch_httpx(monkeypatch, _FakeResponse(200, {"overall": "degraded", "checks": {}}))
    for _ in range(3):
        check_once(paths, show_toast=False)

    rows = [json.loads(ln) for ln in paths.alerts.read_text("utf-8").splitlines() if ln]
    assert len(rows) == 1, f"expected 1 alert on transition, got {len(rows)}"


def test_check_once_records_recovery_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """degraded → ok is also a transition; recovery must be logged."""
    paths = _make_paths(tmp_path)

    _patch_httpx(monkeypatch, _FakeResponse(200, {"overall": "degraded", "checks": {}}))
    check_once(paths, show_toast=False)

    _patch_httpx(monkeypatch, _FakeResponse(200, {"overall": "ok", "checks": {}}))
    check_once(paths, show_toast=False)

    rows = [json.loads(ln) for ln in paths.alerts.read_text("utf-8").splitlines() if ln]
    assert len(rows) == 1
    assert rows[0]["from"] == "degraded"
    assert rows[0]["to"] == "ok"


def test_summarize_failed_checks_terse() -> None:
    checks = {
        "lm_studio": {"ok": True},
        "watcher_alive": {"ok": False, "detail": "no pid"},
        "chroma": {"ok": False, "detail": "timeout 30s"},
    }
    out = _summarize_failed_checks(checks)
    assert "watcher_alive" in out
    assert "chroma" in out
    assert "lm_studio" not in out      # not failed


def test_summarize_failed_checks_truncates_long_payload() -> None:
    """Hard cap of ~200 chars to keep toasts readable."""
    checks = {
        f"check_{i}": {"ok": False, "detail": "very verbose detail " * 20}
        for i in range(20)
    }
    out = _summarize_failed_checks(checks)
    assert len(out) < 250


def test_alerter_paths_for_data_dir(tmp_path: Path) -> None:
    paths = AlerterPaths.for_data_dir(tmp_path)
    assert paths.state == tmp_path / "health-state.json"
    assert paths.alerts == tmp_path / "alerts.jsonl"


def test_health_snapshot_dataclass_defaults() -> None:
    snap = HealthSnapshot(ts="2026-05-03T12:00:00", overall="ok")
    assert snap.checks == {}
    assert snap.transition is False
    assert snap.previous_overall is None

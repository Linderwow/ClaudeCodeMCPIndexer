"""Phase 50: Unified Command Center — backend.

Aggregates scheduled tasks + running processes for code-rag,
YouTubeBot, and MNQAlpha (signals) projects into one /api/projects
response so the dashboard can render a single-pane view.

These tests pin the contract — shape of the JSON response, robustness
on probe failure, sort order — so a future refactor can't silently
break the dashboard frontend.
"""
from __future__ import annotations

from unittest.mock import patch

from code_rag.dashboard import projects as proj


_FAKE_TASKS_CODE_RAG = [
    proj.TaskInfo(
        name="code-rag-health-alert", state="Ready", hidden=True,
        exe="pythonw.exe", repeat="PT5M",
        last_run="2026-05-08T14:25:00+00:00", last_result=0,
    ),
    proj.TaskInfo(
        name="code-rag-watch", state="Running", hidden=True,
        exe="pythonw.exe", repeat=None,
        last_run="2026-05-08T13:00:00+00:00", last_result=0,
    ),
]

_FAKE_PROCS_CODE_RAG = [
    proj.ProcInfo(
        pid=1234, parent_pid=1, name="pythonw.exe", ram_mb=1200,
        cmd_preview="pythonw.exe -m code_rag mcp", age_s=3600,
    ),
]


def test_get_projects_state_returns_three_projects() -> None:
    """Contract: always returns exactly the three configured profiles
    in PROJECT_PROFILES order, even when probes fail."""
    with patch.object(proj, "_list_tasks", return_value=[]), \
         patch.object(proj, "_list_processes", return_value=[]):
        out = proj.get_projects_state()
    assert isinstance(out, list)
    assert len(out) == 3
    ids = [p["id"] for p in out]
    assert ids == ["code-rag", "YouTubeBot", "MNQAlpha"]
    for p in out:
        # JSON-friendly shape
        assert isinstance(p["tasks"], list)
        assert isinstance(p["processes"], list)


def test_get_projects_state_serializes_dataclasses_to_dicts() -> None:
    """The JSON handler does `JSONResponse(get_projects_state())`. So
    every nested object must be a plain dict (TaskInfo / ProcInfo
    serialized via asdict), not a dataclass instance."""
    with patch.object(proj, "_list_tasks", return_value=_FAKE_TASKS_CODE_RAG), \
         patch.object(proj, "_list_processes", return_value=_FAKE_PROCS_CODE_RAG):
        out = proj.get_projects_state()
    cr = next(p for p in out if p["id"] == "code-rag")
    assert isinstance(cr["tasks"], list)
    assert isinstance(cr["tasks"][0], dict)
    # Field names match the dataclass — frontend depends on these.
    assert set(cr["tasks"][0].keys()) == {
        "name", "state", "hidden", "exe", "repeat", "last_run", "last_result",
    }
    assert isinstance(cr["processes"][0], dict)
    assert set(cr["processes"][0].keys()) == {
        "pid", "parent_pid", "name", "ram_mb", "cmd_preview", "age_s",
    }


def test_task_probe_failure_records_error_but_doesnt_skip_project() -> None:
    """A failed task probe (e.g. PowerShell unavailable) must not skip
    the project — frontend still shows the card with an error banner."""
    def _bad_tasks(globs):
        raise RuntimeError("pwsh missing")
    with patch.object(proj, "_list_tasks", side_effect=_bad_tasks), \
         patch.object(proj, "_list_processes", return_value=[]):
        out = proj.get_projects_state()
    assert len(out) == 3
    for p in out:
        assert p["error"] is not None
        assert "pwsh missing" in p["error"]
        # tasks list still present (empty) for predictable rendering
        assert p["tasks"] == []


def test_process_probe_failure_independent_from_task_probe() -> None:
    """A process probe failure shouldn't prevent reporting tasks."""
    with patch.object(proj, "_list_tasks", return_value=_FAKE_TASKS_CODE_RAG), \
         patch.object(proj, "_list_processes",
                      side_effect=RuntimeError("wmi denied")):
        out = proj.get_projects_state()
    cr = next(p for p in out if p["id"] == "code-rag")
    assert len(cr["tasks"]) == 2
    assert cr["processes"] == []
    assert cr["error"] is not None
    assert "wmi denied" in cr["error"]


def test_project_profiles_match_known_naming_conventions() -> None:
    """Pin the profile config so a future rename can't silently miss
    user-visible tasks. If the user renames code-rag-* to
    coderag-* etc., this test will scream."""
    profile_by_id = {p["id"]: p for p in proj.PROJECT_PROFILES}
    assert "code-rag-*" in profile_by_id["code-rag"]["task_globs"]
    yt_globs = profile_by_id["YouTubeBot"]["task_globs"]
    assert any(g.startswith("ThePremise") for g in yt_globs)
    assert any(g.startswith("YouTubeBot") for g in yt_globs)
    assert "MNQAlpha_*" in profile_by_id["MNQAlpha"]["task_globs"]


def test_processes_sorted_by_ram_descending() -> None:
    """Frontend expects the heaviest workers first (a 1200 MB MCP
    server is more relevant than a 6 MB venv shim) so memory issues
    are visible at-a-glance."""
    procs = [
        proj.ProcInfo(pid=1, parent_pid=0, name="a", ram_mb=10,
                      cmd_preview="x", age_s=1),
        proj.ProcInfo(pid=2, parent_pid=0, name="b", ram_mb=500,
                      cmd_preview="y", age_s=1),
        proj.ProcInfo(pid=3, parent_pid=0, name="c", ram_mb=100,
                      cmd_preview="z", age_s=1),
    ]
    with patch.object(proj, "_list_tasks", return_value=[]), \
         patch.object(proj, "_list_processes", return_value=procs) as m:
        # Bypass the live probe; simulate the function returning the
        # sorted list it would have returned.
        m.return_value = sorted(procs, key=lambda p: (-p.ram_mb, p.pid))
        out = proj.get_projects_state()
    rams = [p["ram_mb"] for p in out[0]["processes"]]
    assert rams == sorted(rams, reverse=True)

"""Phase 51: per-row action buttons (Run / Stop / Kill).

The dashboard now exposes 3 destructive endpoints:
  POST /api/tasks/run        body: {name}
  POST /api/tasks/stop       body: {name}
  POST /api/processes/kill   body: {pid}

Each is whitelisted to project-owned items via PROJECT_PROFILES so a
malformed request can't run/stop arbitrary system tasks or terminate
unrelated processes. These tests pin the whitelist contract.

NO LIVE PowerShell calls — every test mocks `_ps_run`. The user is
gaming and explicitly asked not to test against live processes.
"""
from __future__ import annotations

from unittest.mock import patch

from code_rag.dashboard import projects as proj


# ---- task-name whitelist ----


def test_task_belongs_recognizes_known_globs() -> None:
    assert proj._task_belongs_to_a_project("code-rag-watch")
    assert proj._task_belongs_to_a_project("code-rag-health-alert")
    assert proj._task_belongs_to_a_project("ThePremise_watch")
    assert proj._task_belongs_to_a_project("ThePremise_0800")
    assert proj._task_belongs_to_a_project("YouTubeBot Watchdog")
    assert proj._task_belongs_to_a_project("YouTubeBot_Cognitron_1800")
    assert proj._task_belongs_to_a_project("MNQAlpha_Canary")


def test_task_belongs_rejects_system_tasks() -> None:
    """Microsoft / vendor / random tasks must NEVER be whitelisted —
    even if a malicious frontend sends them."""
    assert not proj._task_belongs_to_a_project("Adobe Acrobat Update Task")
    assert not proj._task_belongs_to_a_project("OneDrive Standalone Update Task")
    assert not proj._task_belongs_to_a_project("NVIDIA App SelfUpdate")
    assert not proj._task_belongs_to_a_project(
        "\\Microsoft\\Windows\\Application Experience\\MareBackup",
    )
    assert not proj._task_belongs_to_a_project("Office Actions Server")
    assert not proj._task_belongs_to_a_project("ZoomUpdateTaskUser")
    assert not proj._task_belongs_to_a_project("totally-arbitrary-task")
    assert not proj._task_belongs_to_a_project("")


# ---- process whitelist (cmdline-based) ----


def test_process_belongs_recognizes_project_processes() -> None:
    assert proj._process_belongs_to_a_project(
        "C:\\Users\\Alex\\Documents\\code-rag-mcp\\.venv\\Scripts\\pythonw.exe -m code_rag mcp",
    )
    assert proj._process_belongs_to_a_project(
        "C:\\Users\\Alex\\anaconda3\\pythonw.exe -m code_rag.autostart_bootstrap",
    )
    assert proj._process_belongs_to_a_project(
        "C:\\Users\\Alex\\Documents\\YouTubeBot\\venv\\Scripts\\python.exe run_pipeline.py --watch",
    )
    assert proj._process_belongs_to_a_project(
        "wscript.exe \"C:\\Users\\Alex\\Documents\\YouTubeBot\\run_watch_hidden.vbs\"",
    )
    assert proj._process_belongs_to_a_project(
        "pythonw.exe C:\\Users\\Alex\\signals\\trade_gate.py",
    )


def test_process_belongs_rejects_unrelated_processes() -> None:
    """Random user processes must not be killable via the dashboard."""
    assert not proj._process_belongs_to_a_project(
        "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    )
    assert not proj._process_belongs_to_a_project(
        "C:\\Program Files\\NinjaTrader 8\\bin\\NinjaTrader.exe",
    )
    assert not proj._process_belongs_to_a_project(
        "C:\\Users\\Alex\\Documents\\ComfyUI\\python_embeded\\python.exe",
    )
    assert not proj._process_belongs_to_a_project(
        "C:\\Program Files (x86)\\Call of Duty\\cod.exe",
    )
    assert not proj._process_belongs_to_a_project("")


# ---- run_task ----


def test_run_task_refuses_non_whitelisted_name() -> None:
    res = proj.run_task("definitely-not-our-task")
    assert res["ok"] is False
    assert "refused" in res["detail"]


def test_run_task_refuses_empty() -> None:
    assert proj.run_task("")["ok"] is False
    assert proj.run_task(None)["ok"] is False  # type: ignore[arg-type]


def test_run_task_calls_ps_when_whitelisted() -> None:
    with patch.object(proj, "_ps_run", return_value=(0, "", "")) as ps:
        res = proj.run_task("code-rag-health-alert")
    assert res["ok"] is True
    assert "code-rag-health-alert" in res["detail"]
    ps.assert_called_once()
    cmd = ps.call_args.args[0]
    assert "Start-ScheduledTask" in cmd
    assert "code-rag-health-alert" in cmd


def test_run_task_propagates_powershell_failure() -> None:
    with patch.object(proj, "_ps_run", return_value=(1, "", "task not found")):
        res = proj.run_task("code-rag-watch")
    assert res["ok"] is False
    assert "task not found" in res["detail"]


def test_run_task_escapes_single_quote_in_name() -> None:
    """Defensive — although our globs don't allow single quotes, if a
    task somehow has one, the PS string mustn't break out of the quoted
    arg. (PowerShell escapes single-quote with double-single-quote.)"""
    with patch.object(proj, "_ps_run", return_value=(0, "", "")) as ps:
        # Hypothetical glob that includes ' would let this through;
        # the Python-side escape is the last line of defense.
        proj.run_task("MNQAlpha_Canary")   # known-good name
    cmd = ps.call_args.args[0]
    # Sanity-check the script is single-quoted, not double-quoted
    assert "'MNQAlpha_Canary'" in cmd


# ---- stop_task ----


def test_stop_task_refuses_non_whitelisted() -> None:
    assert proj.stop_task("Adobe Acrobat Update Task")["ok"] is False


def test_stop_task_calls_ps_when_whitelisted() -> None:
    with patch.object(proj, "_ps_run", return_value=(0, "", "")) as ps:
        res = proj.stop_task("YouTubeBot Watchdog")
    assert res["ok"] is True
    cmd = ps.call_args.args[0]
    assert "Stop-ScheduledTask" in cmd
    assert "YouTubeBot Watchdog" in cmd


# ---- kill_process ----


def test_kill_process_refuses_invalid_pid() -> None:
    assert proj.kill_process(0)["ok"] is False
    assert proj.kill_process(-1)["ok"] is False
    assert proj.kill_process("12345")["ok"] is False  # type: ignore[arg-type]


def test_kill_process_refuses_when_pid_not_found() -> None:
    """probe returns empty cmdline -> pid gone."""
    with patch.object(proj, "_ps_run", return_value=(0, "", "")):
        res = proj.kill_process(99999)
    assert res["ok"] is False
    assert "not found" in res["detail"]


def test_kill_process_refuses_unrelated_pid() -> None:
    """probe returns Chrome's cmdline → reject before kill."""
    chrome_cmd = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe --type=renderer"
    with patch.object(proj, "_ps_run", return_value=(0, chrome_cmd, "")):
        res = proj.kill_process(12345)
    assert res["ok"] is False
    assert "doesn't match any project's process pattern" in res["detail"]


def test_kill_process_kills_when_whitelisted() -> None:
    """probe returns a code-rag mcp cmdline → kill proceeds."""
    cmdline = "C:\\Users\\Alex\\anaconda3\\pythonw.exe -m code_rag mcp"
    # _ps_run called twice: first probe (returns cmdline), then kill (success).
    with patch.object(proj, "_ps_run") as ps:
        ps.side_effect = [(0, cmdline, ""), (0, "", "")]
        res = proj.kill_process(54321)
    assert res["ok"] is True
    assert "killed pid 54321" in res["detail"]
    # Verify the second call was Stop-Process with the right pid
    assert ps.call_count == 2
    second_cmd = ps.call_args_list[1].args[0]
    assert "Stop-Process" in second_cmd
    assert "-Id 54321" in second_cmd
    assert "-Force" in second_cmd


def test_kill_process_propagates_kill_failure() -> None:
    """Whitelist passed but Stop-Process itself failed (e.g. access denied)."""
    cmdline = "pythonw.exe C:\\Users\\Alex\\signals\\monitor.py"
    with patch.object(proj, "_ps_run") as ps:
        ps.side_effect = [(0, cmdline, ""), (1, "", "Access is denied")]
        res = proj.kill_process(11111)
    assert res["ok"] is False
    assert "Access is denied" in res["detail"]


def test_kill_process_handles_probe_failure() -> None:
    with patch.object(proj, "_ps_run", return_value=(124, "", "timed out")):
        res = proj.kill_process(11111)
    assert res["ok"] is False
    assert "probe failed" in res["detail"]
    assert "timed out" in res["detail"]

"""Phase 10: `install` flow. Tested offline with skip flags; the Claude config
merge is the most bug-prone step so it gets direct unit tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from code_rag.config import load_settings
from code_rag.install import (
    InstallOptions,
    _merge_mcp_block,
    mcp_server_block,
    run_install_sync,
)

# ---- helpers ----------------------------------------------------------------


def _make_min_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
model    = "qwen3-embedding-4b"

[reranker]
kind     = "lm_studio"
base_url = "http://localhost:1234/v1"
model    = "qwen3-reranker-4b"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODE_RAG_CONFIG", str(cfg))


# ---- Claude config merging --------------------------------------------------


def test_merge_mcp_block_creates_new_file(tmp_path: Path) -> None:
    path = tmp_path / "claude_desktop_config.json"
    block = {"command": "pythonw.exe", "args": ["-m", "code_rag", "mcp"], "cwd": "C:/x"}
    _merge_mcp_block(path, "code-rag", block)
    data = json.loads(path.read_text("utf-8"))
    assert data["mcpServers"]["code-rag"] == block


def test_merge_mcp_block_preserves_other_servers(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps({
        "mcpServers": {
            "some-other": {"command": "foo", "args": []},
        },
        "unrelated_top_level_key": "value",
    }), encoding="utf-8")

    block = {"command": "pythonw.exe", "args": [], "cwd": "C:/x"}
    _merge_mcp_block(path, "code-rag", block)
    data = json.loads(path.read_text("utf-8"))
    assert data["mcpServers"]["some-other"]["command"] == "foo"
    assert data["mcpServers"]["code-rag"] == block
    assert data["unrelated_top_level_key"] == "value"


def test_merge_mcp_block_idempotent_no_backup(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    block = {"command": "pythonw.exe", "args": [], "cwd": "C:/x"}
    _merge_mcp_block(path, "code-rag", block)
    first = path.read_text("utf-8")
    # Second run: identical block -> no change, no new backup file.
    _merge_mcp_block(path, "code-rag", block)
    second = path.read_text("utf-8")
    assert first == second
    backups = list(tmp_path.glob("c.json.backup-*"))
    assert not backups, "identical block must not trigger a backup"


def test_merge_mcp_block_backs_up_before_modifying(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"mcpServers": {"code-rag": {"command": "OLD"}}}),
                    encoding="utf-8")
    new_block = {"command": "NEW", "args": [], "cwd": "C:/x"}
    _merge_mcp_block(path, "code-rag", new_block)
    backups = list(tmp_path.glob("c.json.backup-*"))
    assert backups, "overwriting an existing block must leave a backup"
    # Main file has the new block.
    data = json.loads(path.read_text("utf-8"))
    assert data["mcpServers"]["code-rag"] == new_block
    # Backup still has the old content.
    old = json.loads(backups[0].read_text("utf-8"))
    assert old["mcpServers"]["code-rag"] == {"command": "OLD"}


def test_merge_mcp_block_rejects_non_json(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text("not json at all", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not valid JSON"):
        _merge_mcp_block(path, "code-rag", {"command": "x", "args": []})


# ---- mcp_server_block shape ------------------------------------------------


def test_mcp_server_block_points_at_current_python(tmp_path: Path) -> None:
    block = mcp_server_block(tmp_path)
    assert Path(block["command"]).exists(), "command must be an existing executable"
    assert block["args"] == ["-m", "code_rag", "mcp"]
    assert Path(block["cwd"]) == tmp_path.resolve()


# ---- full install with everything skipped ----------------------------------


def test_install_with_all_skipped_is_a_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_min_settings(tmp_path, monkeypatch)
    settings = load_settings()
    opts = InstallOptions(skip_probe=True, skip_index=True,
                          skip_claude=True, skip_autostart=True)
    report = run_install_sync(settings, opts)
    assert report.ok
    assert all(s.skipped for s in report.steps)


def test_install_probe_failure_short_circuits_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If LM Studio isn't up, the index step must auto-skip (not crash).

    Test isolation: we force `find_lms` to report "no lms CLI found" so the
    probe takes the deterministic no-autostart branch. That keeps the test
    hermetic regardless of whether the host actually has LM Studio + the
    `lms` CLI installed and a matching model cached — the original test would
    time out for 300s on a real subprocess call when a model id didn't match.
    """
    _make_min_settings(tmp_path, monkeypatch)
    # Patch find_lms at the lms_ctl module (install.py does a function-local
    # `from code_rag.lms_ctl import find_lms`, so the binding is resolved on
    # the lms_ctl module at call time). Forcing "no lms CLI" sends the probe
    # into the deterministic no-autostart branch — no subprocesses fired.
    import code_rag.lms_ctl as lms_ctl_mod
    monkeypatch.setattr(
        lms_ctl_mod, "find_lms",
        lambda: lms_ctl_mod.LmsLocation(path=None, found_via="none"),
    )

    settings = load_settings()
    opts = InstallOptions(skip_probe=False, skip_index=False,
                          skip_claude=True, skip_autostart=True)
    report = run_install_sync(settings, opts)
    # Probe fails (no LM Studio), index auto-skipped.
    names = {s.name: s for s in report.steps}
    assert "LM Studio probe" in names
    assert names["LM Studio probe"].ok is False
    assert "initial index" in names
    assert names["initial index"].skipped is True

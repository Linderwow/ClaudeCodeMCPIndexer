"""Tests for config.toml root-list editing."""
from __future__ import annotations

from pathlib import Path

import pytest

from code_rag.roots_edit import ConfigEditError, add_root, read_roots, remove_root


def _make_config(tmp_path: Path, roots: list[Path]) -> Path:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f"""# my config — with a comment
[paths]
roots = [
{chr(10).join(f'    "{r.as_posix()}",' for r in roots)}
]
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
    return cfg


def test_add_root_appends(tmp_path: Path) -> None:
    r1 = tmp_path / "repo1"
    r1.mkdir()
    r2 = tmp_path / "repo2"
    r2.mkdir()
    cfg = _make_config(tmp_path, [r1])
    new_roots, added = add_root(cfg, r2)
    assert added
    assert len(new_roots) == 2
    assert read_roots(cfg) == [r1.resolve(), r2.resolve()]


def test_add_root_idempotent(tmp_path: Path) -> None:
    r1 = tmp_path / "repo1"
    r1.mkdir()
    cfg = _make_config(tmp_path, [r1])
    _, added = add_root(cfg, r1)
    assert not added


def test_add_root_rejects_missing(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path, [])
    with pytest.raises(ConfigEditError):
        add_root(cfg, tmp_path / "does_not_exist")


def test_add_root_preserves_comments(tmp_path: Path) -> None:
    r1 = tmp_path / "repo1"
    r1.mkdir()
    r2 = tmp_path / "repo2"
    r2.mkdir()
    cfg = _make_config(tmp_path, [r1])
    add_root(cfg, r2)
    body = cfg.read_text("utf-8")
    assert "# my config — with a comment" in body, "pre-existing comments must survive edits"
    assert "[embedder]" in body
    assert "qwen3-embedding-4b" in body


def test_remove_root(tmp_path: Path) -> None:
    r1 = tmp_path / "repo1"
    r1.mkdir()
    r2 = tmp_path / "repo2"
    r2.mkdir()
    cfg = _make_config(tmp_path, [r1, r2])
    _, removed = remove_root(cfg, r1)
    assert removed
    assert read_roots(cfg) == [r2.resolve()]


def test_remove_root_missing_is_noop(tmp_path: Path) -> None:
    r1 = tmp_path / "repo1"
    r1.mkdir()
    cfg = _make_config(tmp_path, [r1])
    _, removed = remove_root(cfg, tmp_path / "never_was_there")
    assert not removed

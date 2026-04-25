"""Phase 17 final wiring: `[embedder].preset` in config.toml routes through
factory.build_embedder to the right LMStudioEmbedder model id."""
from __future__ import annotations

from pathlib import Path

import pytest

from code_rag.config import load_settings
from code_rag.embedders.lm_studio import LMStudioEmbedder
from code_rag.factory import build_embedder


def _make_cfg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preset: str | None) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    preset_line = f'preset = "{preset}"\n' if preset else ""
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
model    = "fallback-model-id"
{preset_line}
[reranker]
kind     = "lm_studio"
base_url = "http://localhost:1234/v1"
model    = "qwen3-reranker-4b"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODE_RAG_CONFIG", str(cfg))


def test_no_preset_uses_explicit_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Backward compatibility: without `preset`, the explicit `model` field
    must be honored (so existing config.toml files keep working)."""
    _make_cfg(tmp_path, monkeypatch, preset=None)
    s = load_settings()
    e = build_embedder(s)
    assert isinstance(e, LMStudioEmbedder)
    assert e.model == "fallback-model-id"


def test_preset_routes_to_correct_lms_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A preset must override the explicit `model` field — the factory routes
    via `build_code_embedder` and pins the canonical LM Studio id."""
    _make_cfg(tmp_path, monkeypatch, preset="bge-code-v1")
    s = load_settings()
    e = build_embedder(s)
    assert isinstance(e, LMStudioEmbedder)
    # The preset's `lms_id` (matches what `lms ls` reports), NOT the friendly
    # name and NOT the fallback `model` field.
    assert e.model == "bge-code-v1"


def test_unknown_preset_falls_back_to_explicit_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo in `preset` should NOT silently break the embedder. The factory
    falls back to the explicit `model` field; the build doesn't crash."""
    _make_cfg(tmp_path, monkeypatch, preset="not-a-real-preset-xyz")
    s = load_settings()
    e = build_embedder(s)
    assert isinstance(e, LMStudioEmbedder)
    assert e.model == "fallback-model-id"

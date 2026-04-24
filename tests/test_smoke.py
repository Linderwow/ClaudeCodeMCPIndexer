"""Phase 0 smoke tests — no network, no external deps beyond pip-installed."""
from __future__ import annotations

from pathlib import Path

from code_rag import __version__
from code_rag.config import load_settings
from code_rag.models import Chunk, ChunkKind


def test_version_is_set() -> None:
    assert __version__
    assert __version__.count(".") == 2


def test_config_loads(tmp_path: Path, monkeypatch) -> None:
    # Point CODE_RAG_CONFIG at a minimal config using tmp_path for roots/data.
    root = tmp_path / "fake_root"
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
    s = load_settings()
    assert s.paths.roots == [root]
    assert s.embedder.model == "qwen3-embedding-4b"
    assert s.chroma_dir.parent == s.paths.data_dir


def test_config_resolves_relative_paths_against_config_dir(
    tmp_path: Path, monkeypatch,
) -> None:
    """`./data` and relative roots must resolve against the config file's
    parent directory, not CWD — otherwise the MCP server and CLI (different
    CWDs) would diverge on where the index lives."""
    cfg_dir = tmp_path / "project"
    cfg_dir.mkdir()
    (cfg_dir / "subroot").mkdir()
    cfg = cfg_dir / "config.toml"
    cfg.write_text(
        """
[paths]
roots    = ["./subroot"]
data_dir = "./data"
log_dir  = "./logs"

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
    # Change CWD to something UNRELATED; resolution must not key off CWD.
    monkeypatch.chdir(tmp_path)
    s = load_settings()
    assert s.paths.roots == [(cfg_dir / "subroot").resolve()]
    assert s.paths.data_dir == (cfg_dir / "data").resolve()
    assert s.paths.log_dir == (cfg_dir / "logs").resolve()


def test_chunk_content_addressed_shape() -> None:
    c = Chunk(
        id="abc",
        repo="r",
        path="p.cs",
        language="c_sharp",
        symbol="Foo",
        kind=ChunkKind.CLASS,
        start_line=1,
        end_line=10,
        text="class Foo {}",
    )
    assert c.n_chars == len("class Foo {}")

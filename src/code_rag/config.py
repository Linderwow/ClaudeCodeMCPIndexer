from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.toml"


class PathsConfig(BaseModel):
    roots: list[Path]
    data_dir: Path
    log_dir: Path

    # Root existence is checked in `load_settings` AFTER relative paths are
    # resolved against the config file's directory — otherwise a relative root
    # would be validated against the wrong base (CWD).


class IgnoreConfig(BaseModel):
    globs: list[str] = Field(default_factory=list)


class EmbedderConfig(BaseModel):
    kind: Literal["lm_studio"] = "lm_studio"
    base_url: str
    model: str
    dim: int = 0
    timeout_s: float = 60.0
    batch: int = 32


class RerankerConfig(BaseModel):
    kind: Literal["lm_studio"] = "lm_studio"
    base_url: str
    model: str
    top_k_in: int = 50
    top_k_out: int = 8
    timeout_s: float = 60.0


class VectorStoreConfig(BaseModel):
    kind: Literal["chroma"] = "chroma"
    collection: str = "code_rag_v1"


class GraphStoreConfig(BaseModel):
    kind: Literal["kuzu"] = "kuzu"
    db: str = "graph"


class LexicalStoreConfig(BaseModel):
    kind: Literal["sqlite_fts5"] = "sqlite_fts5"
    db: str = "fts.db"


class ChunkerConfig(BaseModel):
    min_chars: int = 80
    max_chars: int = 2400
    overlap_chars: int = 0


class WatcherConfig(BaseModel):
    debounce_ms: int = 500


class McpConfig(BaseModel):
    name: str = "code-rag"


class Settings(BaseModel):
    paths: PathsConfig
    ignore: IgnoreConfig = Field(default_factory=IgnoreConfig)
    embedder: EmbedderConfig
    reranker: RerankerConfig
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    graph_store: GraphStoreConfig = Field(default_factory=GraphStoreConfig)
    lexical_store: LexicalStoreConfig = Field(default_factory=LexicalStoreConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    watcher: WatcherConfig = Field(default_factory=WatcherConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)

    # Resolved absolute paths to vector DB dir, kuzu DB dir, FTS file.
    @property
    def chroma_dir(self) -> Path:
        return self.paths.data_dir / "chroma"

    @property
    def kuzu_dir(self) -> Path:
        return self.paths.data_dir / self.graph_store.db

    @property
    def fts_path(self) -> Path:
        return self.paths.data_dir / self.lexical_store.db

    @property
    def index_meta_path(self) -> Path:
        return self.paths.data_dir / "index_meta.json"

    @property
    def dynamic_roots_path(self) -> Path:
        """Registry of runtime-added roots (auto-discovered workspaces)."""
        return self.paths.data_dir / "dynamic_roots.json"

    def all_roots(self) -> list[Path]:
        """Union of the user's curated `config.toml` roots and any
        dynamically-added roots persisted to `dynamic_roots.json`.

        The returned list is deduplicated (resolved-path equality) and only
        includes directories that currently exist on disk.
        """
        # Lazy import: dynamic_roots imports config.logging which can loop
        # during config module init.
        from code_rag.dynamic_roots import DynamicRoots
        seen: set[Path] = set()
        out: list[Path] = []
        for r in self.paths.roots:
            try:
                rr = r.resolve()
            except OSError:
                continue
            if rr not in seen and rr.exists() and rr.is_dir():
                seen.add(rr)
                out.append(rr)
        dyn = DynamicRoots.load(self.dynamic_roots_path)
        for rr in dyn.paths():
            if rr not in seen:
                seen.add(rr)
                out.append(rr)
        return out


def _config_path() -> Path:
    env = os.environ.get("CODE_RAG_CONFIG")
    if env:
        p = Path(env)
        if not p.exists():
            raise FileNotFoundError(f"CODE_RAG_CONFIG points to missing file: {p}")
        return p
    return DEFAULT_CONFIG_PATH


def load_settings(path: Path | None = None) -> Settings:
    p = path or _config_path()
    with p.open("rb") as f:
        raw = tomllib.load(f)
    settings = Settings.model_validate(raw)
    # Resolve relative paths against the config file's parent dir so `./data`
    # in config.toml always means `<repo_root>/data` regardless of CWD.
    config_dir = p.resolve().parent
    if not settings.paths.data_dir.is_absolute():
        settings.paths.data_dir = (config_dir / settings.paths.data_dir).resolve()
    if not settings.paths.log_dir.is_absolute():
        settings.paths.log_dir = (config_dir / settings.paths.log_dir).resolve()
    settings.paths.roots = [
        r if r.is_absolute() else (config_dir / r).resolve()
        for r in settings.paths.roots
    ]
    # Ensure writable dirs exist.
    settings.paths.data_dir.mkdir(parents=True, exist_ok=True)
    settings.paths.log_dir.mkdir(parents=True, exist_ok=True)
    return settings

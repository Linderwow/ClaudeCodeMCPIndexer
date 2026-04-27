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
    # Phase 17: opt-in shorthand for one of the curated code-specialized
    # presets (e.g. "bge-code-v1", "codesage-large-v2", "qwen3-embedding-4b").
    # When set AND `model` is left at its default, factory.build_embedder
    # routes through the preset (auto-resolves the right LM Studio model id).
    preset: str | None = None
    # Phase 19: optional small chat-completion model for HyDE generation.
    # If not set or not loaded, HyDE silently falls back to literal-only.
    hyde_model: str | None = None


class RerankerConfig(BaseModel):
    # "lm_studio"     : OpenAI-compatible /v1/rerank endpoint (Jina-style).
    #                   LM Studio itself does NOT implement this endpoint — kept
    #                   for backends that do (vLLM, text-embeddings-inference).
    # "lm_chat"       : Phase 24 listwise reranker that uses /v1/chat/completions
    #                   on a small chat model. The right choice for LM Studio.
    # "cross_encoder" : Phase 29 sentence-transformers CrossEncoder (e.g.
    #                   BAAI/bge-reranker-v2-m3). Faster + more deterministic
    #                   than lm_chat. Requires `pip install -e ".[cross-encoder]"`.
    # "noop"          : explicit no-op, useful for ablation eval and CI.
    kind: Literal["lm_studio", "lm_chat", "cross_encoder", "noop"] = "lm_studio"
    base_url: str
    model: str
    top_k_in: int = 50
    top_k_out: int = 8
    timeout_s: float = 60.0
    # Phase 24: for kind="lm_chat" only. How many top candidates to send to
    # the LLM (the rest stay in their RRF order). Bounded for latency.
    chat_max_candidates: int = 20
    chat_max_chars_per_doc: int = 300
    # Phase 29: for kind="cross_encoder" only. CPU is fine; set to "cuda"
    # if you have a GPU and want to use it.
    device: str | None = None
    # Phase 29: max chars per doc fed into the cross-encoder. 600 covers
    # signature + a few body lines on most languages.
    cross_encoder_max_chars: int = 600


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


class IndexerConfig(BaseModel):
    """Parallel-pipeline tuning (Phase 15).

    `parallel_workers` files-in-flight at the chunk+embed stage; store writes
    are serialized inside the indexer via a single async lock to keep
    Chroma/Kuzu/FTS upserts ordered.
    """
    parallel_workers: int = 4
    embed_concurrency: int = 4   # async tasks per embed batch (server-side parallel)


class WatcherConfig(BaseModel):
    debounce_ms: int = 500


class QueryRewriterConfig(BaseModel):
    """Phase 30: query expansion / rewriting.

    When `enabled = true`, the searcher pre-processes each query into
    casing variants (snake_case ↔ CamelCase ↔ spaced) and — if `model`
    is set and loaded — uses a small chat model to suggest related
    identifiers. Each expansion becomes a parallel search arm.

    Cache is on by default to keep latency predictable on repeated queries.
    Cache file lives under `paths.data_dir`.
    """
    enabled: bool = False
    base_url: str = ""
    model: str | None = None        # if None, only local (free) rewrite runs
    timeout_s: float = 8.0
    cache: bool = True
    db: str = "rewrite_cache.db"


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
    indexer: IndexerConfig = Field(default_factory=IndexerConfig)
    watcher: WatcherConfig = Field(default_factory=WatcherConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)
    query_rewriter: QueryRewriterConfig = Field(default_factory=QueryRewriterConfig)

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

    @property
    def file_hashes_path(self) -> Path:
        """Per-file content hash registry — Phase 14 incremental indexing."""
        return self.paths.data_dir / "file_hashes.db"

    @property
    def rewrite_cache_path(self) -> Path:
        """Phase 30: query rewrite cache (SQLite k/v with TTL)."""
        return self.paths.data_dir / self.query_rewriter.db

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


def _expand(p: Path) -> Path:
    """Expand `~` and environment variables in a config-supplied path.

    Allows the same `config.toml` to travel across machines / accounts:
    `~/RiderProjects` resolves to whoever's home directory, and
    `${USERPROFILE}/code` works on Windows. Unset env vars are left as-is.
    """
    s = os.path.expandvars(str(p))
    s = os.path.expanduser(s)
    return Path(s)


def load_settings(path: Path | None = None) -> Settings:
    p = path or _config_path()
    with p.open("rb") as f:
        raw = tomllib.load(f)
    settings = Settings.model_validate(raw)

    # Step 1: expand `~` and `${VAR}` so config.toml is portable across
    # machines (same file, different home dir).
    settings.paths.data_dir = _expand(settings.paths.data_dir)
    settings.paths.log_dir = _expand(settings.paths.log_dir)
    settings.paths.roots = [_expand(r) for r in settings.paths.roots]

    # Step 2: resolve relative paths against the config file's parent dir so
    # `./data` in config.toml always means `<repo_root>/data` regardless of CWD.
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

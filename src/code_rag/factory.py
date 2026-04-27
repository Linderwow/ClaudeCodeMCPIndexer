"""Single place where concrete implementations are assembled from Settings.

Swapping an embedder, vector store, lexical store, graph store, or reranker
means editing exactly one function here.
"""
from __future__ import annotations

from code_rag.config import Settings
from code_rag.embedders.lm_studio import LMStudioEmbedder
from code_rag.interfaces.embedder import Embedder
from code_rag.interfaces.graph_store import GraphStore
from code_rag.interfaces.lexical_store import LexicalStore
from code_rag.interfaces.reranker import Reranker
from code_rag.interfaces.vector_store import VectorStore
from code_rag.rerankers.lm_chat import LMStudioChatReranker
from code_rag.rerankers.lm_studio import LMStudioReranker
from code_rag.rerankers.noop import NoopReranker
# Cross-encoder reranker is OPTIONAL (requires sentence-transformers). We
# import lazily inside build_reranker so the factory itself never fails to
# import even if the optional dep isn't installed.
from code_rag.stores.chroma_vector import ChromaVectorStore
from code_rag.stores.kuzu_graph import KuzuGraphStore
from code_rag.stores.sqlite_lexical import SqliteLexicalStore


def build_embedder(settings: Settings) -> Embedder:
    e = settings.embedder
    if e.kind == "lm_studio":
        # Phase 17: support `[embedder].preset = "<name>"` as a shorthand for
        # one of the curated code-specialized presets (BGE-Code-v1, CodeSage,
        # Qwen baseline). Explicit `model = "..."` always wins; preset only
        # fires when set AND model is left at its default.
        preset = getattr(e, "preset", None)
        if preset:
            from code_rag.embedders.code_specialized import (
                CODE_EMBEDDER_PRESETS,
                build_code_embedder,
            )
            if preset in CODE_EMBEDDER_PRESETS:
                return build_code_embedder(
                    preset, base_url=e.base_url, timeout_s=e.timeout_s, batch=e.batch,
                )
        return LMStudioEmbedder(
            base_url=e.base_url,
            model=e.model,
            dim=e.dim,
            timeout_s=e.timeout_s,
            batch=e.batch,
        )
    raise ValueError(f"Unknown embedder kind: {e.kind}")


def build_vector_store(settings: Settings) -> VectorStore:
    v = settings.vector_store
    if v.kind == "chroma":
        return ChromaVectorStore(
            persist_dir=settings.chroma_dir,
            collection=v.collection,
            meta_path=settings.index_meta_path,
        )
    raise ValueError(f"Unknown vector store kind: {v.kind}")


def build_lexical_store(settings: Settings) -> LexicalStore:
    lx = settings.lexical_store
    if lx.kind == "sqlite_fts5":
        return SqliteLexicalStore(db_path=settings.fts_path)
    raise ValueError(f"Unknown lexical store kind: {lx.kind}")


def build_reranker(settings: Settings) -> Reranker:
    r = settings.reranker
    if r.kind == "lm_studio":
        return LMStudioReranker(
            base_url=r.base_url,
            model=r.model,
            timeout_s=r.timeout_s,
        )
    if r.kind == "lm_chat":
        return LMStudioChatReranker(
            base_url=r.base_url,
            model=r.model,
            timeout_s=r.timeout_s,
            max_candidates=r.chat_max_candidates,
            max_chars_per_doc=r.chat_max_chars_per_doc,
        )
    if r.kind == "cross_encoder":
        # Phase 29: lazy import so the factory works even when the optional
        # dep isn't installed. Construction itself raises a clear ImportError
        # if sentence-transformers is missing.
        from code_rag.rerankers.cross_encoder import CrossEncoderReranker
        return CrossEncoderReranker(
            model=r.model,
            max_candidates=r.top_k_in,
            max_chars_per_doc=r.cross_encoder_max_chars,
            device=r.device,
        )
    if r.kind == "noop":
        return NoopReranker()
    raise ValueError(f"Unknown reranker kind: {r.kind}")


def build_query_rewriter(settings: Settings) -> object | None:
    """Build the Phase 30 query rewriter, or return None if disabled.

    Returns an `LMStudioQueryRewriter` instance with cache attached. The cache
    is opened here (cheap, just creates the SQLite file). The MCP server /
    CLI is responsible for closing it on shutdown.
    """
    qr = settings.query_rewriter
    if not qr.enabled:
        return None
    from code_rag.retrieval.query_rewriter import (
        LMStudioQueryRewriter,
        RewriteCache,
    )
    cache: RewriteCache | None = None
    if qr.cache:
        cache = RewriteCache(settings.rewrite_cache_path)
        cache.open()
    base_url = qr.base_url or settings.embedder.base_url
    return LMStudioQueryRewriter(
        base_url=base_url,
        model=qr.model or "",
        cache=cache,
        timeout_s=qr.timeout_s,
    )


def build_graph_store(settings: Settings, *, read_only: bool = False) -> GraphStore:
    """Build a graph store.

    Pass `read_only=True` when the caller only needs to query (MCP tool
    handlers, CLI `callers`/`callees`/`symbol` commands). That lets the
    process coexist with a long-running writer (e.g. the autostart watcher
    holding the exclusive write lock).
    """
    g = settings.graph_store
    if g.kind == "kuzu":
        return KuzuGraphStore(db_dir=settings.kuzu_dir, read_only=read_only)
    raise ValueError(f"Unknown graph store kind: {g.kind}")

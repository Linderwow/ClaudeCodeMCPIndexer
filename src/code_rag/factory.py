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
from code_rag.rerankers.lm_studio import LMStudioReranker
from code_rag.stores.chroma_vector import ChromaVectorStore
from code_rag.stores.kuzu_graph import KuzuGraphStore
from code_rag.stores.sqlite_lexical import SqliteLexicalStore


def build_embedder(settings: Settings) -> Embedder:
    e = settings.embedder
    if e.kind == "lm_studio":
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
    raise ValueError(f"Unknown reranker kind: {r.kind}")


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

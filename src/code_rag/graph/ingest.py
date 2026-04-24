"""Adapter that turns a file's extracted symbols+edges into graph-store calls.

Two construction modes:

  - `GraphIngester(store)` -- the classic mode used by tests and the CLI
    `index` command. The caller owns the store lifecycle and opens it once.

  - `GraphIngester.transient(graph_dir)` -- used by the live watcher. Each
    ingest() / delete_by_path() call opens Kuzu, writes, and closes. Kuzu 0.11
    takes a process-exclusive OS file lock even in read-only mode, so holding
    the graph DB open for the life of the watcher would lock out the MCP
    server (which Claude Code spawns per session). Open-close per event keeps
    the write window tiny (~tens of ms) and lets readers coexist almost always.
"""
from __future__ import annotations

from pathlib import Path

from code_rag.graph.extractor import GraphExtractor
from code_rag.interfaces.graph_store import GraphStore
from code_rag.stores.kuzu_graph import KuzuGraphStore


class GraphIngester:
    """Thin glue: parse file -> extract symbols+edges -> upsert into GraphStore."""

    def __init__(self, store: GraphStore) -> None:
        self._store: GraphStore | None = store
        self._transient_dir: Path | None = None
        self._extractor = GraphExtractor()

    @classmethod
    def transient(cls, graph_dir: Path) -> GraphIngester:
        """Open-close Kuzu per write operation. Use from long-lived processes
        (the watcher) that must not hold the DB lock between events."""
        ing = cls.__new__(cls)
        ing._store = None
        ing._transient_dir = graph_dir
        ing._extractor = GraphExtractor()
        return ing

    def delete_by_path(self, path: str) -> None:
        store = self._borrow_store()
        try:
            store.delete_by_path(path)
        finally:
            self._return_store(store)

    def ingest(self, abs_path: Path, rel_path: str, language: str) -> None:
        symbols, edges = self._extractor.extract(abs_path, rel_path, language)
        if not symbols and not edges:
            # Nothing to write — skip acquiring the lock entirely.
            return
        store = self._borrow_store()
        try:
            store.upsert_symbols(symbols)
            store.upsert_edges(edges)
        finally:
            self._return_store(store)

    # ---- internals ----------------------------------------------------------

    def _borrow_store(self) -> GraphStore:
        if self._store is not None:
            return self._store
        assert self._transient_dir is not None
        store = KuzuGraphStore(self._transient_dir, read_only=False)
        store.open()
        return store

    def _return_store(self, store: GraphStore) -> None:
        # If we own the store lifecycle (transient mode), close it so the
        # lock is released between events.
        if self._store is None:
            store.close()

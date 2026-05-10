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

    # Phase 60-D split: `ingest()` used to do {parse + extract + DB write} in
    # one synchronous call. The parse step is CPU-heavy (5-30s on huge
    # auto-generated files like EF Core migrations) and was historically
    # held INSIDE the indexer's writer lock — that serialized parsing across
    # all workers and starved the GPU embedder while one big file was being
    # parsed. Splitting it lets the indexer run `extract()` in a thread pool
    # outside the lock, so multiple workers parse in parallel; only the
    # short DB-write `commit()` stays inside the lock.
    #
    # Old `ingest()` is kept as a thin convenience wrapper for the watcher
    # path, which open-closes Kuzu per event anyway.

    def extract(
        self, abs_path: Path, rel_path: str, language: str,
    ) -> tuple[list, list]:
        """Parse + extract symbols/edges only. NO DB writes — pure CPU work,
        safe to run outside any writer lock and in `asyncio.to_thread`.
        Returns (symbols, edges); pass straight into `commit()`."""
        return self._extractor.extract(abs_path, rel_path, language)

    def commit(self, symbols: list, edges: list) -> None:
        """Write pre-extracted symbols/edges to the graph store. Fast (DB
        writes only) — call this INSIDE the indexer's writer lock so writes
        across stores stay consistent."""
        if not symbols and not edges:
            # Nothing to write — skip acquiring the store entirely.
            return
        store = self._borrow_store()
        try:
            store.upsert_symbols(symbols)
            store.upsert_edges(edges)
        finally:
            self._return_store(store)

    def replace_for_path(
        self, path: str, symbols: list, edges: list,
    ) -> None:
        """Phase 60-F audit fix: atomic per-path replace (delete + upsert) at
        the graph-store level. Use this instead of separate `delete_by_path`
        + `commit` when those calls would otherwise straddle a released
        asyncio lock — the kuzu store's RLock guarantees the whole replace
        happens under one acquisition, so two workers reindexing the same
        path can't interleave their writes."""
        store = self._borrow_store()
        try:
            # KuzuGraphStore exposes replace_for_path; other backends may
            # not — fall back to delete + upsert under whatever locking
            # they offer.
            if hasattr(store, "replace_for_path"):
                store.replace_for_path(path, symbols, edges)  # type: ignore[attr-defined]
            else:
                store.delete_by_path(path)
                if symbols or edges:
                    store.upsert_symbols(symbols)
                    store.upsert_edges(edges)
        finally:
            self._return_store(store)

    def ingest(self, abs_path: Path, rel_path: str, language: str) -> None:
        """Single-call parse-and-commit. Kept for the watcher path (which
        is single-file-at-a-time and re-opens Kuzu per event so the lock
        story is moot). New bulk-indexer code should call `extract()` +
        `commit()` separately so the parse runs outside the writer lock."""
        symbols, edges = self.extract(abs_path, rel_path, language)
        self.commit(symbols, edges)

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

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import kuzu

from code_rag.interfaces.graph_store import Edge, GraphStore, SymbolRef
from code_rag.logging import get

log = get(__name__)


class KuzuGraphStore(GraphStore):
    """Kuzu-backed symbol + edge graph.

    Schema:
        Symbol(path STRING, symbol STRING, kind STRING, start_line INT64, end_line INT64)
            PRIMARY KEY(path, symbol)
        Calls    FROM Symbol TO Symbol  (dst_path may be NULL-ish; stored as "")
        Contains FROM Symbol TO Symbol
        Imports  FROM Symbol TO Symbol  (external edges get a synthetic dst)

    Kuzu requires edge endpoints to be real nodes. For unresolved callees we
    create lightweight "ghost" Symbol rows with path='<extern>' and kind='extern'
    so get_callers still returns something actionable.
    """

    EXTERN_PATH = "<extern>"

    def __init__(self, db_dir: Path, *, read_only: bool = False) -> None:
        # `db_dir` is a conceptual name — kuzu 0.11 treats the path as a single
        # file (it manages its own WAL/data files adjacent to it). We append a
        # `.kz` suffix and ensure the PARENT directory exists.
        #
        # read_only=True is required when another process (the autostart
        # watcher) already holds the DB's exclusive write lock. Kuzu's
        # concurrency model is single-writer + multiple-readers.
        self._path = db_dir.with_suffix(".kz") if db_dir.suffix != ".kz" else db_dir
        self._read_only = read_only
        self._db: Any = None
        self._conn: Any = None
        self._lock = threading.RLock()

    # ---- lifecycle ----------------------------------------------------------

    def open(self, *, retry_seconds: float = 3.0, retry_interval_s: float = 0.1) -> None:
        """Open the Kuzu database.

        Kuzu 0.11 takes an OS-level file lock even in read-only mode, so
        concurrent open() calls can transiently fail when another process is
        currently holding the lock. We retry for `retry_seconds` with small
        backoff before surfacing the error. This covers the common case of a
        reader colliding with a watcher mid-write — such windows are sub-second.
        """
        import time as _time
        self._path.parent.mkdir(parents=True, exist_ok=True)
        deadline = _time.monotonic() + retry_seconds
        last_err: Exception | None = None
        while True:
            try:
                self._db = kuzu.Database(str(self._path), read_only=self._read_only)
                self._conn = kuzu.Connection(self._db)
                break
            except RuntimeError as e:
                last_err = e
                if "lock" not in str(e).lower():
                    raise
                if _time.monotonic() >= deadline:
                    raise
                _time.sleep(retry_interval_s)
        if not self._read_only:
            self._init_schema()
        log.info("kuzu.opened", path=str(self._path),
                 read_only=self._read_only, retried=last_err is not None)

    def close(self) -> None:
        # Phase 37 audit fix: take the lock before nulling so a concurrent
        # `_run` can't observe `self._conn=None` mid-tear-down and crash
        # with `AttributeError: 'NoneType' object has no attribute 'execute'`.
        # Realistic race: shutdown path or fsck calling close() while a
        # search thread is in flight on the read connection.
        with self._lock:
            self._conn = None
            self._db = None

    def _init_schema(self) -> None:
        stmts = [
            # Kuzu requires a single-column primary key. We synthesize `id` from
            # (path, symbol) with a `|` delimiter — neither occurs in real paths
            # or symbol names in our languages.
            "CREATE NODE TABLE IF NOT EXISTS Symbol("
            "  id STRING PRIMARY KEY, path STRING, symbol STRING, kind STRING,"
            "  start_line INT64, end_line INT64"
            ")",
            "CREATE REL TABLE IF NOT EXISTS Calls   (FROM Symbol TO Symbol, src_path STRING)",
            "CREATE REL TABLE IF NOT EXISTS Contains(FROM Symbol TO Symbol, src_path STRING)",
            "CREATE REL TABLE IF NOT EXISTS Imports (FROM Symbol TO Symbol, src_path STRING)",
        ]
        for s in stmts:
            self._run(s)

    @staticmethod
    def _node_id(path: str, symbol: str) -> str:
        return f"{path}|{symbol}"

    # ---- writes -------------------------------------------------------------

    # Phase 60-E: batched UNWIND inserts. The previous per-row CREATE pattern
    # ran ~32 ms per row (Kuzu transaction setup dominates the actual insert
    # cost on small writes). For files with 400+ symbols/edges that worked out
    # to 13+ s of writer-lock holding per file. UNWIND-with-batch runs the
    # whole list in one transaction at ~0.06 ms / row — a 548× speedup measured
    # on the Phase 60-E microbenchmark.
    #
    # Correctness contract preserved: `upsert_symbols` is always preceded by
    # `delete_by_path(rel)` in the indexer, so all rows are guaranteed-new and
    # plain batched CREATE is safe. `upsert_edges` may reference cross-file
    # endpoints that already exist (extern symbols, calls into another file's
    # symbol), so we pre-query existing IDs and batch-CREATE only the missing
    # ones before the edge writes.

    def upsert_symbols(self, symbols: Sequence[SymbolRef]) -> None:
        if not symbols:
            return
        rows = [
            {
                "id":  self._node_id(s.path, s.symbol),
                "p":   s.path,
                "sym": s.symbol,
                "k":   s.kind,
                "sl":  s.start_line,
                "el":  s.end_line,
            }
            for s in symbols
        ]
        with self._lock:
            try:
                self._run(
                    """UNWIND $rows AS row
                       CREATE (s:Symbol {
                         id: row.id, path: row.p, symbol: row.sym,
                         kind: row.k, start_line: row.sl, end_line: row.el
                       })""",
                    {"rows": rows},
                )
            except RuntimeError as e:
                msg = str(e).lower()
                if "already exists" in msg or "duplicate" in msg or "primary key" in msg:
                    # Unhappy path: indexer's delete-by-path didn't fully purge
                    # (concurrent watcher write, lock retry), or a stray prior
                    # state. Fall back to the original per-row merge that does
                    # CREATE-or-UPDATE for each. Slow but correct.
                    log.warning("kuzu.upsert_symbols.batch_collision_fallback",
                                n=len(symbols), err=str(e)[:200])
                    for s in symbols:
                        self._merge_symbol(s.path, s.symbol, s.kind,
                                           s.start_line, s.end_line)
                else:
                    raise

    def upsert_edges(self, edges: Sequence[Edge]) -> None:
        if not edges:
            return
        # Phase 60-E: collect all distinct endpoint IDs that must exist before
        # we can MATCH-and-CREATE the edges. We then bulk-create only the
        # IDs that don't already exist, in a single UNWIND. By edge type
        # (Calls/Contains/Imports) we batch the relationship CREATE separately
        # because the relationship table name has to be a literal in Cypher.
        endpoint_rows: dict[str, dict[str, Any]] = {}
        for e in edges:
            src_id = self._node_id(e.src_path, e.src_symbol)
            dst_path = e.dst_path if e.dst_path else self.EXTERN_PATH
            dst_id = self._node_id(dst_path, e.dst_symbol)
            if src_id not in endpoint_rows:
                endpoint_rows[src_id] = {
                    "id": src_id, "p": e.src_path, "sym": e.src_symbol,
                    "k": "unknown", "sl": 0, "el": 0,
                }
            if dst_id not in endpoint_rows:
                endpoint_rows[dst_id] = {
                    "id": dst_id, "p": dst_path, "sym": e.dst_symbol,
                    "k": "extern" if dst_path == self.EXTERN_PATH else "unknown",
                    "sl": 0, "el": 0,
                }

        with self._lock:
            # Step 1: ensure all endpoint nodes exist. Query existing IDs in
            # one round-trip, then batch-CREATE the missing ones.
            if endpoint_rows:
                ids = list(endpoint_rows.keys())
                existing_rows = self._run(
                    "MATCH (s:Symbol) WHERE s.id IN $ids RETURN s.id",
                    {"ids": ids},
                )
                existing = {str(r[0]) for r in existing_rows if r}
                new_rows = [r for sid, r in endpoint_rows.items() if sid not in existing]
                if new_rows:
                    try:
                        self._run(
                            """UNWIND $rows AS row
                               CREATE (s:Symbol {
                                 id: row.id, path: row.p, symbol: row.sym,
                                 kind: row.k, start_line: row.sl, end_line: row.el
                               })""",
                            {"rows": new_rows},
                        )
                    except RuntimeError as e:
                        msg = str(e).lower()
                        if "already exists" in msg or "duplicate" in msg or "primary key" in msg:
                            # Race with a concurrent writer. Retry per-row.
                            log.warning("kuzu.upsert_edges.endpoint_collision_fallback",
                                        n=len(new_rows), err=str(e)[:200])
                            for r in new_rows:
                                self._merge_symbol(r["p"], r["sym"], r["k"],
                                                   r["sl"], r["el"],
                                                   only_if_missing=True)
                        else:
                            raise

            # Step 2: group edges by relationship table and batch-CREATE.
            buckets: dict[str, list[dict[str, Any]]] = {}
            for e in edges:
                rel_table = {"calls": "Calls", "contains": "Contains",
                             "imports": "Imports"}.get(e.kind, "Calls")
                dst_path = e.dst_path if e.dst_path else self.EXTERN_PATH
                buckets.setdefault(rel_table, []).append({
                    "sid": self._node_id(e.src_path, e.src_symbol),
                    "did": self._node_id(dst_path, e.dst_symbol),
                    "sp":  e.src_path,
                })
            for rel_table, rows in buckets.items():
                self._run(
                    f"""UNWIND $rows AS row
                        MATCH (a:Symbol {{id: row.sid}}), (b:Symbol {{id: row.did}})
                        CREATE (a)-[:{rel_table} {{src_path: row.sp}}]->(b)""",
                    {"rows": rows},
                )

    def delete_by_path(self, path: str) -> None:
        with self._lock:
            # Detach-delete nuke every symbol whose src_path == this path AND
            # every edge whose src_path == this path.
            self._run("MATCH (:Symbol)-[r {src_path: $p}]->(:Symbol) DELETE r", {"p": path})
            self._run("MATCH (s:Symbol {path: $p}) DETACH DELETE s", {"p": path})

    def replace_for_path(
        self, path: str, symbols: Sequence[SymbolRef], edges: Sequence[Edge],
    ) -> None:
        """Phase 60-F audit fix: atomic delete-by-path + upsert under a single
        RLock acquisition. Without this, the indexer's split (delete inside
        asyncio lock, commit outside) lets two workers interleave on the same
        path: A.delete → B.delete → A.commit → B.commit can leave A's stale
        rows behind because B's delete preceded A's upsert. Doing both under
        the kuzu RLock guarantees per-path atomicity even when the asyncio
        writer lock is released between them.
        """
        with self._lock:
            # Same delete behavior as `delete_by_path`, inlined so we don't
            # release+reacquire the RLock.
            self._run(
                "MATCH (:Symbol)-[r {src_path: $p}]->(:Symbol) DELETE r",
                {"p": path},
            )
            self._run(
                "MATCH (s:Symbol {path: $p}) DETACH DELETE s",
                {"p": path},
            )
            # Now upsert. If both lists are empty, no-op; commit early.
            if not symbols and not edges:
                return
            # Inline the upsert work (without re-acquiring RLock) — call
            # internal helpers that don't take the lock again. The public
            # upsert_symbols / upsert_edges acquire `with self._lock` which
            # is fine because RLock is reentrant, but we keep this simple.
            self.upsert_symbols(symbols)
            self.upsert_edges(edges)

    # ---- reads --------------------------------------------------------------

    def callers_of(self, symbol: str, path: str | None = None) -> list[SymbolRef]:
        with self._lock:
            if path:
                q = ("MATCH (a:Symbol)-[:Calls]->(b:Symbol {path: $p, symbol: $s}) "
                     "RETURN a.path, a.symbol, a.kind, a.start_line, a.end_line")
                params: dict[str, Any] = {"s": symbol, "p": path}
            else:
                # Unresolved callee name: match either resolved (path=target path) or extern.
                q = ("MATCH (a:Symbol)-[:Calls]->(b:Symbol) WHERE b.symbol = $s "
                     "RETURN a.path, a.symbol, a.kind, a.start_line, a.end_line")
                params = {"s": symbol}
            return self._rows_to_refs(self._run(q, params))

    def callees_of(self, symbol: str, path: str | None = None) -> list[SymbolRef]:
        with self._lock:
            if path:
                q = ("MATCH (a:Symbol {path: $p, symbol: $s})-[:Calls]->(b:Symbol) "
                     "RETURN b.path, b.symbol, b.kind, b.start_line, b.end_line")
                params: dict[str, Any] = {"s": symbol, "p": path}
            else:
                q = ("MATCH (a:Symbol)-[:Calls]->(b:Symbol) WHERE a.symbol = $s "
                     "RETURN b.path, b.symbol, b.kind, b.start_line, b.end_line")
                params = {"s": symbol}
            return self._rows_to_refs(self._run(q, params))

    def find_symbol(self, symbol: str, path: str | None = None) -> list[SymbolRef]:
        with self._lock:
            if path:
                q = ("MATCH (s:Symbol {symbol: $s, path: $p}) "
                     "RETURN s.path, s.symbol, s.kind, s.start_line, s.end_line")
                params: dict[str, Any] = {"s": symbol, "p": path}
            else:
                q = ("MATCH (s:Symbol) WHERE s.symbol = $s OR s.symbol ENDS WITH $endswith "
                     "RETURN s.path, s.symbol, s.kind, s.start_line, s.end_line")
                params = {"s": symbol, "endswith": f".{symbol}"}
            return self._rows_to_refs(self._run(q, params))

    # ---- internals ----------------------------------------------------------

    def _merge_symbol(
        self, path: str, symbol: str, kind: str, start_line: int, end_line: int,
        *, only_if_missing: bool = False,
    ) -> None:
        sid = self._node_id(path, symbol)
        if only_if_missing:
            rows = self._run(
                "MATCH (s:Symbol {id: $id}) RETURN 1 LIMIT 1", {"id": sid},
            )
            if rows:
                return
        try:
            self._run(
                "CREATE (s:Symbol {id: $id, path: $p, symbol: $sym, kind: $k, "
                "                  start_line: $sl, end_line: $el})",
                {"id": sid, "p": path, "sym": symbol, "k": kind,
                 "sl": start_line, "el": end_line},
            )
        except RuntimeError as e:
            msg = str(e).lower()
            if "already exists" in msg or "duplicate" in msg or "primary key" in msg:
                if not only_if_missing:
                    self._run(
                        "MATCH (s:Symbol {id: $id}) SET s.kind = $k, "
                        "  s.start_line = $sl, s.end_line = $el",
                        {"id": sid, "k": kind, "sl": start_line, "el": end_line},
                    )
                return
            raise

    def _run(self, cypher: str, params: dict[str, Any] | None = None) -> list[list[Any]]:
        if self._conn is None:
            raise RuntimeError("KuzuGraphStore not open")
        result = self._conn.execute(cypher, parameters=params or {})
        rows: list[list[Any]] = []
        # Kuzu QueryResult API: has_next()/get_next() returning list[Any].
        try:
            while result.has_next():
                rows.append(cast(list[Any], result.get_next()))
        except Exception as e:  # pragma: no cover — defensive; Kuzu raises on misuse
            # Log and return partial rows rather than crashing the MCP handler.
            # Silent swallow would hide real query / driver bugs as empty results.
            log.warning("kuzu.run.partial_iteration",
                        err=str(e), partial_rows=len(rows),
                        cypher=cypher[:120])
        return rows

    @staticmethod
    def _rows_to_refs(rows: list[list[Any]]) -> list[SymbolRef]:
        out: list[SymbolRef] = []
        for r in rows:
            if len(r) < 5:
                continue
            out.append(SymbolRef(
                path=str(r[0]),
                symbol=str(r[1]),
                kind=str(r[2]),
                start_line=int(r[3]),
                end_line=int(r[4]),
            ))
        return out

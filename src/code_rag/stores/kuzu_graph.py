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

    def upsert_symbols(self, symbols: Sequence[SymbolRef]) -> None:
        if not symbols:
            return
        with self._lock:
            for s in symbols:
                self._merge_symbol(s.path, s.symbol, s.kind, s.start_line, s.end_line)

    def upsert_edges(self, edges: Sequence[Edge]) -> None:
        if not edges:
            return
        with self._lock:
            for e in edges:
                # Resolve dst: if dst_path is None, synthesize an extern node.
                dst_path = e.dst_path if e.dst_path else self.EXTERN_PATH
                # Ensure both endpoints exist (MERGE semantics via our helper).
                self._merge_symbol(e.src_path, e.src_symbol, "unknown", 0, 0, only_if_missing=True)
                self._merge_symbol(dst_path, e.dst_symbol, "extern" if dst_path == self.EXTERN_PATH
                                   else "unknown", 0, 0, only_if_missing=True)
                rel_table = {"calls": "Calls", "contains": "Contains", "imports": "Imports"}.get(
                    e.kind, "Calls"
                )
                self._run(
                    f"""MATCH (a:Symbol {{id: $sid}}), (b:Symbol {{id: $did}})
                        CREATE (a)-[:{rel_table} {{src_path: $sp}}]->(b)""",
                    {
                        "sid": self._node_id(e.src_path, e.src_symbol),
                        "did": self._node_id(dst_path, e.dst_symbol),
                        "sp":  e.src_path,
                    },
                )

    def delete_by_path(self, path: str) -> None:
        with self._lock:
            # Detach-delete nuke every symbol whose src_path == this path AND
            # every edge whose src_path == this path.
            self._run("MATCH (:Symbol)-[r {src_path: $p}]->(:Symbol) DELETE r", {"p": path})
            self._run("MATCH (s:Symbol {path: $p}) DETACH DELETE s", {"p": path})

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

from __future__ import annotations

import re
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

from code_rag.interfaces.lexical_store import LexicalStore
from code_rag.logging import get
from code_rag.models import Chunk, ChunkKind, SearchHit

log = get(__name__)


# FTS5 unicode61 tokenizer splits on `_`, so `MNQAlpha_V91_compute` tokenizes
# to [mnqalpha, v91, compute]. Queries are split the same way, so BM25 still
# scores identifier hits highly. We keep the default tokenizer for simplicity;
# nested-quote escaping for `tokenchars` in CREATE VIRTUAL TABLE is fiddly and
# the recall gain is marginal with RRF fusion on top.

# FTS5 meta-chars we strip before tokenizing a free-form query. The remaining
# tokens are wrapped in double-quotes when we build the MATCH expression — that
# neutralizes FTS5 keywords (AND, OR, NOT, NEAR) and bracket syntax so a query
# like `NEAR the AND gate` doesn't blow up the parser.
_BAD_CHARS = re.compile(r'["()*:^{}]')


class SqliteLexicalStore(LexicalStore):
    """BM25-backed identifier recall. Sibling to the vector store.

    Schema:
      chunks(id PRIMARY KEY, path, symbol, language, kind, start_line, end_line, text)
      chunks_fts (virtual, content='chunks', BM25 over (symbol, text))
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        # The indexer runs chunking + embedding on thread pools and then issues
        # upserts back to this store; the searcher runs lexical queries via
        # asyncio.to_thread. A single connection shared across threads is OK as
        # long as we serialize access with this lock.
        self._lock = threading.RLock()

    def open(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self._db_path, isolation_level=None, check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id         TEXT PRIMARY KEY,
                repo       TEXT NOT NULL,
                path       TEXT NOT NULL,
                symbol     TEXT,
                language   TEXT NOT NULL,
                kind       TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line   INTEGER NOT NULL,
                text       TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS chunks_path_idx ON chunks(path);

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                symbol, text,
                content='chunks', content_rowid='rowid',
                tokenize='unicode61 remove_diacritics 2'
            );

            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, symbol, text) VALUES (new.rowid, new.symbol, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, symbol, text)
                VALUES ('delete', old.rowid, old.symbol, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, symbol, text)
                VALUES ('delete', old.rowid, old.symbol, old.text);
                INSERT INTO chunks_fts(rowid, symbol, text) VALUES (new.rowid, new.symbol, new.text);
            END;
            """
        )
        log.info("lexical.opened", db=str(self._db_path), count=self.count())

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ---- writes -------------------------------------------------------------

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        conn = self._require()
        rows = [
            (c.id, c.repo, c.path, c.symbol, c.language, c.kind.value,
             c.start_line, c.end_line, c.text)
            for c in chunks
        ]
        with self._lock, conn:
            conn.executemany(
                """INSERT INTO chunks(id, repo, path, symbol, language, kind, start_line, end_line, text)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       repo=excluded.repo,
                       path=excluded.path,
                       symbol=excluded.symbol,
                       language=excluded.language,
                       kind=excluded.kind,
                       start_line=excluded.start_line,
                       end_line=excluded.end_line,
                       text=excluded.text""",
                rows,
            )

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        conn = self._require()
        with self._lock, conn:
            conn.executemany("DELETE FROM chunks WHERE id = ?", [(i,) for i in ids])

    def delete_by_path(self, path: str) -> int:
        conn = self._require()
        with self._lock, conn:
            cur = conn.execute("DELETE FROM chunks WHERE path = ?", (path,))
            return cur.rowcount if cur.rowcount is not None else 0

    # ---- reads --------------------------------------------------------------

    def query(self, text: str, k: int) -> list[SearchHit]:
        conn = self._require()
        match = self._to_fts_query(text)
        if not match:
            return []
        with self._lock:
            rows = conn.execute(
                """SELECT c.id, c.repo, c.path, c.symbol, c.language, c.kind,
                          c.start_line, c.end_line, c.text, bm25(chunks_fts) AS score
                   FROM chunks_fts
                   JOIN chunks c ON c.rowid = chunks_fts.rowid
                   WHERE chunks_fts MATCH ?
                   ORDER BY score ASC
                   LIMIT ?""",
                (match, k),
            ).fetchall()
        hits: list[SearchHit] = []
        # BM25 in FTS5 is lower = better; normalize to [0,1] descending.
        # We don't try to calibrate absolutely; RRF fusion handles that.
        max_rank = len(rows)
        # Preview the tokens the FTS matcher saw; helps debug misses.
        tokens = self._preview_tokens(text)
        reason = f"lexical bm25 matched tokens {tokens!r}"
        for rank, (cid, repo, path, symbol, lang, kind, sl, el, text_, _score) in enumerate(rows):
            # Synthetic monotone score for display; RRF only uses rank anyway.
            disp = 1.0 - (rank / max(1, max_rank))
            hits.append(SearchHit(
                chunk=Chunk(
                    id=cid, repo=repo, path=path, language=lang,
                    symbol=symbol or None,
                    kind=ChunkKind(kind),
                    start_line=int(sl), end_line=int(el),
                    text=text_,
                ),
                score=disp,
                source="lexical",
                match_reason=reason,
            ))
        return hits

    @staticmethod
    def _preview_tokens(text: str) -> list[str]:
        """Show the user what tokens the FTS matcher actually saw. Mirrors
        the cleanup done in _to_fts_query so reasons don't drift."""
        cleaned = _BAD_CHARS.sub(" ", text).strip()
        return [t for t in cleaned.split() if t][:8]

    def count(self) -> int:
        with self._lock:
            row = self._require().execute("SELECT COUNT(*) FROM chunks").fetchone()
            return int(row[0]) if row else 0

    def list_paths(self) -> set[str]:
        """Distinct path values across all stored chunks. Cheap thanks to the
        `chunks_path_idx` index on the chunks table."""
        with self._lock:
            rows = self._require().execute(
                "SELECT DISTINCT path FROM chunks"
            ).fetchall()
        return {str(r[0]) for r in rows}

    # ---- internals ----------------------------------------------------------

    def _require(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SqliteLexicalStore not open")
        return self._conn

    @staticmethod
    def _to_fts_query(text: str) -> str:
        """Convert free-form user input to an FTS5 match expression.

        Strategy:
          - Strip FTS5 meta-chars.
          - Split on whitespace.
          - Wrap each token in double-quotes so FTS5 keywords (AND, OR, NOT,
            NEAR) and odd tokens can't break the parser, then apply a prefix
            `*` outside the quotes for identifier-partial hits.
          - Tokens OR'd together — a single rare token (e.g., MNQAlpha_V91) still
            surfaces the file even if other terms miss.
        """
        cleaned = _BAD_CHARS.sub(" ", text).strip()
        if not cleaned:
            return ""
        toks = [t for t in cleaned.split() if t]
        # `"token"*` = prefix match on the quoted phrase; quoting defuses any
        # keyword interpretation FTS5 would otherwise apply.
        return " OR ".join(f'"{t}"*' for t in toks[:16])

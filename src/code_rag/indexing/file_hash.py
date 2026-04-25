"""Per-file content hash registry — Phase 14 incremental indexing.

The indexer's old behaviour was: every walker tick → re-parse every file →
re-chunk → re-embed (per chunk hash dedup at the chunk_id layer skipped the
upsert, but parsing and embedding still ran). For a 12k-file repo on
re-index, that's 4+ hours wasted re-doing identical work.

This module persists a tiny side-store: `path -> blake3(content)`. Before
parsing a file, the indexer asks "did this content change since last
indexed?" If no, the parse step is skipped entirely. If yes, we re-parse
and update the hash.

Storage: a single SQLite file at `<data_dir>/file_hashes.db`. Same lock
semantics as the lexical store. Tiny — a row per file, no chunk-level
data — so the cost of consulting it is negligible vs the cost saved.

Intentional simplifications
---------------------------
* mtime is **not** part of the key. We always read content and hash it.
  Filesystem mtimes lie across editors / sync tools / VCS checkouts; a
  content hash is the only safe oracle. Reading a file is fast (sequential
  disk read); the savings come from skipping `tree-sitter parse` + the
  per-chunk pipeline downstream.
* We hash the file's RAW BYTES, not its decoded text. Saves a UTF-8 round
  trip and protects against text-decoding ambiguity.
* On any error reading the registry, the indexer falls back to "treat as
  changed" — we never silently skip indexing.
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from blake3 import blake3

from code_rag.logging import get

log = get(__name__)


def hash_bytes(data: bytes) -> str:
    """Stable content fingerprint. blake3 is faster than sha256 and we
    already have it as a dep for chunk_id."""
    return blake3(data).hexdigest(length=16)


class FileHashRegistry:
    """Tiny SQLite-backed map of `rel_path -> content_hash`.

    Used by the indexer to short-circuit the per-file pipeline when nothing
    has changed since the last successful index. Keyed by the SAME `rel_path`
    the rest of the system uses (relative to the matching root), so the
    registry survives root reorders / renames as long as the file's
    relative-from-root path stays stable.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
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
            CREATE TABLE IF NOT EXISTS file_hashes (
                rel_path TEXT PRIMARY KEY,
                hash     TEXT NOT NULL,
                size     INTEGER NOT NULL,
                indexed_at REAL NOT NULL
            );
            """
        )
        log.info("file_hash.opened", db=str(self._db_path), count=self.count())

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def count(self) -> int:
        with self._lock:
            row = self._require().execute(
                "SELECT COUNT(*) FROM file_hashes"
            ).fetchone()
            return int(row[0]) if row else 0

    def get(self, rel_path: str) -> str | None:
        """Return the stored hash for `rel_path`, or None if unknown."""
        with self._lock:
            row = self._require().execute(
                "SELECT hash FROM file_hashes WHERE rel_path = ?", (rel_path,),
            ).fetchone()
            return str(row[0]) if row else None

    def is_unchanged(self, rel_path: str, content: bytes) -> bool:
        """True iff we've indexed this rel_path before AND its content hash
        matches `content`. The indexer uses this as the skip-fast guard."""
        stored = self.get(rel_path)
        if stored is None:
            return False
        return stored == hash_bytes(content)

    def upsert(self, rel_path: str, content: bytes) -> str:
        """Record (rel_path, hash). Returns the hash we just stored."""
        import time as _t
        h = hash_bytes(content)
        with self._lock, self._require():
            self._require().execute(
                """INSERT INTO file_hashes(rel_path, hash, size, indexed_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(rel_path) DO UPDATE SET
                       hash = excluded.hash,
                       size = excluded.size,
                       indexed_at = excluded.indexed_at""",
                (rel_path, h, len(content), _t.time()),
            )
        return h

    def delete(self, rel_path: str) -> None:
        with self._lock, self._require():
            self._require().execute(
                "DELETE FROM file_hashes WHERE rel_path = ?", (rel_path,),
            )

    def list_paths(self) -> set[str]:
        with self._lock:
            rows = self._require().execute(
                "SELECT rel_path FROM file_hashes"
            ).fetchall()
        return {str(r[0]) for r in rows}

    def _require(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("FileHashRegistry not open")
        return self._conn

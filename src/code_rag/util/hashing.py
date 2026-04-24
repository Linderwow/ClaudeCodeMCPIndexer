from __future__ import annotations

from blake3 import blake3


def chunk_id(repo: str, path: str, symbol: str | None, text: str) -> str:
    """Content-addressed chunk id.

    Same inputs -> same id, so re-indexing unchanged code is a no-op.
    Any drift in repo/path/symbol/text produces a new id and the stale chunk
    is collected by the delete-by-path pass on re-index.
    """
    h = blake3()
    h.update(repo.encode("utf-8"))
    h.update(b"\x00")
    h.update(path.encode("utf-8"))
    h.update(b"\x00")
    h.update((symbol or "").encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest(length=16)


def file_hash(data: bytes) -> str:
    """Fast whole-file hash for quick change detection before re-parse."""
    return blake3(data).hexdigest(length=16)

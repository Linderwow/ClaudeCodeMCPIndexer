from __future__ import annotations

from blake3 import blake3


def chunk_id(
    repo: str,
    path: str,
    symbol: str | None,
    text: str,
    start_line: int = 0,
) -> str:
    """Content-addressed chunk id.

    Same inputs -> same id, so re-indexing unchanged code is a no-op.
    Any drift in repo/path/symbol/start_line/text produces a new id and the
    stale chunk is collected by the delete-by-path pass on re-index.

    `start_line` is included so two chunks with identical text but different
    positions in the same file get different ids — common in real codebases
    that copy-paste the same helper function (e.g. JS `pad()` defined three
    times in one file). Without it, Chroma's upsert would reject the batch
    with DuplicateIDError.

    `start_line` defaults to 0 to keep the function call backward-compatible
    for callers that don't have a position; production callers (chunker, doc
    chunker) should always pass it.
    """
    h = blake3()
    h.update(repo.encode("utf-8"))
    h.update(b"\x00")
    h.update(path.encode("utf-8"))
    h.update(b"\x00")
    h.update((symbol or "").encode("utf-8"))
    h.update(b"\x00")
    h.update(str(int(start_line)).encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest(length=16)


def file_hash(data: bytes) -> str:
    """Fast whole-file hash for quick change detection before re-parse."""
    return blake3(data).hexdigest(length=16)

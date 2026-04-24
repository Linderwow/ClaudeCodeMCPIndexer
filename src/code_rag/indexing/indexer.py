from __future__ import annotations

import asyncio
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from code_rag.chunking.docs import DocChunker
from code_rag.chunking.treesitter import TreeSitterChunker
from code_rag.config import Settings
from code_rag.indexing.walker import CODE_EXT, DOC_EXT, Walker
from code_rag.interfaces.embedder import Embedder
from code_rag.interfaces.vector_store import VectorStore
from code_rag.logging import get
from code_rag.models import Chunk

log = get(__name__)


@dataclass
class IndexStats:
    files_seen: int = 0
    files_indexed: int = 0
    files_skipped: int = 0
    chunks_emitted: int = 0
    chunks_upserted: int = 0
    paths_purged: int = 0
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "files_seen": self.files_seen,
            "files_indexed": self.files_indexed,
            "files_skipped": self.files_skipped,
            "chunks_emitted": self.chunks_emitted,
            "chunks_upserted": self.chunks_upserted,
            "paths_purged": self.paths_purged,
            "elapsed_s": round(self.elapsed_s, 3),
            "errors": self.errors[:10],
        }


@dataclass
class _FileBatch:
    rel_path: str
    chunks: list[Chunk]


class Indexer:
    """Walks roots, parses, embeds, and upserts into the vector store.

    Idempotent per file: we delete-by-path first, then insert the current
    chunks. Reindex is a source-of-truth operation regardless of rename/move/delete.
    """

    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        vector_store: VectorStore,
        lexical_store: object | None = None,       # wired in Phase 2
        graph_store: object | None = None,         # wired in Phase 3
    ) -> None:
        self._settings = settings
        self._embed = embedder
        self._vec = vector_store
        self._lex = lexical_store
        self._graph = graph_store
        self._chunker = TreeSitterChunker(
            min_chars=settings.chunker.min_chars,
            max_chars=settings.chunker.max_chars,
        )
        self._doc_chunker = DocChunker(
            min_chars=settings.chunker.min_chars,
            max_chars=settings.chunker.max_chars,
        )
        # Walker sees the UNION of config.toml roots + dynamically-added roots
        # (auto-discovered workspaces). Keeping the file list dynamic here is
        # what makes `ensure_workspace_indexed` + live watcher work seamlessly.
        self._walker = Walker(settings.all_roots(), settings.ignore.globs)

    # ---- public -------------------------------------------------------------

    async def reindex_all(self) -> IndexStats:
        stats = IndexStats()
        t0 = time.monotonic()
        for _root, abs_path, lang in self._walker.iter_code():
            await self._process_file(abs_path, lang, stats, is_doc=False)
        for _root, abs_path, lang in self._walker.iter_docs():
            await self._process_file(abs_path, lang, stats, is_doc=True)
        stats.elapsed_s = time.monotonic() - t0
        log.info("indexer.done", **stats.as_dict())
        return stats

    async def reindex_path(self, path: Path) -> IndexStats:
        """Reindex a single file or a single subtree."""
        stats = IndexStats()
        t0 = time.monotonic()
        target = path.resolve()
        if target.is_file():
            suffix = target.suffix.lower()
            if suffix in CODE_EXT:
                await self._process_file(target, CODE_EXT[suffix], stats, is_doc=False)
            elif suffix in DOC_EXT:
                await self._process_file(target, DOC_EXT[suffix], stats, is_doc=True)
            else:
                stats.files_skipped += 1
        else:
            for _root, abs_path, lang in self._walker.iter_code():
                try:
                    if not abs_path.resolve().is_relative_to(target):
                        continue
                except ValueError:
                    continue
                await self._process_file(abs_path, lang, stats, is_doc=False)
            for _root, abs_path, lang in self._walker.iter_docs():
                try:
                    if not abs_path.resolve().is_relative_to(target):
                        continue
                except ValueError:
                    continue
                await self._process_file(abs_path, lang, stats, is_doc=True)
        stats.elapsed_s = time.monotonic() - t0
        log.info("indexer.reindex_path.done", target=str(target), **stats.as_dict())
        return stats

    async def remove_path(self, rel_path: str) -> None:
        """Purge all chunks for a file that no longer exists (watcher delete event)."""
        self._vec.delete_by_path(rel_path)
        if self._lex is not None:
            self._lex.delete_by_path(rel_path)  # type: ignore[attr-defined]
        if self._graph is not None:
            self._graph.delete_by_path(rel_path)  # type: ignore[attr-defined]

    # ---- per-file pipeline --------------------------------------------------

    async def _process_file(
        self,
        abs_path: Path,
        language: str,
        stats: IndexStats,
        *,
        is_doc: bool,
    ) -> None:
        stats.files_seen += 1
        rel, repo = _rel_and_repo(abs_path, self._settings.all_roots())

        # CPU-bound parse goes off the event loop.
        try:
            if is_doc:
                chunks = await asyncio.to_thread(
                    self._doc_chunker.chunk_file, repo, abs_path, rel,
                )
            else:
                chunks = await asyncio.to_thread(
                    self._chunker.chunk_file, repo, abs_path, rel, language,
                )
        except Exception as e:
            stats.errors.append(f"chunk:{rel}:{type(e).__name__}:{e}")
            return

        # Empty file: purge any stale chunks and return. This is the only case
        # where we delete without a successful embed to replace the chunks.
        if not chunks:
            purged = self._vec.delete_by_path(rel)
            stats.paths_purged += 1 if purged else 0
            if self._lex is not None:
                try:
                    self._lex.delete_by_path(rel)  # type: ignore[attr-defined]
                except Exception as e:
                    stats.errors.append(f"lex_delete:{rel}:{e}")
            if self._graph is not None:
                try:
                    self._graph.delete_by_path(rel)  # type: ignore[attr-defined]
                except Exception as e:
                    stats.errors.append(f"graph_delete:{rel}:{e}")
            stats.files_skipped += 1
            return

        stats.chunks_emitted += len(chunks)

        # Embed FIRST. If the embedder is down, we want the previous chunks to
        # stay searchable rather than deleting them and leaving the file
        # invisible until the embedder comes back.
        try:
            vectors = await self._embed.embed([c.text for c in chunks])
        except Exception as e:
            stats.errors.append(f"embed:{rel}:{type(e).__name__}:{e}")
            return

        # Only now do we delete stale chunks (source-of-truth semantics) and
        # insert the fresh ones.
        purged = self._vec.delete_by_path(rel)
        stats.paths_purged += 1 if purged else 0
        if self._lex is not None:
            try:
                self._lex.delete_by_path(rel)  # type: ignore[attr-defined]
            except Exception as e:
                stats.errors.append(f"lex_delete:{rel}:{e}")
        if self._graph is not None:
            try:
                self._graph.delete_by_path(rel)  # type: ignore[attr-defined]
            except Exception as e:
                stats.errors.append(f"graph_delete:{rel}:{e}")

        self._vec.upsert(chunks, vectors)
        if self._lex is not None:
            try:
                self._lex.upsert(chunks)  # type: ignore[attr-defined]
            except Exception as e:
                stats.errors.append(f"lex_upsert:{rel}:{e}")
        if self._graph is not None:
            try:
                # Graph ingester reparses the file to get call-site edges that
                # the chunker intentionally drops.
                self._graph.ingest(abs_path, rel, language)  # type: ignore[attr-defined]
            except Exception as e:
                stats.errors.append(f"graph_upsert:{rel}:{e}")

        stats.files_indexed += 1
        stats.chunks_upserted += len(chunks)
        log.debug("indexer.file", path=rel, chunks=len(chunks))


def _rel_and_repo(abs_path: Path, roots: Iterable[Path]) -> tuple[str, str]:
    """Return (posix-relative-path-from-matching-root, root_name).

    Longest-prefix wins: if `abs_path` is under multiple roots (rare — e.g.
    one root is an ancestor of another), we attribute it to the most specific
    root so the relative path and `repo` label are as narrow as possible.
    """
    ap = abs_path.resolve()
    best: tuple[str, str] | None = None
    best_len = -1
    for r in roots:
        try:
            r_res = r.resolve()
            rel = ap.relative_to(r_res)
        except (OSError, ValueError):
            continue
        depth = len(r_res.parts)
        if depth > best_len:
            best = (rel.as_posix(), r_res.name)
            best_len = depth
    if best is not None:
        return best
    return ap.as_posix(), "unknown"

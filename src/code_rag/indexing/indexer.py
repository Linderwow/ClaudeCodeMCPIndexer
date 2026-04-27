from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from code_rag.chunking.docs import DocChunker
from code_rag.chunking.treesitter import TreeSitterChunker
from code_rag.config import Settings
from code_rag.indexing.file_hash import FileHashRegistry
from code_rag.indexing.summary import synthesize_file_summary
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
    files_skipped_unchanged: int = 0  # Phase 14: hash-skip
    chunks_emitted: int = 0
    chunks_upserted: int = 0
    paths_purged: int = 0
    paths_gc: int = 0  # stale paths reaped because their files vanished
    elapsed_s: float = 0.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "files_seen": self.files_seen,
            "files_indexed": self.files_indexed,
            "files_skipped": self.files_skipped,
            "files_skipped_unchanged": self.files_skipped_unchanged,
            "chunks_emitted": self.chunks_emitted,
            "chunks_upserted": self.chunks_upserted,
            "paths_purged": self.paths_purged,
            "paths_gc": self.paths_gc,
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
        file_hashes: FileHashRegistry | None = None,  # wired in Phase 14
    ) -> None:
        self._settings = settings
        self._embed = embedder
        self._vec = vector_store
        self._lex = lexical_store
        self._graph = graph_store
        self._hashes = file_hashes
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
        # Phase 15: serialize store writes across parallel file workers.
        # `_process_file` acquires this lock around the delete+upsert critical
        # section; chunking and embedding stay outside the lock so workers
        # truly proceed concurrently on those CPU/IO-bound stages.
        self._store_lock: asyncio.Lock = asyncio.Lock()

    # ---- public -------------------------------------------------------------

    async def reindex_all(self) -> IndexStats:
        """Phase 15: bounded-parallel pipeline.

        Files are processed concurrently up to `settings.indexer.parallel_workers`
        in flight at once. Each worker runs the per-file pipeline (read → hash-
        skip check → chunk → embed → flush), with the per-store writes
        serialized via `_store_lock` so Chroma/Kuzu/FTS see ordered upserts.

        With workers=4 and a typical mix of NL+identifier-heavy LM Studio
        embedding latency, this drops wall clock by roughly 3.5-4x vs the
        previous strictly-sequential loop without inflating embedder load
        beyond what LM Studio handles cleanly (see Phase D -- we proved 1
        embedder slot at 488 ms/req is healthy; 4 in-flight requests fit).
        """
        stats = IndexStats()
        t0 = time.monotonic()
        # Track every path the walker visits this pass. Anything in the index
        # that's NOT in this set represents a stale entry (file deleted, repo
        # removed, root unconfigured, glob now ignores it, file too big, etc.)
        # and gets reaped after the walk finishes -- see `_gc_stale_paths`.
        seen: set[str] = set()
        seen_lock = asyncio.Lock()  # for the seen set across workers
        # Single store-write lock: Chroma/Kuzu/FTS can't safely interleave
        # writes from multiple coroutines on the same connection.
        self._store_lock = asyncio.Lock()
        sem = asyncio.Semaphore(max(1, self._settings.indexer.parallel_workers))

        async def _one(abs_path: Path, lang: str, is_doc: bool) -> None:
            async with sem:
                rel = await self._process_file(abs_path, lang, stats, is_doc=is_doc)
                if rel is not None:
                    async with seen_lock:
                        seen.add(rel)

        tasks: list[asyncio.Task[None]] = [
            asyncio.create_task(_one(abs_path, lang, is_doc=False))
            for _root, abs_path, lang in self._walker.iter_code()
        ]
        tasks.extend(
            asyncio.create_task(_one(abs_path, lang, is_doc=True))
            for _root, abs_path, lang in self._walker.iter_docs()
        )
        if tasks:
            await asyncio.gather(*tasks)
        await self._gc_stale_paths(seen, stats)
        stats.elapsed_s = time.monotonic() - t0
        log.info("indexer.done", **stats.as_dict())
        return stats

    async def _gc_stale_paths(self, seen: set[str], stats: IndexStats) -> None:
        """Purge index entries whose source files weren't visited this pass.

        Diff = (paths-in-lex) - (paths-walked-now). Anything in the diff is
        either:
          * deleted from disk while the watcher was offline,
          * now ignored by config,
          * outside any configured root after a roots edit,
          * over the file-size cap,
          * orphaned by a renamed file we missed.
        Either way it's safe to remove; if the file comes back, the next
        walker pass will re-ingest it via the normal upsert path.
        """
        if self._lex is None:
            return
        try:
            indexed = self._lex.list_paths()  # type: ignore[attr-defined]
        except Exception as e:
            stats.errors.append(f"gc_list_paths:{e}")
            return
        stale = indexed - seen
        if not stale:
            return
        log.info("indexer.gc.start", stale_paths=len(stale))
        for rel in stale:
            try:
                self._vec.delete_by_path(rel)
                self._lex.delete_by_path(rel)  # type: ignore[attr-defined]
                if self._graph is not None:
                    self._graph.delete_by_path(rel)  # type: ignore[attr-defined]
                if self._hashes is not None:
                    self._hashes.delete(rel)
                stats.paths_gc += 1
            except Exception as e:
                stats.errors.append(f"gc_delete:{rel}:{e}")
        log.info("indexer.gc.done", paths_gc=stats.paths_gc)

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
        # Drop the file-hash row so a future re-creation of this rel_path is
        # never falsely "skip-unchanged"-ed against stale content.
        if self._hashes is not None:
            with contextlib.suppress(Exception):
                self._hashes.delete(rel_path)  # best effort -- registry is not source-of-truth

    # ---- per-file pipeline --------------------------------------------------

    async def _process_file(
        self,
        abs_path: Path,
        language: str,
        stats: IndexStats,
        *,
        is_doc: bool,
    ) -> str | None:
        """Process one file. Returns the rel-path so `reindex_all` can track
        which paths were visited (for the stale-path GC pass)."""
        stats.files_seen += 1
        rel, repo = _rel_and_repo(abs_path, self._settings.all_roots())

        # Phase 14: hash-skip. If we've indexed this exact content before,
        # the per-chunk pipeline (parse → chunk → embed → upsert) is purely
        # busywork -- same content always produces same chunk_ids and same
        # vectors. Reading the bytes once + a blake3 hash is ~milliseconds;
        # parsing + embedding a single 200-line file is ~seconds. Net win.
        try:
            content = abs_path.read_bytes()
        except OSError as e:
            stats.errors.append(f"read:{rel}:{type(e).__name__}:{e}")
            return rel
        if self._hashes is not None and self._hashes.is_unchanged(rel, content):
            stats.files_skipped_unchanged += 1
            log.debug("indexer.skip_unchanged", path=rel)
            return rel

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
            return rel

        # Empty file: purge any stale chunks and return. This is the only case
        # where we delete without a successful embed to replace the chunks.
        if not chunks:
            async with self._store_lock:
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
            return rel

        # Phase 31: synthesize one extra "table of contents" chunk per file
        # — path / language / list of symbols. Gives the embedder a single
        # high-level handle on the file for natural-language queries that
        # don't share tokens with any specific symbol body. Skipped for
        # docs (DocChunker already produces meaningful prose chunks). Also
        # skipped if the file produced 0 chunks (handled above).
        if not is_doc:
            summary = synthesize_file_summary(repo, rel, language, chunks)
            if summary is not None:
                chunks = [*chunks, summary]

        stats.chunks_emitted += len(chunks)

        # Embed FIRST. If the embedder is down, we want the previous chunks to
        # stay searchable rather than deleting them and leaving the file
        # invisible until the embedder comes back.
        try:
            vectors = await self._embed.embed([c.text for c in chunks])
        except Exception as e:
            stats.errors.append(f"embed:{rel}:{type(e).__name__}:{e}")
            return rel

        # Only now do we delete stale chunks (source-of-truth semantics) and
        # insert the fresh ones. All store writes are serialized via the
        # store-lock so parallel workers can't race on Chroma/FTS/Kuzu.
        async with self._store_lock:
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
                    # Graph ingester reparses the file to get call-site edges
                    # that the chunker intentionally drops.
                    self._graph.ingest(abs_path, rel, language)  # type: ignore[attr-defined]
                except Exception as e:
                    stats.errors.append(f"graph_upsert:{rel}:{e}")

        # Stamp the content hash now that we've fully ingested. Doing it AFTER
        # the upserts means a crash mid-embed leaves the registry pointing at
        # the previous successfully-indexed content, so the next pass retries.
        if self._hashes is not None:
            try:
                self._hashes.upsert(rel, content)
            except Exception as e:
                stats.errors.append(f"hash_upsert:{rel}:{e}")

        stats.files_indexed += 1
        stats.chunks_upserted += len(chunks)
        log.debug("indexer.file", path=rel, chunks=len(chunks))
        return rel


def _rel_and_repo(abs_path: Path, roots: Iterable[Path]) -> tuple[str, str]:
    """Return (posix-relative-path-from-matching-root, root_name).

    Longest-prefix wins: if `abs_path` is under multiple roots (rare -- e.g.
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

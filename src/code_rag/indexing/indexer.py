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
from code_rag.indexing.walker import CODE_EXT, DOC_EXT, MAX_FILE_BYTES, Walker
from code_rag.interfaces.embedder import Embedder
from code_rag.interfaces.vector_store import VectorStore
from code_rag.logging import get
from code_rag.models import Chunk
from code_rag.util.globs import matches_any

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
        """Phase 35: streaming producer/consumer pipeline.

        Pre-Phase-35, this method materialized every walker entry into an
        `asyncio.create_task(_one(...))` via list comprehension *before*
        any consumer ran. On a corpus with thousands of node_modules
        siblings to skip, the SYNCHRONOUS walker enumeration blocked the
        asyncio event loop for many minutes — long enough that the first
        embedded chunk didn't land until the entire walk completed. That
        broke the bge-code-v1 swap attempt and is generally a bad
        architectural shape: chunking + embedding should overlap with
        walking, not wait for it.

        The new shape is a bounded producer/consumer:

            producer (in a thread)
                walks the filesystem and pushes (path, lang, is_doc) onto
                an asyncio.Queue. Sleeps when consumers can't keep up,
                so the queue stays bounded.

            consumers (parallel_workers of them)
                each pull from the queue and run _process_file. Use the
                same `_store_lock` so Chroma/FTS/Kuzu writes serialize.

        The producer runs in `asyncio.to_thread` because Walker.iter_code
        is a sync generator; running it on the event loop directly would
        re-introduce the blocking we're fixing. asyncio.Queue is
        thread-safe via `loop.call_soon_threadsafe` under the hood for
        cross-thread put.

        Throughput: chunks start landing within seconds of process start
        instead of minutes. Total wall clock unchanged or better (no
        difference in steady state; massive improvement in time-to-first-
        chunk and time-to-progress-visible).
        """
        stats = IndexStats()
        t0 = time.monotonic()
        # Track every path the walker visits this pass. Anything in the index
        # that's NOT in this set represents a stale entry (file deleted, repo
        # removed, root unconfigured, glob now ignores it, file too big, etc.)
        # and gets reaped after the walk finishes — see `_gc_stale_paths`.
        seen: set[str] = set()
        seen_lock = asyncio.Lock()
        # Phase 38: do NOT reassign self._store_lock here. Live watcher
        # consumers may already be awaiting the constructor-created lock
        # at this moment; replacing the instance leaves them holding a
        # dead reference while new callers acquire a different lock,
        # which would let two writers enter the Chroma/FTS/Kuzu critical
        # section concurrently. The constructor's lock is the single
        # source of truth for the indexer's lifetime.
        n_workers = max(1, self._settings.indexer.parallel_workers)

        # Bounded queue: (abs_path: Path, lang: str, is_doc: bool) | None
        # The None sentinel signals consumers that the producer is done —
        # we put one per consumer so they all exit cleanly.
        queue: asyncio.Queue[tuple[Path, str, bool] | None] = asyncio.Queue(
            maxsize=n_workers * 4,
        )
        loop = asyncio.get_running_loop()

        def _producer_sync() -> None:
            """Walk the filesystem and push tuples onto the queue from
            a worker thread. Uses run_coroutine_threadsafe to put items
            on the asyncio.Queue from outside the loop thread."""
            for _root, abs_path, lang in self._walker.iter_code():
                fut = asyncio.run_coroutine_threadsafe(
                    queue.put((abs_path, lang, False)), loop,
                )
                fut.result()  # block in the thread; ensures queue back-pressure.
            for _root, abs_path, lang in self._walker.iter_docs():
                fut = asyncio.run_coroutine_threadsafe(
                    queue.put((abs_path, lang, True)), loop,
                )
                fut.result()
            # One sentinel per consumer so each can `break` out of its loop.
            for _ in range(n_workers):
                fut = asyncio.run_coroutine_threadsafe(queue.put(None), loop)
                fut.result()

        async def _consumer() -> None:
            # Phase 37 audit fix: every iteration must call task_done(),
            # even when _process_file raises. Previously a single
            # uncaught exception killed the consumer mid-flight without
            # marking the queue item as done — Producer's `queue.put()`
            # would then block forever once the queue filled.
            while True:
                item = await queue.get()
                try:
                    if item is None:
                        return
                    abs_path, lang, is_doc = item
                    try:
                        rel = await self._process_file(
                            abs_path, lang, stats, is_doc=is_doc,
                        )
                    except Exception as e:
                        # _process_file's contract is to log + record the
                        # error in stats and return; if any path raises
                        # anyway, we still want the worker to keep going.
                        stats.errors.append(
                            f"consumer:{getattr(abs_path, 'name', '?')}:{type(e).__name__}:{e}",
                        )
                        rel = None
                    if rel is not None:
                        async with seen_lock:
                            seen.add(rel)
                finally:
                    queue.task_done()

        # Run producer + N consumers concurrently.
        producer_task = asyncio.create_task(asyncio.to_thread(_producer_sync))
        consumer_tasks = [
            asyncio.create_task(_consumer()) for _ in range(n_workers)
        ]
        await producer_task
        # return_exceptions so a single sick worker doesn't tank the rest
        # of the reindex; the per-worker try/finally above already guards
        # task_done(), so gather is just collecting results.
        await asyncio.gather(*consumer_tasks, return_exceptions=True)

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
        """Reindex a single file or a single subtree.

        Three branches:
          * target doesn't exist  -> treat as delete (purge any prior chunks)
          * target is a file      -> process just that file
          * target is a directory -> walk INSIDE target only (NOT the whole
            indexed tree) so subtree reindex stays O(files-under-target),
            not O(all-files-in-config-roots).

        The "doesn't exist" branch is critical for the live watcher: editors
        often produce transient events where the file briefly disappears
        between save+rename. Without this short-circuit, every such phantom
        event used to fall into the directory branch and walk the entire
        12k+-file index for nothing — eating 11+ minutes of CPU per event
        and stacking up under the watcher's serial drain loop.
        """
        stats = IndexStats()
        t0 = time.monotonic()
        target = path.resolve()

        # 1) Target gone — purge any chunks attributable to this path and bail.
        if not target.exists():
            rel_guess = self._best_rel_path(target)
            if rel_guess is not None:
                try:
                    await self.remove_path(rel_guess)
                    stats.paths_purged += 1
                except Exception as e:
                    stats.errors.append(f"remove_missing:{rel_guess}:{e}")
            stats.elapsed_s = time.monotonic() - t0
            log.info("indexer.reindex_path.done", target=str(target), **stats.as_dict())
            return stats

        # 2) Single file — fast path, no walk.
        if target.is_file():
            suffix = target.suffix.lower()
            if suffix in CODE_EXT:
                await self._process_file(target, CODE_EXT[suffix], stats, is_doc=False)
            elif suffix in DOC_EXT:
                await self._process_file(target, DOC_EXT[suffix], stats, is_doc=True)
            else:
                stats.files_skipped += 1
            stats.elapsed_s = time.monotonic() - t0
            log.info("indexer.reindex_path.done", target=str(target), **stats.as_dict())
            return stats

        # 3) Subtree walk — bounded to TARGET ONLY (not all roots). We still
        #    apply ignore globs and size caps via the walker's helpers.
        ignored = self._settings.ignore.globs
        for fp in target.rglob("*"):
            try:
                if not fp.is_file():
                    continue
            except OSError:
                continue
            if matches_any(fp, ignored):
                continue
            try:
                if fp.stat().st_size > MAX_FILE_BYTES:
                    continue
            except OSError:
                continue
            suffix = fp.suffix.lower()
            if suffix in CODE_EXT:
                await self._process_file(fp, CODE_EXT[suffix], stats, is_doc=False)
            elif suffix in DOC_EXT:
                await self._process_file(fp, DOC_EXT[suffix], stats, is_doc=True)

        stats.elapsed_s = time.monotonic() - t0
        log.info("indexer.reindex_path.done", target=str(target), **stats.as_dict())
        return stats

    def _best_rel_path(self, abs_path: Path) -> str | None:
        """Resolve `abs_path` to its index-relative form (relative to whichever
        configured/dynamic root contains it). Returns None if outside every
        root — in which case we have no chunks to purge."""
        ap = abs_path  # may not exist on disk; just compare on the path string
        with contextlib.suppress(OSError):
            ap = abs_path.resolve()
        for r in self._settings.all_roots():
            try:
                rr = r.resolve()
            except OSError:
                continue
            try:
                return ap.relative_to(rr).as_posix()
            except ValueError:
                continue
        return None

    async def remove_path(self, rel_path: str) -> None:
        """Purge all chunks for a file that no longer exists (watcher delete event).

        Phase 36-H: each store's delete is wrapped in suppress() so a
        Kuzu lock conflict (common when MCP servers hold the read-only
        graph handle) doesn't abort the whole removal — vector + lexical
        + hash purges still happen, graph just retains a stale entry
        until the next successful delete-by-path. A flood of .tmp file
        events used to spam the watcher's error log because graph errors
        were uncaught here, while _process_file already had this guard.
        """
        with contextlib.suppress(Exception):
            self._vec.delete_by_path(rel_path)
        if self._lex is not None:
            with contextlib.suppress(Exception):
                self._lex.delete_by_path(rel_path)  # type: ignore[attr-defined]
        if self._graph is not None:
            with contextlib.suppress(Exception):
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
        # Phase 37 audit fix: wrap vec.delete in suppress to mirror the
        # remove_path hardening — a single transient store failure shouldn't
        # leave the file half-deleted.
        if not chunks:
            async with self._store_lock:
                try:
                    purged = self._vec.delete_by_path(rel)
                    stats.paths_purged += 1 if purged else 0
                except Exception as e:
                    stats.errors.append(f"vec_delete:{rel}:{e}")
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
        # Phase 37 audit fix: track per-store success. Previously
        # `vec.upsert` had no try/except so any Chroma hiccup propagated up,
        # killing the consumer worker AND skipping `task_done()`. Worse,
        # the hash registry was stamped unconditionally below, so a partial
        # write left the file silently missing from search forever.
        vec_ok = lex_ok = graph_ok = True
        async with self._store_lock:
            try:
                purged = self._vec.delete_by_path(rel)
                stats.paths_purged += 1 if purged else 0
            except Exception as e:
                stats.errors.append(f"vec_delete:{rel}:{e}")
                vec_ok = False
            if self._lex is not None:
                try:
                    self._lex.delete_by_path(rel)  # type: ignore[attr-defined]
                except Exception as e:
                    stats.errors.append(f"lex_delete:{rel}:{e}")
                    lex_ok = False
            if self._graph is not None:
                try:
                    self._graph.delete_by_path(rel)  # type: ignore[attr-defined]
                except Exception as e:
                    stats.errors.append(f"graph_delete:{rel}:{e}")
                    graph_ok = False

            try:
                self._vec.upsert(chunks, vectors)
            except Exception as e:
                stats.errors.append(f"vec_upsert:{rel}:{e}")
                vec_ok = False
            if self._lex is not None:
                try:
                    self._lex.upsert(chunks)  # type: ignore[attr-defined]
                except Exception as e:
                    stats.errors.append(f"lex_upsert:{rel}:{e}")
                    lex_ok = False
            if self._graph is not None:
                try:
                    # Graph ingester reparses the file to get call-site edges
                    # that the chunker intentionally drops.
                    self._graph.ingest(abs_path, rel, language)  # type: ignore[attr-defined]
                except Exception as e:
                    stats.errors.append(f"graph_upsert:{rel}:{e}")
                    graph_ok = False

        # Stamp the content hash ONLY when every store accepted the write.
        # Otherwise leave the hash unchanged so the next pass retries — that
        # closes the partial-write window where a transient Kuzu lock would
        # have left the file out of search results forever (the next pass
        # would have hit `is_unchanged → skip` if we'd stamped the hash).
        all_ok = vec_ok and lex_ok and graph_ok
        if all_ok and self._hashes is not None:
            try:
                self._hashes.upsert(rel, content)
            except Exception as e:
                stats.errors.append(f"hash_upsert:{rel}:{e}")

        if all_ok:
            stats.files_indexed += 1
            stats.chunks_upserted += len(chunks)
        log.debug("indexer.file", path=rel, chunks=len(chunks),
                  all_ok=all_ok, vec_ok=vec_ok, lex_ok=lex_ok, graph_ok=graph_ok)
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

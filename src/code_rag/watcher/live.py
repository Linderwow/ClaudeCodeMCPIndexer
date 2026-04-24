"""Live filesystem watcher with debounced incremental reindex.

Design notes
------------
- watchdog runs its event callbacks on a private thread; we can't safely touch
  async/Chroma/Kuzu from there. So the watcher JUST enqueues events onto a
  thread-safe queue and an async `_drain` task drains/debounces/acts on them.
- Debounce window collapses rapid saves (e.g., editors' multi-write flush) into
  a single reindex per file.
- Deletes and renames are handled symmetrically: we always delete-by-path first,
  then (if the file still exists) re-chunk + re-embed + re-upsert. The delete
  path is what makes the index converge regardless of event ordering.
"""
from __future__ import annotations

import asyncio
import contextlib
import queue
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from code_rag.indexing.walker import CODE_EXT, DOC_EXT, MAX_FILE_BYTES
from code_rag.logging import get
from code_rag.util.globs import matches_any

if TYPE_CHECKING:
    from code_rag.config import Settings
    from code_rag.indexing.indexer import Indexer

log = get(__name__)


class EventKind(Enum):
    UPSERT = "upsert"
    DELETE = "delete"


@dataclass(order=True)
class _Pending:
    # Earliest deadline wins in the heap-like loop; we use a simple dict and
    # re-scan each tick to keep the code obvious.
    deadline: float
    path: str = field(compare=False)
    kind: EventKind = field(default=EventKind.UPSERT, compare=False)


class _Handler(FileSystemEventHandler):
    """Pushes raw events onto a thread-safe queue for the async drain."""

    def __init__(self, q: queue.Queue[FileSystemEvent]) -> None:
        self._q = q

    def on_any_event(self, event: FileSystemEvent) -> None:
        # We coalesce in the drain; just push.
        if event.is_directory:
            return
        self._q.put(event)


class LiveWatcher:
    """Watches configured roots and feeds an Indexer incrementally."""

    def __init__(
        self,
        settings: Settings,
        indexer: Indexer,
    ) -> None:
        self._settings = settings
        self._indexer = indexer
        self._queue: queue.Queue[FileSystemEvent] = queue.Queue()
        # watchdog re-exports `Observer` as a variable that returns a platform-
        # specific observer class; using it as a type annotation is invalid.
        self._observer: Any = None
        self._stop = asyncio.Event()
        self._debounce_s = settings.watcher.debounce_ms / 1000.0

    async def run(self) -> None:
        """Start the observer, run the drain loop, stop gracefully on cancel."""
        self._observer = Observer()
        handler = _Handler(self._queue)
        # Union of config roots + dynamic (auto-discovered) roots. A root that
        # vanishes between startups is skipped (watchdog crashes on missing).
        all_roots = [r for r in self._settings.all_roots() if r.exists()]
        for root in all_roots:
            self._observer.schedule(handler, str(root), recursive=True)
        self._observer.start()
        log.info("watcher.started", roots=[str(r) for r in all_roots])

        try:
            await self._drain()
        finally:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            log.info("watcher.stopped")

    def request_stop(self) -> None:
        self._stop.set()

    # ---- drain loop ---------------------------------------------------------

    async def _drain(self) -> None:
        pending: dict[str, _Pending] = {}
        tick = 0.05  # 50ms poll cadence

        while not self._stop.is_set():
            # Pull as many events as are ready without blocking.
            while True:
                try:
                    ev = self._queue.get_nowait()
                except queue.Empty:
                    break
                self._schedule_event(ev, pending)

            # Flush anything whose debounce has elapsed.
            now = time.monotonic()
            due = [p for p in pending.values() if p.deadline <= now]
            for p in due:
                pending.pop(p.path, None)
                await self._apply(p)

            await asyncio.sleep(tick)

    def _schedule_event(self, ev: FileSystemEvent, pending: dict[str, _Pending]) -> None:
        paths: list[tuple[str, EventKind]] = []
        src = Path(_as_str(ev.src_path))
        # Event types: 'created' | 'modified' | 'deleted' | 'moved'
        if ev.event_type == "deleted":
            paths.append((str(src), EventKind.DELETE))
        elif ev.event_type == "moved":
            # Treat as upsert(new) + delete(old). ORDER MATTERS: upsert the
            # destination FIRST so if the watcher crashes mid-rename we end up
            # with duplicate chunks at both paths rather than data loss. The
            # surviving old-path entries are reaped on next full reindex or
            # when either file is touched again.
            dest = getattr(ev, "dest_path", None)
            if dest:
                paths.append((str(Path(_as_str(dest))), EventKind.UPSERT))
            paths.append((str(src), EventKind.DELETE))
        else:
            paths.append((str(src), EventKind.UPSERT))

        now = time.monotonic()
        for p, kind in paths:
            if not self._is_indexable(Path(p), kind):
                continue
            # Latest event for a path wins — extend the deadline and override kind.
            pending[p] = _Pending(deadline=now + self._debounce_s, path=p, kind=kind)

    def _is_indexable(self, path: Path, kind: EventKind) -> bool:
        # Deletes are always handled (so stale entries get purged even for
        # paths we wouldn't have indexed — cheap and harmless).
        if kind is EventKind.DELETE:
            return not matches_any(path, self._settings.ignore.globs)
        if matches_any(path, self._settings.ignore.globs):
            return False
        suffix = path.suffix.lower()
        if suffix in CODE_EXT or suffix in DOC_EXT:
            # Size cap — watchdog events don't know size, so we stat if possible.
            try:
                if path.exists() and path.stat().st_size > MAX_FILE_BYTES:
                    return False
            except OSError:
                return False
            return True
        return False

    async def _apply(self, p: _Pending) -> None:
        try:
            if p.kind is EventKind.DELETE:
                rel = self._rel_path(Path(p.path))
                if rel is not None:
                    await self._indexer.remove_path(rel)
                    log.info("watcher.deleted", path=rel)
            else:
                # Route through reindex_path; Indexer handles chunk+embed+upsert
                # and does the delete-then-insert pass internally.
                stats = await self._indexer.reindex_path(Path(p.path))
                log.info("watcher.reindexed", path=p.path, **stats.as_dict())
        except Exception as e:
            log.exception("watcher.apply_error", path=p.path, err=str(e))

    def _rel_path(self, abs_path: Path) -> str | None:
        ap = abs_path.resolve()
        for r in self._settings.all_roots():
            try:
                return ap.relative_to(r.resolve()).as_posix()
            except ValueError:
                continue
        return None


def _as_str(p: str | bytes) -> str:
    """watchdog's FileSystemEvent.src_path is typed `str | bytes`; normalize."""
    return p.decode("utf-8", "replace") if isinstance(p, bytes) else p


# ---- top-level helpers -----------------------------------------------------


async def watch_forever(settings: Settings, indexer: Indexer) -> None:
    """Block until cancelled. Swallowed CancelledError at the top so shutdown
    is quiet when Ctrl-C is pressed."""
    watcher = LiveWatcher(settings, indexer)
    task = asyncio.create_task(watcher.run())
    with contextlib.suppress(asyncio.CancelledError):
        await task

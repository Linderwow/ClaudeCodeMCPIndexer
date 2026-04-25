"""Windowless boot-time entry point.

Invoked by Task Scheduler via:
    pythonw.exe -m code_rag.autostart_bootstrap

pythonw.exe has no associated console, so this process is fully hidden — no
tray icon, no flash, no window. The full boot chain runs here:

    1. Load config.
    2. Ensure LM Studio is running with the embedder loaded.
       (idempotent — if it's already up, step returns immediately)
    3. Block in `code-rag watch` equivalent, watching configured roots for
       changes and keeping the index fresh.

All events go to <data_dir>/../logs/autostart.log so failures are debuggable
after a reboot without having to attach a terminal.
"""
from __future__ import annotations

import asyncio
import contextlib
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

from code_rag.config import load_settings
from code_rag.factory import (
    build_embedder,
    build_lexical_store,
    build_vector_store,
)
from code_rag.graph.ingest import GraphIngester
from code_rag.indexing.file_hash import FileHashRegistry
from code_rag.indexing.indexer import Indexer
from code_rag.lms_ctl import ensure_lm_studio_ready
from code_rag.logging import configure, get
from code_rag.stores.chroma_vector import ChromaVectorStore
from code_rag.watcher.live import watch_forever

log = get(__name__)


def _plain_log(log_path: Path, line: str) -> None:
    """Append a stamped line to the autostart log file.

    We use a dedicated plain-text log (separate from the structured JSONL) so
    non-technical users can eyeball it after a failed boot.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now(UTC).isoformat()}  {line}\n")


async def _run() -> int:
    # 1) Load settings. If this fails, log and bail — nothing we can do.
    try:
        settings = load_settings()
    except Exception as e:
        # Last-ditch log to a well-known location.
        fallback = Path.home() / ".code-rag-autostart-failure.log"
        fallback.write_text(
            f"{datetime.now(UTC).isoformat()}  load_settings failed: {e}\n{traceback.format_exc()}",
            encoding="utf-8",
        )
        return 2

    autostart_log = settings.paths.log_dir / "autostart.log"
    configure(settings.paths.log_dir, level="INFO")

    _plain_log(autostart_log, "bootstrap starting")
    log.info("autostart.begin", python=sys.executable)

    # 2) Ensure LM Studio is up with the embedder loaded.
    extra: tuple[str, ...] = ()
    if settings.reranker.kind == "lm_studio" and settings.reranker.model:
        extra = (settings.reranker.model,)
    result = await asyncio.to_thread(
        ensure_lm_studio_ready,
        settings.embedder.base_url,
        settings.embedder.model,
        extra,
    )
    for s in result.steps:
        _plain_log(autostart_log, f"lms: {s}")
    if not result.ok:
        _plain_log(autostart_log, f"lms FAILED: {result.error}")
        log.error("autostart.lm_studio_failed", error=result.error)
        # Don't exit — the watcher can still run (it just won't reindex on changes
        # until LM Studio comes up, which might happen manually). Log and proceed.
        _plain_log(autostart_log, "continuing without LM Studio; watcher will retry on each event")

    # 3) Open stores and launch the watcher. The watcher will catch up on any
    #    changes that happened while the machine was off.
    embedder = build_embedder(settings)
    vec = build_vector_store(settings)
    lex = build_lexical_store(settings)

    try:
        # If LM Studio ISN'T up, health() will throw. Swallow and let the watcher
        # retry per-event.
        try:
            await embedder.health()
        except Exception as e:
            _plain_log(autostart_log, f"embedder health deferred: {e}")

        try:
            meta = ChromaVectorStore.build_meta(
                embedder_kind=settings.embedder.kind,
                embedder_model=embedder.model,
                embedder_dim=embedder.dim if embedder.dim > 0 else 1,
            )
            vec.open(meta)
            lex.open()
        except Exception as e:
            _plain_log(autostart_log, f"open stores FAILED: {e}")
            log.exception("autostart.open_failed")
            return 3

        # Transient graph ingest: open+close Kuzu per event so MCP/CLI readers
        # can query the graph in the gaps (Kuzu 0.11 takes an exclusive OS
        # file lock even in read-only mode).
        # File-hash registry: skip parse+embed when content unchanged (Phase 14).
        hashes = FileHashRegistry(settings.file_hashes_path)
        try:
            hashes.open()
        except Exception as e:
            _plain_log(autostart_log, f"file_hash open failed (non-fatal): {e}")
        indexer = Indexer(
            settings, embedder, vec,
            lexical_store=lex,
            graph_store=GraphIngester.transient(settings.kuzu_dir),
            file_hashes=hashes,
        )

        try:
            await indexer.reindex_all()
            _plain_log(autostart_log, "catch-up reindex done")
        except Exception as e:
            _plain_log(autostart_log, f"catch-up reindex failed (non-fatal): {e}")
            log.exception("autostart.catchup_failed")

        # Phase 23: post-catch-up fsck with auto-repair. Cheap (a few
        # seconds), non-destructive (only safe repairs), and surfaces drift
        # / orphans before they become user-visible bugs. Runs once per
        # logon — periodic re-runs are handled by a separate scheduled task.
        try:
            from code_rag.ops import fsck
            report = fsck(settings, vec, lex, auto_fix=True)
            _plain_log(autostart_log,
                       f"fsck: ok={report.ok} issues={len(report.issues)} fixed={len(report.fixed)}")
        except Exception as e:
            _plain_log(autostart_log, f"fsck failed (non-fatal): {e}")

        # Phase 23: snapshot OpenMetrics on each boot so the operator can
        # eyeball trends across reboots without a running Prometheus server.
        try:
            from code_rag.ops import metrics_text
            metrics_path = settings.paths.log_dir / "metrics.prom"
            metrics_path.write_text(metrics_text(settings, vec, lex), encoding="utf-8")
        except Exception as e:
            _plain_log(autostart_log, f"metrics snapshot failed (non-fatal): {e}")

        _plain_log(autostart_log, "watcher starting")
        await watch_forever(settings, indexer)
        _plain_log(autostart_log, "watcher exited normally")
        return 0
    finally:
        vec.close()
        lex.close()
        with contextlib.suppress(Exception):
            hashes.close()


def main() -> None:
    """Entry point for `pythonw.exe -m code_rag.autostart_bootstrap`."""
    try:
        code = asyncio.run(_run())
    except KeyboardInterrupt:
        code = 0
    except Exception as e:
        fallback = Path.home() / ".code-rag-autostart-failure.log"
        with fallback.open("a", encoding="utf-8") as f:
            f.write(
                f"{datetime.now(UTC).isoformat()}  uncaught: {e}\n{traceback.format_exc()}\n"
            )
        code = 1
    sys.exit(code)


if __name__ == "__main__":
    main()

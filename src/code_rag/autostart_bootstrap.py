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

    # Phase 32: singleton lock — refuse to run a second autostart bootstrap.
    # Without this, a second logon trigger or a manual `schtasks /Run` while
    # the first instance is still alive produces two watchers competing on
    # the same Chroma collection. We intentionally re-use the watcher.lock
    # path so the autostart bootstrap and a manually-launched `code-rag watch`
    # mutually exclude.
    from code_rag.util.proc_hygiene import SingletonLock
    lock_path = settings.paths.data_dir / "watcher.lock"
    lock = SingletonLock(lock_path).__enter__()
    if not lock.acquired:
        _plain_log(autostart_log,
                   f"another watcher instance is alive (lockfile {lock_path}); exiting")
        return 0

    _plain_log(autostart_log, "bootstrap starting")
    log.info("autostart.begin", python=sys.executable)

    # Phase 39: clear any leftover Stop All intent marker. Semantics: the
    # user said "stay stopped until I hit Start All OR restart the PC".
    # The autostart task fires `AtLogon`, so a PC restart -> we run -> we
    # delete the marker before continuing. Without this, the watcher
    # would refuse to come back even after a reboot.
    from code_rag.util.stop_marker import (
        clear_intentionally_stopped,
        is_intentionally_stopped,
    )
    if (is_intentionally_stopped(settings.paths.data_dir)
            and clear_intentionally_stopped(settings.paths.data_dir,
                                            reason="autostart_bootstrap.logon")):
        _plain_log(autostart_log,
                   "cleared stale Stop All marker (PC restart resumes service)")

    # 2) Ensure LM Studio is up with the embedder loaded.
    # Pre-load any auxiliary chat/rerank models so the first MCP query after
    # boot doesn't pay a 5-10s JIT-load penalty.
    #
    # Phase 29: cross_encoder reranker doesn't go through LM Studio at all
    # (it's loaded by sentence-transformers in-process). Skip those.
    extra: tuple[str, ...] = ()
    rer = settings.reranker
    if rer.kind in ("lm_studio", "lm_chat") and rer.model:
        extra = (*extra, rer.model)
    hyde_model = getattr(settings.embedder, "hyde_model", None)
    if hyde_model and hyde_model not in extra:
        extra = (*extra, str(hyde_model))
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

        # File-hash registry: skip parse+embed when content unchanged (Phase 14).
        hashes = FileHashRegistry(settings.file_hashes_path)
        try:
            hashes.open()
        except Exception as e:
            _plain_log(autostart_log, f"file_hash open failed (non-fatal): {e}")

        # Phase 35: graph ingestion is SKIPPED during the catch-up reindex.
        #
        # Why: GraphIngester.transient opens+closes Kuzu per file event.
        # On a fresh full reindex (27K+ files), that's 54K+ open/close
        # cycles. Kuzu's storage layout amplifies this badly — observed in
        # production: graph.kz grew to 13.5 GB after only 2K files (~6 MB
        # per file in graph alone, vs ~2 KB per chunk in FTS — 3000x more
        # data). Per-file ingest then takes 2-3 minutes because each open
        # has to load the giant DB. Net: catch-up reindex slows to a crawl
        # (10-30 chunks/min instead of 500+).
        #
        # The graph is only used by the optional get_callers / get_callees
        # MCP tools, which are not the critical path for retrieval. We let
        # the watcher's STEADY-STATE per-file ingest populate the graph
        # incrementally as users edit files — that path stays fast because
        # graph.kz starts small.
        #
        # Users who need the full graph immediately should run
        # `code-rag index` (CLI) — it uses GraphIngester(graph) non-
        # transient mode which holds Kuzu open for the entire reindex,
        # avoiding the open/close churn.
        indexer_for_catchup = Indexer(
            settings, embedder, vec,
            lexical_store=lex,
            graph_store=None,                 # ← skip graph during catch-up
            file_hashes=hashes,
        )

        try:
            await indexer_for_catchup.reindex_all()
            _plain_log(autostart_log, "catch-up reindex done (graph skipped — incremental)")
        except Exception as e:
            _plain_log(autostart_log, f"catch-up reindex failed (non-fatal): {e}")
            log.exception("autostart.catchup_failed")

        # Hand off to a graph-enabled indexer for steady-state watch_forever.
        # In transient mode this is fine: per-file events incur a small
        # open/close cost on a small graph.kz.
        indexer = Indexer(
            settings, embedder, vec,
            lexical_store=lex,
            graph_store=GraphIngester.transient(settings.kuzu_dir),
            file_hashes=hashes,
        )

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
        # Run the file watcher and the LM Studio zombie-instance janitor in
        # parallel. The janitor periodically detects duplicate auto-loaded
        # model instances (e.g. `<model>:2`) and unloads them — recovers the
        # ~2.5 GB VRAM each duplicate would otherwise hold. Best-effort: a
        # janitor crash never propagates into the watcher.
        from code_rag.util.lm_janitor import janitor_loop
        janitor_cfg = getattr(settings, "lm_janitor", None)
        janitor_enabled = getattr(janitor_cfg, "enabled", True) if janitor_cfg else True
        janitor_interval = float(getattr(janitor_cfg, "interval_s", 60.0)
                                 if janitor_cfg else 60.0)
        if janitor_enabled:
            # Phase 38 (audit fix): the bare `asyncio.gather` cancels the
            # watcher if the janitor ever raises. The janitor's internal
            # try/except is broad, but defensive: spawn the janitor as a
            # FIRE-AND-FORGET task whose exceptions are logged but never
            # propagate. The watcher is the primary task; if it exits the
            # main coroutine returns and the janitor task is cancelled
            # cleanly in the finally block.
            janitor_task = asyncio.create_task(
                janitor_loop(settings.embedder.base_url, janitor_interval),
                name="lm_janitor_loop",
            )
            try:
                await watch_forever(settings, indexer)
            finally:
                if not janitor_task.done():
                    janitor_task.cancel()
                with contextlib.suppress(BaseException):
                    await janitor_task
                # Surface any non-cancel exception from the janitor for
                # debuggability (already logged inside the janitor's loop).
                if (janitor_task.done()
                        and not janitor_task.cancelled()
                        and janitor_task.exception() is not None):
                    err = janitor_task.exception()
                    _plain_log(
                        autostart_log,
                        f"lm_janitor exited with {type(err).__name__}: {err}",
                    )
        else:
            await watch_forever(settings, indexer)
        _plain_log(autostart_log, "watcher exited normally")
        return 0
    finally:
        vec.close()
        lex.close()
        with contextlib.suppress(Exception):
            hashes.close()
        # Release the Phase 32 singleton lock.
        with contextlib.suppress(Exception):
            lock.__exit__(None, None, None)


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

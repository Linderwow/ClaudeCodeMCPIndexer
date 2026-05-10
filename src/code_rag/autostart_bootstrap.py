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

    # Phase 39 + Phase 42: respect the Stop All marker.
    #
    # The user contract is: "stay stopped until I hit Start All OR restart
    # the PC". The previous implementation assumed this task only fires
    # AtLogon, so it cleared the marker on every fire. False — the task
    # has `-RestartCount 5 -RestartInterval 1 min`, so when Stop All
    # terminates the python process Task Scheduler interprets the non-zero
    # exit as "task crashed" and respawns the action up to 5 times.
    # Clearing the marker on every fire defeats Stop All any time that
    # restart policy kicks in.
    #
    # Phase 42 fix: only clear the marker when it predates the last system
    # boot (proves a real reboot happened since Stop All). Otherwise: bail
    # immediately. The watcher stays dead until Start All clears the
    # marker explicitly.
    from code_rag.util.stop_marker import (
        clear_intentionally_stopped,
        is_intentionally_stopped,
        marker_predates_last_boot,
    )
    if is_intentionally_stopped(settings.paths.data_dir):
        if marker_predates_last_boot(settings.paths.data_dir):
            clear_intentionally_stopped(settings.paths.data_dir,
                                        reason="autostart_bootstrap.post_reboot")
            _plain_log(autostart_log,
                       "cleared stale Stop All marker (PC restart resumes service)")
        else:
            _plain_log(autostart_log,
                       "Stop All marker present and post-Stop-All boot — "
                       "respecting user intent, bailing without starting watcher")
            log.info("autostart.bail.stop_marker_in_session")
            return

    # 2) Ensure the embedder server is up.
    #
    # Phase 60-H (Pattern B autonomy): the bootstrap is the canonical
    # entry-point for "code-rag should be alive". On Windows logon, the
    # `code-rag-watch` scheduled task fires this — so this is the right
    # place to (re)launch vLLM if it's down, instead of relying on a
    # separate daily timed task.
    #
    # Stop-marker-aware: if `data/.stopped` is present (set by the
    # dashboard's Stop button), we DO NOT launch vLLM. The user said
    # "stay stopped" — respect that across logon. Earlier in this same
    # bootstrap function we already check the stop marker and bail out
    # before reaching this point, so getting here means the user wants
    # the stack up.
    from urllib.parse import urlparse
    base_url = settings.embedder.base_url
    try:
        is_vllm = urlparse(base_url).port == 8000
    except (ValueError, AttributeError):
        is_vllm = False

    if is_vllm:
        try:
            from code_rag.lms_ctl import server_is_up
            up = server_is_up(base_url, timeout_s=2.0)
        except Exception:
            up = False

        if up:
            _plain_log(autostart_log, f"vllm probe: up at {base_url}")
        else:
            _plain_log(
                autostart_log,
                f"vllm probe: down at {base_url} — invoking resume script"
            )
            # Invoke the same idempotent script the manual "Start" button uses.
            # Spawn detached so a slow vLLM cold-start (~30 s) doesn't block
            # the bootstrap from getting on with the watcher.
            import subprocess as _sp
            ps_script = (
                Path(__file__).parent.parent.parent
                / "scripts" / "resume_code_rag.ps1"
            )
            if ps_script.exists():
                try:
                    _sp.Popen(
                        ["powershell.exe", "-NoProfile", "-ExecutionPolicy",
                         "Bypass", "-File", str(ps_script)],
                        stdout=_sp.DEVNULL, stderr=_sp.DEVNULL,
                        stdin=_sp.DEVNULL, close_fds=True,
                        creationflags=0x08000000,  # CREATE_NO_WINDOW
                    )
                    _plain_log(
                        autostart_log,
                        f"  spawned {ps_script.name} (detached); vLLM coming up"
                    )
                except OSError as e:
                    _plain_log(autostart_log,
                               f"  FAILED to spawn resume script: {e}")
            else:
                _plain_log(
                    autostart_log,
                    f"  WARN: resume script missing at {ps_script} "
                    "— vLLM stays down until manual Start"
                )
    else:
        # Legacy LM Studio path (kind="lm_studio" pointing at :1234, etc.).
        # Pre-load any auxiliary chat/rerank models so the first MCP query
        # after boot doesn't pay a 5-10s JIT-load penalty.
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
            # Don't exit — the watcher can still run (it just won't reindex
            # on changes until LM Studio comes up, which might happen
            # manually). Log and proceed.
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
            # Phase 60-O (audit C2): `embedder.dim` is a property that RAISES
            # when not yet probed -- the previous `embedder.dim if embedder.dim
            # > 0 else 1` ternary couldn't actually short-circuit because the
            # first read raises before the comparison runs. Read the underlying
            # `_dim` attribute (default 0) so the guard works.
            _raw_dim = getattr(embedder, "_dim", 0)
            embedder_dim = _raw_dim if _raw_dim > 0 else 1
            meta = ChromaVectorStore.build_meta(
                embedder_kind=settings.embedder.kind,
                embedder_model=embedder.model,
                embedder_dim=embedder_dim,
            )
            # Phase 60-O (audit C1): chromadb's `PersistentClient(path=...)`
            # runs SQLite WAL recovery on open with NO timeout. A half-corrupt
            # chroma.sqlite3 from an unclean shutdown can hang here forever
            # while the bootstrap holds watcher.lock -- the user observed a
            # 30-min zombie holding the lock that never reached
            # "watcher starting." Wrap in an asyncio.wait_for so a hang is
            # surfaced rather than silently consuming the boot.
            _plain_log(autostart_log,
                       f"opening chroma (dim={embedder_dim})...")
            await asyncio.wait_for(
                asyncio.to_thread(vec.open, meta), timeout=120.0,
            )
            _plain_log(autostart_log, "chroma opened")
            await asyncio.wait_for(
                asyncio.to_thread(lex.open), timeout=60.0,
            )
            _plain_log(autostart_log, "lex opened")
        except asyncio.TimeoutError:
            _plain_log(autostart_log,
                       "open stores TIMEOUT (>120s on chroma OR >60s on lex) "
                       "-- likely WAL recovery on a corrupt sqlite. Bailing "
                       "with rc=4 so Task Scheduler restart can retry.")
            log.exception("autostart.open_timeout")
            return 4
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

        # Phase 60-K: chroma-wipe detector. Overnight unclean shutdowns
        # (sleep mid-write, WSL crash, the auto-stop loop's pkill firing
        # while a chroma write was in flight) can leave chromadb in a
        # state where its own integrity check resets the embeddings table
        # to zero on next open. fts.db isn't affected (independent SQLite
        # WAL), so we end up with N chunks and 0 vectors — search dense-
        # arm dies, cross-encoder rerank gets nothing useful to re-rank,
        # MCP results visibly degrade.
        #
        # Without this guard, the catch-up reindex below sees file_hashes
        # full ("we already indexed all of these") and is a no-op against
        # the empty chroma. Detect + clear file_hashes so the catch-up
        # actually re-vectorizes everything.
        # Audit fix: probe each store's count individually so a chromadb
        # internal error (which is the very signal we're trying to
        # detect) doesn't disable the detector. The fts threshold is
        # `> 0` not `> 100` so small repos are also covered — the
        # `file_hashes > 0` co-condition keeps us from false-positive
        # firing on a brand-new install.
        chroma_count = -1
        fts_count = -1
        hash_count = -1
        try:
            chroma_count = vec.count()
        except Exception as e:
            _plain_log(autostart_log, f"wipe-detector chroma.count failed: {e}")
        try:
            fts_count = lex.count() if hasattr(lex, "count") else -1
        except Exception as e:
            _plain_log(autostart_log, f"wipe-detector lex.count failed: {e}")
        try:
            hash_count = hashes.count()
        except Exception as e:
            _plain_log(autostart_log, f"wipe-detector hashes.count failed: {e}")

        # Phase 60-O (audit M2): treat fts_count==-1 (probe failure) as
        # "unknown but consistent with wipe" so the SQLite-locked-by-watcher
        # case during autostart restart doesn't silently disable detection.
        # The chroma==0 + hashes>0 invariant already proves the wipe; fts
        # was just our cross-check.
        fts_consistent_with_wipe = (fts_count > 0) or (fts_count == -1)
        if chroma_count == 0 and fts_consistent_with_wipe and hash_count > 0:
            _plain_log(
                autostart_log,
                f"WIPE DETECTED: chroma=0 vectors but fts={fts_count} chunks "
                f"and file_hashes={hash_count} entries. Clearing file_hashes "
                "to force a full re-vectorize via the catch-up loop below."
            )
            try:
                cleared = hashes.clear_all()
                _plain_log(
                    autostart_log,
                    f"  cleared {cleared} file_hash entries; catch-up will re-embed everything"
                )
                log.warning("autostart.chroma_wipe_detected",
                            fts=fts_count, file_hashes_cleared=cleared)
                # Phase 60-M: write a sticky marker the dashboard can read.
                # Without this, the dashboard's wipe_recovery state classifier
                # races: vectors transitions 0 -> N+ within the first poll
                # interval, so the briefly-zero window is invisible. The
                # marker lives until the dashboard sees vectors back to ~95%
                # of fts and clears it (see operations._index_status).
                try:
                    import json as _json
                    import time as _t
                    wipe_marker = settings.paths.data_dir / ".wipe_recovery"
                    wipe_marker.write_text(_json.dumps({
                        "detected_at": _t.time(),
                        "fts_at_detect": fts_count,
                        "file_hashes_cleared": cleared,
                    }), encoding="utf-8")
                    _plain_log(autostart_log,
                               f"  wrote {wipe_marker.name} marker for dashboard")
                except Exception as me:
                    _plain_log(autostart_log,
                               f"  WARN: could not write wipe_recovery marker: {me}")
            except Exception as e:
                _plain_log(autostart_log,
                           f"WARN: file_hashes.clear_all failed: {e} — "
                           "catch-up will be a no-op until file_hashes is wiped manually")

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

        # Phase 60-I: auto-stop loop. Kills vLLM after N hours of no
        # chroma writes so YouTubeBot's ComfyUI render can claim the
        # GPU. Wakes back up on the next watcher file event. Disabled
        # by default — user opts in via [auto_stop] enabled=true in
        # config.toml. Uses the same fire-and-forget pattern as the
        # janitor so a glitch can't kill the watcher.
        from code_rag.util.auto_stop_loop import auto_stop_loop
        autostop_cfg = getattr(settings, "auto_stop", None)
        autostop_enabled = (
            getattr(autostop_cfg, "enabled", False)
            if autostop_cfg else False
        )
        autostop_interval = float(
            getattr(autostop_cfg, "interval_s", 60.0)
            if autostop_cfg else 60.0
        )
        autostop_idle_seconds = float(
            getattr(autostop_cfg, "idle_seconds", 7200)
            if autostop_cfg else 7200
        )
        autostop_task: asyncio.Task | None = None
        if autostop_enabled:
            autostop_task = asyncio.create_task(
                auto_stop_loop(
                    settings,
                    interval_s=autostop_interval,
                    idle_seconds=autostop_idle_seconds,
                ),
                name="auto_stop_loop",
            )
            _plain_log(
                autostart_log,
                f"auto_stop loop started: idle_seconds={autostop_idle_seconds:.0f}s, "
                f"interval_s={autostop_interval:.0f}s"
            )

        # Phase 60-M: round-4 audit minor — startup probe of /api/v0/models
        # before launching the janitor loop. Post-Phase-60 the embedder.kind
        # is historically "lm_studio" but the URL points at vLLM, which
        # returns 404 on the LM Studio admin API. The janitor would happily
        # poll every interval_s for the lifetime of the watcher (one wasted
        # HTTP per cycle). The probe lets us short-circuit when the admin
        # API isn't really there.
        if janitor_enabled:
            try:
                import httpx as _hx
                _root = settings.embedder.base_url.rstrip("/")
                if _root.endswith("/v1"):
                    _root = _root[: -len("/v1")]
                _probe = _hx.get(f"{_root}/api/v0/models", timeout=2.0)
                if _probe.status_code == 404:
                    janitor_enabled = False
                    _plain_log(
                        autostart_log,
                        "lm_janitor disabled: /api/v0/models returns 404 "
                        "(server is vLLM/TEI/sglang, not LM Studio)"
                    )
            except Exception as e:
                # Server unreachable -- janitor would also fail; skip cleanly.
                janitor_enabled = False
                _plain_log(
                    autostart_log,
                    f"lm_janitor disabled: probe failed: {type(e).__name__}: {e}"
                )

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
            except Exception as e:
                # Phase 60-O (audit M4): surface watch_forever crashes to the
                # plain autostart.log so post-mortems don't have to dig
                # through ~/.code-rag-autostart-failure.log.
                _plain_log(
                    autostart_log,
                    f"watcher CRASHED: {type(e).__name__}: {e}",
                )
                raise
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
                # Phase 60-I: also tear down the auto-stop loop if it ran.
                if autostop_task is not None and not autostop_task.done():
                    autostop_task.cancel()
                    with contextlib.suppress(BaseException):
                        await autostop_task
        else:
            try:
                await watch_forever(settings, indexer)
            except Exception as e:
                _plain_log(
                    autostart_log,
                    f"watcher CRASHED: {type(e).__name__}: {e}",
                )
                raise
            finally:
                # Phase 60-I: even when the janitor is disabled, the
                # auto-stop task may be running. Tear it down on exit.
                if autostop_task is not None and not autostop_task.done():
                    autostop_task.cancel()
                    with contextlib.suppress(BaseException):
                        await autostop_task
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

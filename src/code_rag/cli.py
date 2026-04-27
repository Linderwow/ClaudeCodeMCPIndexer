from __future__ import annotations

import asyncio
import contextlib
import json
import subprocess
import sys
from pathlib import Path

import click

from code_rag import __version__
from code_rag.config import load_settings
from code_rag.embedders.lm_studio import LMStudioEmbedder
from code_rag.factory import (
    build_embedder,
    build_graph_store,
    build_lexical_store,
    build_query_rewriter,
    build_reranker,
    build_vector_store,
)
from code_rag.graph.ingest import GraphIngester
from code_rag.indexing.indexer import Indexer
from code_rag.logging import configure, get
from code_rag.rerankers.lm_studio import LMStudioReranker
from code_rag.retrieval.search import HybridSearcher, SearchParams
from code_rag.stores.chroma_vector import ChromaVectorStore

log = get(__name__)


def _force_utf8_streams() -> None:
    """Windows' default console is cp1252 and chokes on em-dashes, arrows, etc.
    Reconfigure stdout/stderr to UTF-8 so help text and logs always render.
    Called at import time so `--help` (which fires before Click invokes main())
    benefits too."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            with contextlib.suppress(OSError, ValueError):
                reconfigure(encoding="utf-8", errors="replace")


_force_utf8_streams()


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="code-rag")
@click.option("--log-level", default="INFO", show_default=True,
              type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False))
@click.pass_context
def main(ctx: click.Context, log_level: str) -> None:
    """code-rag-mcp -- local code + doc RAG as an MCP server."""
    settings = load_settings()
    configure(settings.paths.log_dir, level=log_level.upper())
    ctx.ensure_object(dict)
    ctx.obj["settings"] = settings


@main.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.pass_context
def add_root(ctx: click.Context, path: Path) -> None:
    """Add a codebase directory to the indexed set.

    Appends to [paths].roots in config.toml without disturbing other content.
    Run `code-rag index --path <path>` afterwards to build its chunks.
    """
    from code_rag.roots_edit import ConfigEditError
    from code_rag.roots_edit import add_root as _add
    try:
        roots, added = _add(_resolved_config_path(ctx), path.resolve())
    except ConfigEditError as e:
        click.echo(f"FAIL  {e}", err=True)
        sys.exit(1)
    if added:
        click.echo(f"added: {path.resolve()}")
        click.echo(f"  now indexing {len(roots)} root(s). Run `code-rag index --path \"{path.resolve()}\"` to build it.")
    else:
        click.echo(f"already indexed: {path.resolve()}")


@main.command()
@click.argument("path", type=click.Path(path_type=Path))
@click.pass_context
def remove_root(ctx: click.Context, path: Path) -> None:
    """Remove a codebase directory from the indexed set.

    Does not delete the on-disk index for that root -- call `code-rag index`
    after removing if you want the store pruned.
    """
    from code_rag.roots_edit import remove_root as _remove
    roots, removed = _remove(_resolved_config_path(ctx), path)
    if removed:
        click.echo(f"removed: {path}")
        click.echo(f"  {len(roots)} root(s) remain.")
    else:
        click.echo(f"not in config: {path}")


def _resolved_config_path(ctx: click.Context) -> Path:
    """Return the config.toml path actually being used (env override or default)."""
    import os
    env = os.environ.get("CODE_RAG_CONFIG")
    if env:
        return Path(env)
    from code_rag.config import DEFAULT_CONFIG_PATH
    return DEFAULT_CONFIG_PATH


@main.command()
@click.pass_context
def config(ctx: click.Context) -> None:
    """Print the resolved config (no secrets in here to redact)."""
    settings = ctx.obj["settings"]
    click.echo(settings.model_dump_json(indent=2))


@main.command()
@click.pass_context
def ping(ctx: click.Context) -> None:
    """Probe LM Studio: list models, confirm configured embedder is loaded, print dim."""
    settings = ctx.obj["settings"]

    async def _run() -> int:
        emb = LMStudioEmbedder(
            base_url=settings.embedder.base_url,
            model=settings.embedder.model,
            dim=settings.embedder.dim,
            timeout_s=settings.embedder.timeout_s,
            batch=settings.embedder.batch,
        )
        try:
            await emb.health()
            click.echo(f"OK  embedder={settings.embedder.model}  dim={emb.dim}")
            return 0
        except Exception as e:
            click.echo(f"FAIL  {type(e).__name__}: {e}", err=True)
            return 1
        finally:
            await emb.aclose()

    sys.exit(asyncio.run(_run()))


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Summarize configured roots and data dir. Indexing-free -- safe to run anytime."""
    settings = ctx.obj["settings"]
    click.echo(f"version      {__version__}")
    click.echo(f"data_dir     {settings.paths.data_dir}")
    click.echo(f"log_dir      {settings.paths.log_dir}")
    click.echo(f"roots ({len(settings.paths.roots)}):")
    for r in settings.paths.roots:
        exists = "OK " if r.exists() else "MISS"
        click.echo(f"  [{exists}] {r}")
    click.echo(f"embedder     {settings.embedder.kind}:{settings.embedder.model}")
    click.echo(f"reranker     {settings.reranker.kind}:{settings.reranker.model}")
    click.echo(f"vector       {settings.vector_store.kind}:{settings.vector_store.collection}")
    click.echo(f"graph        {settings.graph_store.kind}")
    click.echo(f"lexical      {settings.lexical_store.kind}")


@main.command()
@click.option("--path", "path_", type=click.Path(path_type=Path),
              help="Reindex a single file or subtree. Omit to reindex everything.")
@click.pass_context
def index(ctx: click.Context, path_: Path | None) -> None:
    """Build or refresh the code index (walk -> chunk -> embed -> Chroma)."""
    settings = ctx.obj["settings"]

    async def _run() -> int:
        from code_rag.indexing.file_hash import FileHashRegistry
        embedder = build_embedder(settings)
        vec = build_vector_store(settings)
        lex = build_lexical_store(settings)
        graph = build_graph_store(settings)
        hashes = FileHashRegistry(settings.file_hashes_path)
        try:
            await embedder.health()
        except Exception as e:
            click.echo(f"FAIL  embedder health: {type(e).__name__}: {e}", err=True)
            return 1
        meta = ChromaVectorStore.build_meta(
            embedder_kind=settings.embedder.kind,
            embedder_model=embedder.model,
            embedder_dim=embedder.dim,
        )
        try:
            vec.open(meta)
            lex.open()
            graph.open()
            hashes.open()
        except Exception as e:
            click.echo(f"FAIL  index open: {e}", err=True)
            return 1
        try:
            indexer = Indexer(
                settings, embedder, vec,
                lexical_store=lex,
                graph_store=GraphIngester(graph),
                file_hashes=hashes,
            )
            stats = (await indexer.reindex_path(path_)) if path_ else (await indexer.reindex_all())
            click.echo(json.dumps(stats.as_dict(), indent=2))
            return 0
        finally:
            vec.close()
            lex.close()
            graph.close()
            hashes.close()
            await embedder.aclose()

    sys.exit(asyncio.run(_run()))


@main.command()
@click.argument("query", nargs=-1, required=True)
@click.option("-k", "--top-k", default=8, show_default=True, type=int)
@click.option("--path-glob", default=None, help="Filter hits by path glob (fnmatch).")
@click.option("--lang", default=None, help="Filter by language (python, c_sharp, typescript, tsx, javascript).")
@click.pass_context
def search(ctx: click.Context, query: tuple[str, ...], top_k: int,
           path_glob: str | None, lang: str | None) -> None:
    """Hybrid search: vector + lexical -> RRF -> rerank -> top-k."""
    settings = ctx.obj["settings"]
    query_str = " ".join(query)

    async def _run() -> int:
        embedder = build_embedder(settings)
        vec = build_vector_store(settings)
        lex = build_lexical_store(settings)
        reranker = build_reranker(settings)

        if not settings.index_meta_path.exists():
            click.echo("no index yet -- run `code-rag index` first", err=True)
            return 1

        try:
            await embedder.health()
        except Exception as e:
            click.echo(f"FAIL  embedder: {e}", err=True)
            return 1

        meta = ChromaVectorStore.build_meta(
            embedder_kind=settings.embedder.kind,
            embedder_model=embedder.model,
            embedder_dim=embedder.dim,
        )
        try:
            vec.open(meta)
            lex.open()
        except Exception as e:
            click.echo(f"FAIL  open: {e}", err=True)
            return 1

        try:
            searcher = HybridSearcher(embedder, vec, lex, reranker)
            params = SearchParams(
                k_final=top_k,
                k_vector=settings.reranker.top_k_in,
                k_lexical=settings.reranker.top_k_in,
                k_rerank_in=settings.reranker.top_k_in,
                path_glob=path_glob,
                language=lang,
            )
            hits = await searcher.search(query_str, params)
            for i, h in enumerate(hits, start=1):
                click.echo(f"{i:>2}. [{h.source} {h.score:+.3f}] {h.chunk.path}"
                           f":{h.chunk.start_line}-{h.chunk.end_line}  {h.chunk.symbol or ''}")
            if not hits:
                click.echo("(no hits)")
            return 0
        finally:
            vec.close()
            lex.close()
            await embedder.aclose()
            if isinstance(reranker, LMStudioReranker):
                await reranker.aclose()

    sys.exit(asyncio.run(_run()))


@main.command()
@click.option("--skip-probe",     is_flag=True, help="Don't probe LM Studio.")
@click.option("--skip-index",     is_flag=True, help="Don't build the initial index.")
@click.option("--skip-claude",    is_flag=True, help="Don't touch any Claude config.")
@click.option("--skip-autostart", is_flag=True, help="Don't register the boot-time watcher.")
@click.option("--force-reindex",  is_flag=True, help="Rebuild the index even if one exists.")
@click.pass_context
def install(
    ctx: click.Context,
    skip_probe: bool, skip_index: bool, skip_claude: bool, skip_autostart: bool,
    force_reindex: bool,
) -> None:
    """Idempotent one-shot setup: probe LM Studio, build the initial index,
    wire the MCP server into every Claude config we find, and register the
    Task Scheduler autostart. Safe to rerun -- each step is a no-op if already done.
    """
    from code_rag.install import InstallOptions, run_install_sync
    settings = ctx.obj["settings"]
    opts = InstallOptions(
        skip_probe=skip_probe,
        skip_index=skip_index,
        skip_claude=skip_claude,
        skip_autostart=skip_autostart,
        force_reindex=force_reindex,
    )
    report = run_install_sync(settings, opts)
    click.echo("code-rag install:")
    for step in report.steps:
        click.echo(step.fmt())
    click.echo()
    if report.ok:
        click.echo("=== install complete ===")
        sys.exit(0)
    else:
        click.echo("=== install finished with failures (see above) ===", err=True)
        sys.exit(1)


@main.command()
@click.pass_context
def doctor(ctx: click.Context) -> None:
    """One-shot readiness check: config, roots, LM Studio, index meta.

    Exit code 0 if everything's live, 1 if any check fails. Safe to invoke
    from Task Scheduler's "if command fails, retry" policy.
    """
    settings = ctx.obj["settings"]
    failed = 0

    async def _run() -> int:
        nonlocal failed

        def ok(label: str) -> None:
            click.echo(f"  [OK]   {label}")

        def fail(label: str, why: str) -> None:
            nonlocal failed
            failed += 1
            click.echo(f"  [FAIL] {label}: {why}", err=True)

        click.echo(f"code-rag doctor  v{__version__}")
        click.echo("--- config ---")
        ok(f"config file: {settings.paths.data_dir}")
        for r in settings.paths.roots:
            (ok if r.exists() else lambda lbl: fail(lbl, "missing"))(f"root {r}")

        click.echo("--- embedder (LM Studio) ---")
        emb = build_embedder(settings)
        try:
            await emb.health()
            ok(f"{settings.embedder.model} dim={emb.dim}")
        except Exception as e:
            fail("embedder.health", f"{type(e).__name__}: {e}")
        finally:
            if isinstance(emb, LMStudioEmbedder):
                await emb.aclose()

        click.echo("--- reranker (LM Studio) ---")
        rer = build_reranker(settings)
        try:
            await rer.health()
            ok(f"{settings.reranker.model}")
        except Exception as e:
            # Reranker is best-effort (search falls back to no-op).
            fail("reranker.health", f"{type(e).__name__}: {e}")
        finally:
            if isinstance(rer, LMStudioReranker):
                await rer.aclose()

        click.echo("--- index ---")
        if settings.index_meta_path.exists():
            meta = json.loads(settings.index_meta_path.read_text("utf-8"))
            ok(f"meta: {meta.get('embedder_model')} dim={meta.get('embedder_dim')} "
               f"updated={meta.get('updated_at')}")
            # Mismatch between config embedder and indexed embedder is fatal —
            # indices from different models are incompatible (different dims).
            if meta.get("embedder_model") and meta["embedder_model"] != settings.embedder.model:
                # Allow preset-routed mismatch where the index used the preset's
                # canonical lms_id instead of the literal `model` field.
                preset = getattr(settings.embedder, "preset", None)
                if preset:
                    from code_rag.embedders.code_specialized import CODE_EMBEDDER_PRESETS
                    expected = CODE_EMBEDDER_PRESETS.get(preset)
                    if expected and expected.lms_id == meta["embedder_model"]:
                        pass  # consistent via preset
                    else:
                        fail("index meta", f"embedder mismatch: meta={meta['embedder_model']} != config={settings.embedder.model}")
                else:
                    fail("index meta", f"embedder mismatch: meta={meta['embedder_model']} != config={settings.embedder.model}")

            vec_count: int | None = None
            lex_count: int | None = None
            if settings.chroma_dir.exists():
                vec = build_vector_store(settings)
                try:
                    from code_rag.models import IndexMeta
                    meta_obj = IndexMeta.model_validate_json(
                        settings.index_meta_path.read_text("utf-8"),
                    )
                    vec.open(meta_obj)
                    vec_count = vec.count()
                    ok(f"vector chunks: {vec_count}")
                except Exception as e:
                    fail("vec.count", str(e))
                finally:
                    with contextlib.suppress(Exception):
                        vec.close()
            if settings.fts_path.exists():
                lex = build_lexical_store(settings)
                try:
                    lex.open()
                    lex_count = lex.count()
                    ok(f"lexical chunks: {lex_count}")
                finally:
                    with contextlib.suppress(Exception):
                        lex.close()
            # Drift detection: vector vs lexical counts must match exactly,
            # otherwise hybrid search has phantom hits in one but not the other.
            if vec_count is not None and lex_count is not None:
                if vec_count == lex_count:
                    ok("stores in sync (no drift)")
                else:
                    fail("stores", f"DRIFT: vec={vec_count} lex={lex_count}; run `code-rag fsck --fix`")
        else:
            fail("index", "no index yet -- run `code-rag index`")

        click.echo("--- dynamic roots ---")
        try:
            from code_rag.dynamic_roots import DynamicRoots
            dyn = DynamicRoots.load(settings.dynamic_roots_path)
            live = sum(1 for e in dyn.entries if e.path.exists())
            dead = len(dyn.entries) - live
            if not dyn.entries:
                ok("none registered (pure auto-discovery; Claude registers on demand)")
            elif dead == 0:
                ok(f"{live} dynamic root(s), all live")
            else:
                fail("dynamic_roots", f"{live} live, {dead} dead (run `code-rag fsck --fix`)")
        except Exception as e:
            fail("dynamic_roots", str(e))

        click.echo("--- autostart ---")
        import platform as _pf
        import shutil
        sys_name = _pf.system()
        if sys_name == "Windows":
            import subprocess as _sp
            try:
                r = _sp.run(["schtasks", "/Query", "/TN", "code-rag-watch"],
                            capture_output=True, text=True, check=False)
                if r.returncode == 0:
                    ok("Task Scheduler entry 'code-rag-watch' registered")
                else:
                    fail("autostart", "Task Scheduler entry MISSING; run `code-rag install`")
            except FileNotFoundError:
                fail("autostart", "schtasks not on PATH")
        elif sys_name == "Darwin":
            plist = Path.home() / "Library/LaunchAgents/com.code-rag-mcp.watcher.plist"
            if plist.exists():
                ok(f"launchd agent installed at {plist}")
            else:
                fail("autostart", "launchd agent missing; run `scripts/setup.sh`")
        elif sys_name == "Linux":
            if shutil.which("systemctl"):
                import subprocess as _sp
                r = _sp.run(["systemctl", "--user", "is-enabled", "code-rag-watcher.service"],
                            capture_output=True, text=True, check=False)
                if r.returncode == 0:
                    ok("systemd user unit 'code-rag-watcher.service' enabled")
                else:
                    fail("autostart", "systemd user unit not enabled; run `scripts/setup.sh`")

        click.echo()
        if failed:
            click.echo(f"=== {failed} check(s) failed ===", err=True)
            return 1
        click.echo("=== all green ===")
        return 0

    sys.exit(asyncio.run(_run()))


@main.command()
@click.pass_context
def mcp(ctx: click.Context) -> None:
    """Run as an MCP stdio server. Claude Code spawns this and reuses stdin/stdout."""
    from code_rag.mcp_server.server import main_entry
    main_entry()


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Bind address. KEEP at 127.0.0.1 unless you know what you're doing.")
@click.option("--port", default=7321, show_default=True, type=int)
@click.option("--no-browser", is_flag=True,
              help="Don't auto-open the dashboard in a web browser.")
@click.pass_context
def dashboard(ctx: click.Context, host: str, port: int, no_browser: bool) -> None:
    """Launch the local control dashboard in a web browser.

    Big START/STOP buttons for LM Studio + the watcher + loaded models. Polls
    /api/status every 2s so the UI is always fresh. Ctrl-C in this terminal
    to stop the dashboard server (the actual stack keeps running).

    Under pythonw (no attached console), uvicorn would crash trying to write
    to a None stderr. We detect that case and route stdout+stderr to a log
    file in `<data_dir>/../logs/dashboard.log` so the autostart task stays up.
    """
    settings = ctx.obj["settings"]
    if sys.stdout is None or sys.stderr is None:
        log_path = settings.paths.log_dir / "dashboard.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # of the dashboard process so uvicorn's per-request log lines flush.
        log_file = log_path.open("a", encoding="utf-8", buffering=1)
        sys.stdout = log_file
        sys.stderr = log_file

    # Phase 32: refuse to start a second dashboard instance. Task Scheduler's
    # IgnoreNew policy covers same-task spawns, but a manual `schtasks /Run`
    # or a venv-launcher quirk can still produce duplicates.
    from code_rag.util.proc_hygiene import SingletonLock
    lock_path = settings.paths.data_dir / "dashboard.lock"
    with SingletonLock(lock_path) as lock:
        if not lock.acquired:
            click.echo(
                f"another dashboard instance is already running "
                f"(see {lock_path}); exiting.", err=True,
            )
            sys.exit(0)
        from code_rag.dashboard.server import serve
        click.echo(f"code-rag dashboard -> http://{host}:{port}/")
        click.echo("Ctrl-C to stop (the LM Studio + watcher stack keeps running).")
        serve(host=host, port=port, open_browser=not no_browser)


@main.command()
@click.pass_context
def bootstrap(ctx: click.Context) -> None:
    """Ensure LM Studio is running with the embedder loaded (what autostart does).

    Idempotent. Useful for manually verifying the boot chain works before
    registering autostart, or for kicking the server back up after a crash.
    """
    settings = ctx.obj["settings"]
    from code_rag.lms_ctl import ensure_lm_studio_ready
    extra: tuple[str, ...] = ()
    if settings.reranker.kind == "lm_studio" and settings.reranker.model:
        extra = (settings.reranker.model,)
    result = ensure_lm_studio_ready(
        settings.embedder.base_url,
        settings.embedder.model,
        extra,
    )
    click.echo("bootstrap steps:")
    for s in result.steps:
        click.echo(f"  - {s}")
    if result.ok:
        click.echo("=== LM Studio ready ===")
        sys.exit(0)
    click.echo(f"FAIL  {result.error}", err=True)
    sys.exit(1)


@main.command()
@click.pass_context
def watch(ctx: click.Context) -> None:
    """Watch configured roots and incrementally reindex on change.

    Debounced (config.toml [watcher].debounce_ms). Safe to run unattended --
    designed to be invoked at login by Task Scheduler (see docs/autostart.md).

    Phase 32: refuses to start a second watcher (singleton lockfile). Two
    watchers indexing the same Chroma collection at once would race on
    upsert/delete; the lock makes that physically impossible.
    """
    settings = ctx.obj["settings"]
    from code_rag.util.proc_hygiene import SingletonLock
    lock_path = settings.paths.data_dir / "watcher.lock"
    lock = SingletonLock(lock_path).__enter__()
    if not lock.acquired:
        click.echo(
            f"another watcher instance is already running "
            f"(see {lock_path}); exiting.", err=True,
        )
        sys.exit(0)

    async def _run() -> int:
        embedder = build_embedder(settings)
        vec = build_vector_store(settings)
        lex = build_lexical_store(settings)
        try:
            await embedder.health()
        except Exception as e:
            click.echo(f"FAIL  embedder health: {e}", err=True)
            return 1
        meta = ChromaVectorStore.build_meta(
            embedder_kind=settings.embedder.kind,
            embedder_model=embedder.model,
            embedder_dim=embedder.dim,
        )
        try:
            vec.open(meta)
            lex.open()
        except Exception as e:
            click.echo(f"FAIL  open: {e}", err=True)
            return 1
        try:
            from code_rag.watcher.live import watch_forever
            # Transient graph ingest: Kuzu is opened per-event and closed
            # immediately after, so MCP/CLI readers can query the graph in
            # the gaps. See code_rag/graph/ingest.py for rationale.
            indexer = Indexer(
                settings, embedder, vec,
                lexical_store=lex,
                graph_store=GraphIngester.transient(settings.kuzu_dir),
            )
            await indexer.reindex_all()
            await watch_forever(settings, indexer)
            return 0
        finally:
            vec.close()
            lex.close()
            await embedder.aclose()

    try:
        sys.exit(asyncio.run(_run()))
    except KeyboardInterrupt:
        click.echo("watcher stopped", err=True)
        sys.exit(0)
    finally:
        # Release the singleton lock so the next watcher start can acquire.
        lock.__exit__(None, None, None)


@main.command()
@click.argument("symbol", required=True)
@click.option("--path", "path_", default=None, help="Optional: limit to this file path.")
@click.pass_context
def callers(ctx: click.Context, symbol: str, path_: str | None) -> None:
    """List symbols that CALL the given symbol."""
    settings = ctx.obj["settings"]
    graph = build_graph_store(settings, read_only=True)
    try:
        graph.open()
        refs = graph.callers_of(symbol, path=path_)
        for r in refs:
            click.echo(f"{r.path}:{r.start_line}  {r.symbol}  [{r.kind}]")
        if not refs:
            click.echo("(no callers found)")
    finally:
        graph.close()


@main.command()
@click.argument("symbol", required=True)
@click.option("--path", "path_", default=None, help="Optional: limit to this file path.")
@click.pass_context
def callees(ctx: click.Context, symbol: str, path_: str | None) -> None:
    """List symbols that the given symbol CALLS."""
    settings = ctx.obj["settings"]
    graph = build_graph_store(settings, read_only=True)
    try:
        graph.open()
        refs = graph.callees_of(symbol, path=path_)
        for r in refs:
            tag = "[extern]" if r.path == "<extern>" else f"{r.path}:{r.start_line}"
            click.echo(f"{tag}  {r.symbol}  [{r.kind}]")
        if not refs:
            click.echo("(no callees found)")
    finally:
        graph.close()


@main.command()
@click.argument("symbol", required=True)
@click.option("--path", "path_", default=None)
@click.pass_context
def symbol(ctx: click.Context, symbol: str, path_: str | None) -> None:
    """Find a symbol by name -- useful to confirm a file was indexed."""
    settings = ctx.obj["settings"]
    graph = build_graph_store(settings, read_only=True)
    try:
        graph.open()
        refs = graph.find_symbol(symbol, path=path_)
        for r in refs:
            click.echo(f"{r.path}:{r.start_line}-{r.end_line}  {r.symbol}  [{r.kind}]")
        if not refs:
            click.echo("(not found)")
    finally:
        graph.close()


@main.command()
@click.argument("fixture", type=click.Path(exists=True, path_type=Path))
@click.option("-k", "--top-k", default=10, show_default=True, type=int)
@click.option("--json-out", "json_out", type=click.Path(path_type=Path), default=None,
              help="Write full per-case report here; summary still printed to stdout.")
@click.option("--label", default=None,
              help="Identifier for this run (e.g. 'after BGE swap'). Stored in the report.")
@click.option("--baseline", "baseline", type=click.Path(exists=True, path_type=Path), default=None,
              help="Path to a previous report's JSON. Compute pp-deltas vs it and print a diff.")
@click.option("--fail-on-regression", "fail_on_regression", default=None, type=float,
              help="Exit non-zero if any headline metric regresses by more than N pp vs --baseline.")
@click.pass_context
def eval(
    ctx: click.Context,
    fixture: Path, top_k: int, json_out: Path | None,
    label: str | None, baseline: Path | None, fail_on_regression: float | None,
) -> None:
    """Evaluate retrieval quality against a JSON fixture of (query, expected) pairs.

    Use `--baseline previous_report.json` to print pp-deltas in every metric
    so you can confirm a change (embedder swap, rerank tweak, chunker fix)
    actually moved the needle. Combine with `--fail-on-regression 2.0` for a
    CI gate that blocks commits that drop quality by >2 pp.
    """
    settings = ctx.obj["settings"]
    from code_rag.eval.harness import EvalReport, load_cases, run_eval

    async def _run() -> int:
        embedder = build_embedder(settings)
        vec = build_vector_store(settings)
        lex = build_lexical_store(settings)
        reranker = build_reranker(settings)
        if not settings.index_meta_path.exists():
            click.echo("no index yet -- run `code-rag index` first", err=True)
            return 1
        await embedder.health()
        meta = ChromaVectorStore.build_meta(
            embedder_kind=settings.embedder.kind,
            embedder_model=embedder.model,
            embedder_dim=embedder.dim,
        )
        vec.open(meta)
        lex.open()
        try:
            # Wire HyDE if a chat model is configured. Mirrors the MCP server's
            # construction (mcp_server/server.py::ServerResources.open) so the
            # eval measures the SAME pipeline production uses, not a leaner one.
            from code_rag.retrieval.hyde import HydeRetrieverPlan, LMStudioHyDEGenerator
            hyde_model = getattr(settings.embedder, "hyde_model", None)
            try:
                hyde_gen = (
                    LMStudioHyDEGenerator(
                        base_url=settings.embedder.base_url,
                        model=str(hyde_model),
                        timeout_s=15.0,
                    ) if hyde_model else None
                )
                hyde_plan = HydeRetrieverPlan(generator=hyde_gen)
            except Exception as e:
                click.echo(f"[warn] HyDE init failed, falling back to literal-only: {e}", err=True)
                hyde_plan = HydeRetrieverPlan(generator=None)
            searcher = HybridSearcher(
                embedder, vec, lex, reranker, hyde_plan=hyde_plan,
            )
            cases = load_cases(fixture)
            index_meta_dict = json.loads(settings.index_meta_path.read_text("utf-8"))
            # Call run_eval (async) directly -- run_eval_sync wraps asyncio.run
            # which can't nest inside this CLI's existing event loop.
            report = await run_eval(
                cases, searcher,
                top_k=top_k, rerank_in=settings.reranker.top_k_in,
                label=label, index_meta=index_meta_dict,
            )
            click.echo(json.dumps(report.summary(), indent=2))
            click.echo("\nper-tag:", err=True)
            click.echo(json.dumps(report.per_tag(), indent=2), err=True)

            if baseline is not None:
                baseline_data = json.loads(baseline.read_text("utf-8"))
                # Reconstruct an EvalReport from the saved dict for diffing.
                from code_rag.eval.harness import EvalResult
                base_report = EvalReport(
                    cases=[
                        EvalResult(
                            query=str(c.get("query", "")),
                            rank=c.get("rank"),
                            latency_ms=float(c.get("latency_ms", 0.0)),
                            top_paths=list(c.get("top_paths", [])),
                            tag=c.get("tag"),
                        )
                        for c in baseline_data.get("cases", [])
                    ],
                    label=baseline_data.get("label"),
                    index_meta=baseline_data.get("index_meta"),
                )
                diff = report.diff(base_report)
                click.echo("\ndiff vs baseline:")
                click.echo(json.dumps(diff, indent=2))
                if fail_on_regression is not None:
                    worst = min(diff["deltas_pp"].values()) if diff["deltas_pp"] else 0.0
                    if worst < -fail_on_regression:
                        click.echo(
                            f"\nFAIL: regression of {abs(worst):.2f} pp exceeds "
                            f"threshold of {fail_on_regression} pp",
                            err=True,
                        )
                        return 2

            if json_out:
                json_out.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
                click.echo(f"[report written to {json_out}]", err=True)
            return 0
        finally:
            vec.close()
            lex.close()
            await embedder.aclose()
            if isinstance(reranker, LMStudioReranker):
                await reranker.aclose()

    sys.exit(asyncio.run(_run()))


@main.command("reap")
@click.option("--kill", is_flag=True,
              help="Actually terminate orphans. Without this flag we DRY-RUN.")
@click.option("--quiet", is_flag=True,
              help="Suppress per-process output; print only the summary.")
@click.pass_context
def reap_cmd(ctx: click.Context, kill: bool, quiet: bool) -> None:
    """Phase 32: find (and optionally kill) orphaned code_rag processes.

    "Orphaned" = a code_rag MCP / dashboard / watcher process whose
    legitimate ancestor (claude.exe for MCP, Task Scheduler for the others)
    is no longer alive. These accumulate over time as Claude Code sessions
    crash or restart without sending a clean shutdown signal to their MCP
    subprocess.

    Without `--kill` this is a DRY-RUN: it lists what WOULD be reaped so
    you can sanity-check before pulling the trigger.

    Designed to be safe to run on a schedule (e.g. every 10 min). Won't
    touch live MCP servers attached to running Claude Code instances; only
    reaps subprocesses whose parent has died.
    """
    _ = ctx
    from code_rag.util.proc_hygiene import reap_orphans
    report = reap_orphans(kill=kill)
    n_alive = len(report["alive"])
    n_orphans = len(report["orphans"])
    n_killed = len(report["killed"])
    if not quiet:
        if n_orphans == 0:
            click.echo(f"clean: {n_alive} live code_rag process(es), 0 orphans")
        else:
            mode = "KILLED" if kill else "WOULD KILL"
            click.echo(f"{n_orphans} orphan(s) found ({mode}):")
            for p in report["orphans"]:
                click.echo(f"  pid={p['pid']:>6}  kind={p['kind'] or '?':<10s}  reason={p['reason']}")
                click.echo(f"          cmd: {p['cmdline_preview']}")
            click.echo(f"\nlive (kept): {n_alive}")
            for p in report["alive"]:
                click.echo(f"  pid={p['pid']:>6}  kind={p['kind'] or '?':<10s}")
    if kill:
        click.echo(f"\nkilled: {n_killed}/{n_orphans}")
    sys.exit(0 if (n_orphans == 0 or kill) else 1)


@main.command("eval-gate")
@click.option("--fixture", type=click.Path(exists=True, path_type=Path), default=None,
              help="Eval fixture to run. Default: src/code_rag/eval/fixtures/manual_eval.json")
@click.option("--baseline", type=click.Path(path_type=Path), default=None,
              help="Locked baseline JSON. Default: data/eval/baseline_locked.json")
@click.option("--max-regression-pp", default=1.0, show_default=True, type=float,
              help="Fail if any headline metric drops by more than this (in pp).")
@click.option("--write-baseline", is_flag=True,
              help="Write the current run as the new locked baseline. Use after "
                   "a verified improvement to update the gate.")
@click.option("--label", default=None,
              help="Optional label written into the saved report.")
@click.pass_context
def eval_gate(
    ctx: click.Context, fixture: Path | None, baseline: Path | None,
    max_regression_pp: float, write_baseline: bool, label: str | None,
) -> None:
    """Phase 26: regression gate over the canonical eval fixture.

    Pipeline:
      1. Run the standard eval (same searcher as MCP / production).
      2. Compare every headline metric vs the locked baseline.
      3. Exit non-zero if any metric regresses more than --max-regression-pp.

    Use this in a pre-commit hook, CI step, or before merging a phase. With
    `--write-baseline`, the current run becomes the new gate floor — only do
    that after you've verified the change is a real improvement.
    """
    from code_rag.eval.harness import EvalReport, load_cases, run_eval
    settings = ctx.obj["settings"]
    repo_root = Path(__file__).resolve().parents[2]
    fixture_path = fixture or (
        repo_root / "src" / "code_rag" / "eval" / "fixtures" / "manual_eval.json"
    )
    baseline_path = baseline or (repo_root / "data" / "eval" / "baseline_locked.json")

    async def _run() -> int:
        embedder = build_embedder(settings)
        vec = build_vector_store(settings)
        lex = build_lexical_store(settings)
        reranker = build_reranker(settings)
        if not settings.index_meta_path.exists():
            click.echo("FAIL  no index yet — run `code-rag index` first", err=True)
            return 1
        await embedder.health()
        meta = ChromaVectorStore.build_meta(
            embedder_kind=settings.embedder.kind,
            embedder_model=embedder.model,
            embedder_dim=embedder.dim,
        )
        vec.open(meta)
        lex.open()
        # Phase 30: wire the query rewriter so eval measures the SAME
        # pipeline production uses (rewriter on or off based on config).
        rewriter = build_query_rewriter(settings)
        try:
            searcher = HybridSearcher(
                embedder, vec, lex, reranker, rewriter=rewriter,
            )
            cases = load_cases(fixture_path)
            index_meta_dict = json.loads(settings.index_meta_path.read_text("utf-8"))
            report = await run_eval(
                cases, searcher,
                top_k=10, rerank_in=settings.reranker.top_k_in,
                label=label or "eval-gate",
                index_meta=index_meta_dict,
            )
            click.echo(json.dumps(report.summary(), indent=2))

            if write_baseline:
                baseline_path.parent.mkdir(parents=True, exist_ok=True)
                baseline_path.write_text(
                    json.dumps(report.as_dict(), indent=2), encoding="utf-8",
                )
                click.echo(f"\n[wrote new baseline -> {baseline_path}]", err=True)
                return 0

            if not baseline_path.exists():
                click.echo(
                    f"\nno baseline at {baseline_path} — run with --write-baseline "
                    f"once to lock the current run as the floor.",
                    err=True,
                )
                return 0

            baseline_data = json.loads(baseline_path.read_text("utf-8"))
            from code_rag.eval.harness import EvalResult
            base_report = EvalReport(
                cases=[
                    EvalResult(
                        query=str(c.get("query", "")),
                        rank=c.get("rank"),
                        latency_ms=float(c.get("latency_ms", 0.0)),
                        top_paths=list(c.get("top_paths", [])),
                        tag=c.get("tag"),
                    )
                    for c in baseline_data.get("cases", [])
                ],
                label=baseline_data.get("label"),
                index_meta=baseline_data.get("index_meta"),
            )
            diff = report.diff(base_report)
            click.echo("\ndiff vs baseline:")
            click.echo(json.dumps(diff, indent=2))
            worst = min(diff["deltas_pp"].values()) if diff["deltas_pp"] else 0.0
            if worst < -max_regression_pp:
                click.echo(
                    f"\nFAIL  {abs(worst):.2f}pp regression > {max_regression_pp}pp "
                    f"threshold. Investigate before merging.",
                    err=True,
                )
                return 2
            click.echo(
                f"\nPASS  worst delta {worst:+.2f}pp within {max_regression_pp}pp threshold.",
            )
            return 0
        finally:
            vec.close()
            lex.close()
            await embedder.aclose()
            if isinstance(reranker, LMStudioReranker):
                await reranker.aclose()
            # Phase 30: tear down rewriter cache + http client.
            if rewriter is not None:
                from code_rag.retrieval.query_rewriter import LMStudioQueryRewriter
                if isinstance(rewriter, LMStudioQueryRewriter):
                    await rewriter.aclose()
                    if rewriter._cache is not None:  # pyright: ignore[reportPrivateUsage]
                        rewriter._cache.close()  # pyright: ignore[reportPrivateUsage]

    sys.exit(asyncio.run(_run()))


@main.command("fsck")
@click.option("--fix", "auto_fix", is_flag=True,
              help="Attempt safe auto-repairs (drop missing dynamic roots, prune orphan file-hash rows).")
@click.option("--json-out", "json_out", type=click.Path(path_type=Path), default=None,
              help="Write the full report as JSON.")
@click.pass_context
def fsck_cmd(ctx: click.Context, auto_fix: bool, json_out: Path | None) -> None:
    """Phase 23: walk all stores looking for inconsistencies. Reports drift
    between vector/lexical/graph stores, dangling references, dead dynamic
    roots, and orphaned file-hash entries. With --fix, applies the safe
    auto-repairs."""
    from code_rag.ops import fsck
    settings = ctx.obj["settings"]
    vec = build_vector_store(settings)
    lex = build_lexical_store(settings)
    if not settings.index_meta_path.exists():
        click.echo("no index yet -- run `code-rag index` first", err=True)
        sys.exit(1)
    meta_text = settings.index_meta_path.read_text("utf-8")
    from code_rag.models import IndexMeta
    meta = IndexMeta.model_validate_json(meta_text)
    vec.open(meta)
    lex.open()
    try:
        report = fsck(settings, vec, lex, auto_fix=auto_fix)
        click.echo(json.dumps(report.summary(), indent=2))
        if json_out:
            json_out.write_text(json.dumps(report.summary(), indent=2), encoding="utf-8")
        if not report.ok:
            sys.exit(2)
    finally:
        vec.close()
        lex.close()


@main.command("metrics")
@click.option("--out", "out", type=click.Path(path_type=Path), default=None,
              help="Write OpenMetrics text to this path. Default: stdout.")
@click.pass_context
def metrics_cmd(ctx: click.Context, out: Path | None) -> None:
    """Phase 23: emit an OpenMetrics-format snapshot of index health.

    Pipe to a file periodically and point Prometheus or local Grafana at
    it. Free, no external services required.
    """
    from code_rag.ops import metrics_text
    settings = ctx.obj["settings"]
    vec = build_vector_store(settings)
    lex = build_lexical_store(settings)
    if not settings.index_meta_path.exists():
        click.echo("# code-rag: no index yet (run `code-rag index`)", err=True)
        sys.exit(1)
    meta_text_str = settings.index_meta_path.read_text("utf-8")
    from code_rag.models import IndexMeta
    meta = IndexMeta.model_validate_json(meta_text_str)
    vec.open(meta)
    lex.open()
    try:
        text = metrics_text(settings, vec, lex)
        if out:
            out.write_text(text, encoding="utf-8")
        else:
            click.echo(text)
    finally:
        vec.close()
        lex.close()


@main.group()
def embedder() -> None:
    """Manage the embedder model — list presets, A/B switch, list current."""


@embedder.command("list")
def embedder_list() -> None:
    """List the curated code-specialized embedder presets."""
    from code_rag.embedders.code_specialized import list_presets
    for p in list_presets():
        click.echo(f"  {p.name:25s} dim={p.expected_dim or '?':>5}  hf={p.huggingface_repo}")
        click.echo(f"  {'':25s}   {p.notes}")


@embedder.command("switch")
@click.argument("preset_name")
@click.option("--no-download", is_flag=True,
              help="Skip auto `lms get` even if the model is missing.")
@click.option("--yes", is_flag=True,
              help="Skip the confirmation prompt about the data wipe.")
@click.pass_context
def embedder_switch(
    ctx: click.Context, preset_name: str, no_download: bool, yes: bool,
) -> None:
    """Phase 17: A/B switch the embedder model with one command.

    Steps (all idempotent, all reversible):
      1. Validate the preset name.
      2. `lms get <hf-repo>` if not already downloaded.
      3. Edit config.toml: set `[embedder].preset = "<name>"` and clear
         the `model` override so the factory routes via preset.
      4. WIPE data/{chroma,fts.db,graph.kz,index_meta.json,file_hashes.db}
         (the new model has different vector dim — incompatible with old
         vectors; IndexMeta would refuse to open anyway).
      5. Run a fresh `code-rag index`.
    """
    from code_rag.embedders.code_specialized import CODE_EMBEDDER_PRESETS
    if preset_name not in CODE_EMBEDDER_PRESETS:
        click.echo(f"unknown preset: {preset_name}", err=True)
        click.echo(f"available: {sorted(CODE_EMBEDDER_PRESETS)}", err=True)
        sys.exit(2)
    preset = CODE_EMBEDDER_PRESETS[preset_name]
    settings = ctx.obj["settings"]

    if not yes:
        click.echo(f"About to switch to {preset.name} ({preset.lms_id}, dim={preset.expected_dim}).")
        click.echo("This WIPES the existing index (model dim differs) and re-embeds from scratch.")
        click.echo("Current chunks: see `code-rag index-stats`.  Estimated reindex time: 30-60 min.")
        if not click.confirm("Proceed?"):
            click.echo("aborted.")
            return

    # 1. lms get if missing.
    if not no_download:
        from code_rag.lms_ctl import find_lms, model_is_loaded
        loc = find_lms()
        if loc.path is None:
            click.echo("FAIL: lms CLI not found. Install LM Studio + run `lms bootstrap`, then retry.", err=True)
            sys.exit(3)
        if not model_is_loaded(settings.embedder.base_url, preset.lms_id):
            click.echo(f"Downloading {preset.huggingface_repo} via lms get ...")
            r = subprocess.run([str(loc.path), "get", preset.huggingface_repo],
                               capture_output=False, check=False)
            if r.returncode != 0:
                click.echo("FAIL: lms get exited non-zero. Resolve manually then re-run with --no-download.", err=True)
                sys.exit(4)
            click.echo(f"Loading {preset.lms_id} ...")
            subprocess.run([str(loc.path), "load", preset.lms_id], capture_output=False, check=False)

    # 2. Edit config.toml.
    cfg_path = _resolved_config_path(ctx)
    cfg_text = cfg_path.read_text(encoding="utf-8")
    import re as _re
    if "preset" in cfg_text and _re.search(r"^\s*preset\s*=", cfg_text, _re.MULTILINE):
        cfg_text = _re.sub(r"^\s*preset\s*=.*$", f'preset = "{preset.name}"',
                           cfg_text, count=1, flags=_re.MULTILINE)
    else:
        # Insert under [embedder] header.
        cfg_text = _re.sub(
            r"(\[embedder\][^\[]*?)(\n\s*\[)",
            rf'\1preset = "{preset.name}"\n\2',
            cfg_text, count=1, flags=_re.DOTALL,
        )
    cfg_path.write_text(cfg_text, encoding="utf-8")
    click.echo(f"config.toml updated: [embedder].preset = \"{preset.name}\"")

    # 3. Wipe stores (model dim changes -> existing vectors incompatible).
    import shutil as _sh
    for sub in ("chroma", "graph"):
        p = settings.paths.data_dir / sub
        if p.exists():
            _sh.rmtree(p, ignore_errors=True)
    for f in ("fts.db", "fts.db-wal", "fts.db-shm",
              "graph.kz", "graph.kz.wal", "graph.kz.shadow",
              "index_meta.json", "file_hashes.db"):
        fp = settings.paths.data_dir / f
        if fp.exists():
            with contextlib.suppress(OSError):
                fp.unlink()
    click.echo("data dir wiped (model dim incompatibility).")

    click.echo("\nNext: run `code-rag index` to populate the new index.")
    click.echo("After it finishes, re-run `code-rag eval src/code_rag/eval/fixtures/manual_eval.json`")
    click.echo("with --baseline pointing at your previous baseline to measure the lift.")


@main.command("git-log-index")
@click.option("--root", "root", type=click.Path(exists=True, path_type=Path), default=None,
              help="Index only this single root. Default: every configured root that's a git repo.")
@click.option("--max-commits", default=2000, show_default=True, type=int)
@click.option("--max-chars", default=2400, show_default=True, type=int,
              help="Per-chunk char cap; longer diffs get truncated with a `git show` pointer.")
@click.pass_context
def git_log_index(ctx: click.Context, root: Path | None, max_commits: int, max_chars: int) -> None:
    """Phase 22: walk `git log -p` and index every (commit x file) diff hunk.

    Run on demand. Once-per-machine bulk job; the live watcher does NOT call
    this on file events. Re-run after major histories change (e.g. force-push,
    monorepo merge) -- re-runs are idempotent because chunk_ids are content-
    addressed (sha + path + diff body).
    """
    from code_rag.indexing.git_log import index_git_log
    settings = ctx.obj["settings"]

    async def _run() -> int:
        embedder = build_embedder(settings)
        vec = build_vector_store(settings)
        lex = build_lexical_store(settings)
        try:
            await embedder.health()
        except Exception as e:
            click.echo(f"FAIL  embedder: {e}", err=True)
            return 1
        meta = ChromaVectorStore.build_meta(
            embedder_kind=settings.embedder.kind,
            embedder_model=embedder.model,
            embedder_dim=embedder.dim,
        )
        vec.open(meta)
        lex.open()
        try:
            roots = [root] if root else settings.all_roots()
            total_chunks = 0
            for r in roots:
                rr = r.resolve()
                chunks = index_git_log(
                    rr, repo_label=rr.name,
                    max_commits=max_commits, max_chars_per_chunk=max_chars,
                )
                if not chunks:
                    click.echo(f"  [skip] {rr.name}: no git history (or git missing)")
                    continue
                # Embed in batches and upsert.
                texts = [c.text for c in chunks]
                vectors = await embedder.embed(texts)
                vec.upsert(chunks, vectors)
                lex.upsert(chunks)
                total_chunks += len(chunks)
                click.echo(f"  [ok]  {rr.name}: {len(chunks)} diff chunks")
            click.echo(f"\nTOTAL  {total_chunks} chunks across {len(roots)} root(s)")
            return 0
        finally:
            vec.close()
            lex.close()
            await embedder.aclose()

    sys.exit(asyncio.run(_run()))


@main.command("eval-mine")
@click.option("--output", "output", type=click.Path(path_type=Path), required=True,
              help="Where to write the JSON eval fixture.")
@click.option("--max-pairs", default=500, show_default=True, type=int)
@click.option("--transcripts-dir", type=click.Path(path_type=Path),
              default=None,
              help="Override transcripts root (default: ~/.claude/projects).")
@click.option("--include-source/--no-include-source", default=False,
              help="Keep _source debug fields on each pair.")
@click.option("--filter-to-index/--no-filter-to-index", default=True,
              show_default=True,
              help="Phase 26: drop cases whose expected paths aren't in the "
                   "current vector store. Gives a true recall ceiling.")
@click.pass_context
def eval_mine(
    ctx: click.Context, output: Path, max_pairs: int,
    transcripts_dir: Path | None, include_source: bool,
    filter_to_index: bool,
) -> None:
    """Mine (query, expected_path) eval pairs from your Claude Code transcripts.

    Walks every JSONL under ~/.claude/projects/, finds user questions
    followed by Claude's first Read/Edit on a file, and emits a fixture
    compatible with `code-rag eval`. Free, real, in-distribution ground truth.

    Paths are normalized against your configured `[paths].roots`. With
    `--filter-to-index` (default ON), the fixture is further filtered to only
    include cases whose ground-truth paths exist in the current Chroma
    collection — without this, mined cases referencing deleted files or
    worktree clones depress recall artificially.
    """
    from code_rag.eval.harness import EvalCase, filter_cases_to_paths
    from code_rag.eval.mine_transcripts import mine_all
    settings = ctx.obj["settings"]
    td = transcripts_dir or (Path.home() / ".claude" / "projects")
    if not td.exists():
        click.echo(f"transcripts dir not found: {td}", err=True)
        sys.exit(2)

    pairs = mine_all(td, settings.all_roots(), max_pairs=max_pairs)
    cases_dicts = [p.to_case() for p in pairs]
    if not include_source:
        for c in cases_dicts:
            c.pop("_source", None)
    n_mined = len(cases_dicts)

    if filter_to_index:
        if not settings.index_meta_path.exists():
            click.echo("WARN  no index yet — skipping --filter-to-index", err=True)
        else:
            from code_rag.models import IndexMeta
            meta = IndexMeta.model_validate_json(
                settings.index_meta_path.read_text("utf-8"),
            )
            vec = build_vector_store(settings)
            try:
                vec.open(meta)
                # list_paths is concrete on ChromaVectorStore; skip silently
                # if a future backend doesn't implement it.
                indexed = vec.list_paths() if hasattr(vec, "list_paths") else None
            finally:
                with contextlib.suppress(Exception):
                    vec.close()
            if indexed is not None:
                cases = [EvalCase.from_dict(c) for c in cases_dicts]
                filtered = filter_cases_to_paths(cases, indexed)
                kept_queries = {c.query for c in filtered}
                cases_dicts = [c for c in cases_dicts if c["query"] in kept_queries]
                # Also prune expected[] entries that aren't in the index, mirroring
                # the EvalCase-level pruning so the on-disk fixture matches.
                for c in cases_dicts:
                    c["expected"] = [
                        e for e in c["expected"] if e.get("path") in indexed
                    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cases_dicts, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    click.echo(f"mined {n_mined} pairs from {td}", err=True)
    if filter_to_index and n_mined != len(cases_dicts):
        click.echo(f"filtered to {len(cases_dicts)} in-corpus pairs "
                   f"({n_mined - len(cases_dicts)} dropped)", err=True)
    click.echo(f"wrote -> {output}", err=True)


@main.command()
@click.option("-n", "--n", "n", default=30, show_default=True, type=int,
              help="Number of repetitions per query.")
@click.option("-q", "--queries", type=click.Path(exists=True, path_type=Path),
              default=None,
              help="JSON list of query strings. Defaults to a small built-in set.")
@click.pass_context
def bench(ctx: click.Context, n: int, queries: Path | None) -> None:
    """Measure p50 / p95 / p99 latency of the full search pipeline.

    Runs each query N times against the real index + LM Studio embedder and
    reports per-query and aggregate latency. Useful for tuning rerank top-k.
    """
    settings = ctx.obj["settings"]
    import statistics
    import time

    async def _run() -> int:
        embedder = build_embedder(settings)
        vec = build_vector_store(settings)
        lex = build_lexical_store(settings)
        reranker = build_reranker(settings)
        if not settings.index_meta_path.exists():
            click.echo("no index yet - run `code-rag index` first", err=True)
            return 1
        await embedder.health()
        meta = ChromaVectorStore.build_meta(
            embedder_kind=settings.embedder.kind,
            embedder_model=embedder.model,
            embedder_dim=embedder.dim,
        )
        vec.open(meta)
        lex.open()
        try:
            searcher = HybridSearcher(embedder, vec, lex, reranker)
            if queries is not None:
                import json as _json
                q_list = _json.loads(queries.read_text("utf-8"))
                if not isinstance(q_list, list) or not all(isinstance(x, str) for x in q_list):
                    click.echo("queries file must be a JSON array of strings", err=True)
                    return 1
            else:
                q_list = [
                    "OnBarUpdate",
                    "place market order",
                    "position sizer",
                    "trailing stop logic",
                    "ATR indicator",
                ]

            params = SearchParams(
                k_final=8,
                k_vector=settings.reranker.top_k_in,
                k_lexical=settings.reranker.top_k_in,
                k_rerank_in=settings.reranker.top_k_in,
            )

            # Warm-up (populate caches): one run of each query.
            for q in q_list:
                await searcher.search(q, params)

            all_latencies_ms: list[float] = []
            per_query: dict[str, list[float]] = {}
            for q in q_list:
                lats: list[float] = []
                for _ in range(n):
                    t0 = time.monotonic()
                    await searcher.search(q, params)
                    lats.append((time.monotonic() - t0) * 1000)
                per_query[q] = lats
                all_latencies_ms.extend(lats)

            def pct(xs: list[float], p: float) -> float:
                if not xs:
                    return 0.0
                xs = sorted(xs)
                idx = min(len(xs) - 1, max(0, round(p * (len(xs) - 1))))
                return xs[idx]

            report = {
                "queries": len(q_list),
                "repetitions": n,
                "aggregate": {
                    "p50_ms": round(statistics.median(all_latencies_ms), 1),
                    "p95_ms": round(pct(all_latencies_ms, 0.95), 1),
                    "p99_ms": round(pct(all_latencies_ms, 0.99), 1),
                    "mean_ms": round(statistics.mean(all_latencies_ms), 1),
                },
                "per_query": {
                    q: {
                        "p50_ms": round(statistics.median(lats), 1),
                        "p95_ms": round(pct(lats, 0.95), 1),
                        "mean_ms": round(statistics.mean(lats), 1),
                    }
                    for q, lats in per_query.items()
                },
            }
            click.echo(json.dumps(report, indent=2))
            return 0
        finally:
            vec.close()
            lex.close()
            await embedder.aclose()
            if isinstance(reranker, LMStudioReranker):
                await reranker.aclose()

    sys.exit(asyncio.run(_run()))


@main.command("index-stats")
@click.pass_context
def index_stats(ctx: click.Context) -> None:
    """Show counts and index metadata (no network calls)."""
    settings = ctx.obj["settings"]
    if not settings.index_meta_path.exists():
        click.echo("no index yet -- run `code-rag index`", err=True)
        sys.exit(1)
    meta_text = settings.index_meta_path.read_text("utf-8")
    click.echo(meta_text)
    # Open the collection read-only just to count. The collection name is
    # namespaced by embedder model+dim, so we compute the same hash the store
    # uses when opening -- see ChromaVectorStore._resolved_name.
    import chromadb

    from code_rag.models import IndexMeta
    from code_rag.stores.chroma_vector import ChromaVectorStore
    if settings.chroma_dir.exists():
        client = chromadb.PersistentClient(path=str(settings.chroma_dir))
        meta_obj = IndexMeta.model_validate_json(meta_text)
        coll_name = ChromaVectorStore._resolved_name(
            settings.vector_store.collection, meta_obj,
        )
        try:
            coll = client.get_collection(coll_name)
            click.echo(f"chunks: {coll.count()}  (collection: {coll_name})")
        except Exception as e:
            click.echo(f"chunks: (unavailable: {e}; collection={coll_name!r})")


@main.group()
def roots() -> None:
    """Manage indexing roots -- both the curated `config.toml` set and the
    dynamically auto-added set (via MCP `ensure_workspace_indexed`).
    """


@roots.command("list")
@click.pass_context
def roots_list(ctx: click.Context) -> None:
    """List all active roots (config + dynamic), tagged by origin."""
    from code_rag.dynamic_roots import DynamicRoots
    settings = ctx.obj["settings"]
    click.echo("# config.toml roots")
    for r in settings.paths.roots:
        click.echo(f"  [config]   {r.as_posix()}")
    dyn = DynamicRoots.load(settings.dynamic_roots_path)
    if not dyn.entries:
        click.echo("# dynamic roots: (none)")
        return
    click.echo("# dynamic roots")
    for e in dyn.entries:
        exists = "✓" if e.path.exists() else "✗"
        click.echo(f"  [dynamic]  {e.path.as_posix()}  "
                   f"{exists}  added={e.added_at}  last_used={e.last_used_at}")


@roots.command("remove")
@click.argument("path", type=click.Path(path_type=Path))
@click.pass_context
def roots_remove(ctx: click.Context, path: Path) -> None:
    """Remove a dynamic root. Config roots must be edited in config.toml."""
    from code_rag.dynamic_roots import DynamicRoots
    settings = ctx.obj["settings"]
    dyn = DynamicRoots.load(settings.dynamic_roots_path)
    if dyn.remove(path):
        click.echo(f"removed dynamic root: {path.resolve().as_posix()}")
    else:
        click.echo(f"not in dynamic roots: {path}", err=True)
        sys.exit(1)


@roots.command("prune")
@click.option("--days", type=int, default=30, show_default=True,
              help="Prune dynamic roots not used in this many days.")
@click.pass_context
def roots_prune(ctx: click.Context, days: int) -> None:
    """Prune stale dynamic roots (by last_used_at timestamp)."""
    from code_rag.dynamic_roots import DynamicRoots
    settings = ctx.obj["settings"]
    dyn = DynamicRoots.load(settings.dynamic_roots_path)
    pruned = dyn.prune_stale(days)
    if not pruned:
        click.echo(f"no dynamic roots older than {days} days")
        return
    for p in pruned:
        click.echo(f"pruned: {p.as_posix()}")
    click.echo(f"total pruned: {len(pruned)}")

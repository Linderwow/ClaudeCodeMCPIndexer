from __future__ import annotations

import asyncio
import contextlib
import json
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
            if isinstance(embedder, LMStudioEmbedder):
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
            if isinstance(embedder, LMStudioEmbedder):
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
            if settings.chroma_dir.exists():
                import chromadb
                try:
                    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
                    coll = client.get_collection(settings.vector_store.collection)
                    ok(f"vector chunks: {coll.count()}")
                except Exception as e:
                    fail("chroma.count", str(e))
            if settings.fts_path.exists():
                lex = build_lexical_store(settings)
                try:
                    lex.open()
                    ok(f"lexical chunks: {lex.count()}")
                finally:
                    lex.close()
        else:
            fail("index", "no index yet -- run `code-rag index`")

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
    """
    settings = ctx.obj["settings"]

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
            if isinstance(embedder, LMStudioEmbedder):
                await embedder.aclose()

    try:
        sys.exit(asyncio.run(_run()))
    except KeyboardInterrupt:
        click.echo("watcher stopped", err=True)
        sys.exit(0)


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
            searcher = HybridSearcher(embedder, vec, lex, reranker)
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
            if isinstance(embedder, LMStudioEmbedder):
                await embedder.aclose()
            if isinstance(reranker, LMStudioReranker):
                await reranker.aclose()

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
            if isinstance(embedder, LMStudioEmbedder):
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
@click.pass_context
def eval_mine(
    ctx: click.Context, output: Path, max_pairs: int,
    transcripts_dir: Path | None, include_source: bool,
) -> None:
    """Mine (query, expected_path) eval pairs from your Claude Code transcripts.

    Walks every JSONL under ~/.claude/projects/, finds user questions
    followed by Claude's first Read/Edit on a file, and emits a fixture
    compatible with `code-rag eval`. Free, real, in-distribution ground truth.

    Paths are normalized against your configured `[paths].roots`, so the
    fixture matches what's actually in your index.
    """
    from code_rag.eval.mine_transcripts import mine_all
    settings = ctx.obj["settings"]
    td = transcripts_dir or (Path.home() / ".claude" / "projects")
    if not td.exists():
        click.echo(f"transcripts dir not found: {td}", err=True)
        sys.exit(2)

    pairs = mine_all(td, settings.all_roots(), max_pairs=max_pairs)
    cases = [p.to_case() for p in pairs]
    if not include_source:
        for c in cases:
            c.pop("_source", None)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    click.echo(f"mined {len(cases)} pairs from {td}", err=True)
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
            if isinstance(embedder, LMStudioEmbedder):
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

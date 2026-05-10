"""Phase 60-E: profile per-stage cost of the indexer pipeline.

Builds a *separate* index (under --data-dir) so the live one isn't disturbed,
runs reindex_path against --target, then prints the StageTimes summary +
per-stage % of wall clock + derived per-symbol/per-edge graph latency.

Usage:
    .venv/Scripts/python.exe scripts/profile_indexer.py \\
        --target C:/Users/Alex/RiderProjects/ai-lab \\
        --data-dir C:/Users/Alex/Documents/code-rag-mcp/data_profile

Pre-loaded scenarios:
    --preload-graph PATH   copy an existing graph.kz into --data-dir before
                            profiling, so we measure write cost on a populated
                            DB (matches production behaviour where graph
                            grows to 200+ MB during a full reindex).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from code_rag.config import Settings  # noqa: E402
from code_rag.factory import (  # noqa: E402
    build_embedder,
    build_graph_store,
    build_lexical_store,
    build_vector_store,
)
from code_rag.graph.ingest import GraphIngester  # noqa: E402
from code_rag.indexing.indexer import Indexer  # noqa: E402
from code_rag.stores.chroma_vector import ChromaVectorStore  # noqa: E402


def _load_settings(data_dir: Path, target: Path) -> Settings:
    """Load real config.toml but redirect data_dir + roots so we don't
    touch the live index."""
    from code_rag.config import load_settings
    s = load_settings(REPO / "config.toml")
    # Override data_dir to keep the profile run isolated.
    s.paths.data_dir = data_dir
    # Ignore the user's pinned roots — profile target only.
    s.paths.roots = [target]
    return s


async def _run(args: argparse.Namespace) -> int:
    target  = Path(args.target).resolve()
    data    = Path(args.data_dir).resolve()
    data.mkdir(parents=True, exist_ok=True)

    if args.preload_graph:
        src = Path(args.preload_graph).resolve()
        dst = data / "graph.kz"
        if not src.exists():
            print(f"FAIL  preload-graph not found: {src}", file=sys.stderr)
            return 1
        print(f"copy preload graph: {src}  ->  {dst}  "
              f"({src.stat().st_size/1024/1024:.1f} MB)")
        shutil.copy2(src, dst)

    settings = _load_settings(data, target)

    embedder = build_embedder(settings)
    vec      = build_vector_store(settings)
    lex      = build_lexical_store(settings)
    graph    = build_graph_store(settings)

    try:
        await embedder.health()
    except Exception as e:
        print(f"FAIL  embedder health: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    meta = ChromaVectorStore.build_meta(
        embedder_kind=settings.embedder.kind,
        embedder_model=embedder.model,
        embedder_dim=embedder.dim,
    )
    vec.open(meta)
    lex.open()
    graph.open()
    try:
        # NOTE: file_hashes=None on purpose so every file is reprocessed
        # (matches a fresh-reindex scenario rather than a no-op rerun).
        indexer = Indexer(
            settings, embedder, vec,
            lexical_store=lex,
            graph_store=GraphIngester(graph),
            file_hashes=None,
        )
        t0 = time.monotonic()
        stats = await indexer.reindex_path(target)
        wall = time.monotonic() - t0
    finally:
        vec.close()
        lex.close()
        graph.close()
        await embedder.aclose()

    summary = stats.as_dict()
    summary["wall_clock_s"] = round(wall, 3)

    print("\n=== INDEX STATS ===")
    print(json.dumps(summary, indent=2))

    st = stats.stage_times
    print("\n=== STAGE BREAKDOWN (% of wall clock, single-worker-equivalent sum) ===")
    rows = [
        ("read",          st.read_s),
        ("chunk",         st.chunk_s),
        ("embed",         st.embed_s),
        ("graph_extract", st.graph_extract_s),
        ("lock_wait",     st.lock_wait_s),
        ("vec_delete",    st.vec_delete_s),
        ("lex_delete",    st.lex_delete_s),
        ("graph_delete",  st.graph_delete_s),
        ("vec_upsert",    st.vec_upsert_s),
        ("lex_upsert",    st.lex_upsert_s),
        ("graph_commit",  st.graph_commit_s),
        ("hash",          st.hash_s),
    ]
    rows.sort(key=lambda r: -r[1])
    for label, val in rows:
        bar = "#" * int(val / max(rows[0][1], 0.001) * 40) if rows[0][1] else ""
        pct_wall = val / wall * 100 if wall else 0
        print(f"  {label:14s}  {val:8.3f} s   {pct_wall:5.1f}% wall   {bar}")

    print("\n=== DERIVED ===")
    n = stats.files_indexed or 1
    print(f"  files_indexed: {stats.files_indexed}")
    print(f"  chunks/file:   {stats.chunks_emitted/n:.1f}")
    print(f"  symbols/file:  {st.symbols_total/n:.1f}")
    print(f"  edges/file:    {st.edges_total/n:.1f}")
    print(f"  wall/file:     {wall/n*1000:.1f} ms")
    print(f"  files/sec:     {n/wall:.2f}")
    print(f"  chunks/sec:    {stats.chunks_emitted/wall:.2f}")
    if st.symbols_total:
        print(f"  graph_commit/symbol: {st.graph_commit_s/st.symbols_total*1000:.2f} ms")
    if st.edges_total:
        print(f"  graph_commit/edge:   {st.graph_commit_s/st.edges_total*1000:.2f} ms")

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target",   required=True, help="path or subtree to reindex")
    p.add_argument("--data-dir", required=True, help="ISOLATED data dir for the profile")
    p.add_argument("--preload-graph", default=None,
                   help="copy this graph.kz into --data-dir before run "
                        "(simulates a populated DB)")
    args = p.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())

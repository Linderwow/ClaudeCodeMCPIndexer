"""MCP server exposing the code-rag tools over stdio.

Registered tools (must match `_HANDLERS` and `TOOLS` below):
  * search_code              — hybrid search (vector + lexical + rerank)
  * get_chunk_text           — fetch full text of a truncated chunk by id
  * get_symbol               — locate a symbol by name
  * get_callers              — graph walk: who calls X
  * get_callees              — graph walk: what does X call
  * get_file_context         — file's symbols + immediate graph neighborhood
  * index_stats              — counts + metadata + last-updated
  * ensure_workspace_indexed — auto-register a new repo & kick off indexing

The MCP server itself is READ-ONLY on the three stores. The one exception
is `ensure_workspace_indexed`, which:
  (a) appends to `data/dynamic_roots.json` (its own file — not a store), and
  (b) spawns a DETACHED `code-rag index --path <p>` subprocess that opens
      the stores in its own process. The live watcher remains the sole
      in-process writer.

All tools return structured JSON; the client (Claude Code) sees concise text
blocks and can pass them straight into a prompt.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from code_rag.config import Settings, load_settings
from code_rag.factory import (
    build_embedder,
    build_graph_store,
    build_lexical_store,
    build_reranker,
    build_vector_store,
)
from code_rag.interfaces.embedder import Embedder
from code_rag.interfaces.graph_store import GraphStore
from code_rag.interfaces.lexical_store import LexicalStore
from code_rag.interfaces.reranker import Reranker
from code_rag.interfaces.vector_store import VectorStore
from code_rag.logging import get
from code_rag.rerankers.lm_studio import LMStudioReranker
from code_rag.retrieval.search import HybridSearcher, SearchParams
from code_rag.stores.chroma_vector import ChromaVectorStore

log = get(__name__)


# ---- Tool schemas ---------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="search_code",
        description=(
            "PREFER THIS OVER `grep` / `rg` for ANY code-location question. "
            "Hybrid retrieval (semantic embedding + BM25 lexical + cross-encoder "
            "rerank) across the full indexed corpus — code AND docs. Returns "
            "ranked structure-aware chunks with path, line range, symbol, and "
            "a `match_reason` explaining why each hit was selected. Use free-form "
            "natural-language queries ('where does the strategy size positions?') "
            "as readily as identifiers ('OnBarUpdate'). Only fall back to grep "
            "if `no_confident_match: true` comes back.\n\n"
            "FIRST-TIME-IN-A-REPO: if you're working in a repository whose "
            "root you don't recognize from prior searches, call "
            "`ensure_workspace_indexed` with the repo's absolute path ONCE at "
            "the start of the session. That auto-registers the repo for "
            "ongoing watching and kicks off a background index. Subsequent "
            "`search_code` calls will see results as chunks land."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query":     {"type": "string", "description": "Natural-language intent or exact identifier."},
                "k":         {"type": "integer", "default": 8, "minimum": 1, "maximum": 50},
                "path_glob": {"type": "string", "description": "fnmatch pattern to filter path (e.g. '**/*.cs')."},
                "lang":      {"type": "string", "enum": ["python", "c_sharp", "typescript", "tsx", "javascript"]},
                "min_score": {
                    "type": "number", "default": 0.0,
                    "description": "Confidence threshold. Hits below this are dropped; if all are dropped, response sets no_confident_match=true. Tuned range: 0.2-0.4 for rerank-backed searches.",
                },
                "max_chars_per_hit": {
                    "type": "integer", "default": 1200,
                    "description": "Smart-truncate each chunk to this many chars (preserves signature + head). 0 = no truncation. Use `get_chunk_text` to fetch the full body if truncated.",
                },
                "attach_neighbors": {
                    "type": "boolean", "default": False,
                    "description": "Also return 1-hop graph neighborhood (callers + callees) for each hit. Off by default to keep the call fast; enable when planning an edit.",
                },
                "recent_files_only_days": {
                    "type": "integer", "minimum": 0,
                    "description": "Phase 37-K. Drop hits whose source file's mtime is older than N days. Useful for 'what did we change recently around X?' queries — keeps year-old code from drowning out actively-edited files. 7-30 is the typical range.",
                },
                "prefer_root": {
                    "type": "string",
                    "description": "Phase 37-K. Boost hits whose path is under this root (case-insensitive prefix match). Useful when working in repo A and you want results from A to outrank look-alikes in repo B. Boost factor is +25%% per hit; non-matching hits untouched. The boost applies after rerank so match_reason shows it.",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="get_chunk_text",
        description=(
            "Fetch the FULL text of a chunk by id. Use after `search_code` with "
            "max_chars_per_hit returned a truncated hit you want to read in full."
        ),
        inputSchema={
            "type": "object",
            "properties": {"chunk_id": {"type": "string"}},
            "required": ["chunk_id"],
        },
    ),
    Tool(
        name="get_symbol",
        description=(
            "Locate a symbol by exact name in the graph index. Returns file path, "
            "line range, and kind. FASTER than `search_code` when you already know "
            "the name."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "path": {"type": "string", "description": "Optional: limit to this file."},
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="get_callers",
        description=(
            "List every symbol that CALLS the given one (1-hop reverse call edge). "
            "PREFER OVER grepping for the symbol name — this skips comments, strings, "
            "and same-name unrelated symbols, and resolves cross-file calls."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "path":   {"type": "string", "description": "Optional: limit target to this file."},
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="get_callees",
        description=(
            "List every symbol that the given symbol CALLS (1-hop forward call edge). "
            "Use to map what a function depends on before editing it."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "path":   {"type": "string"},
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="get_file_context",
        description=(
            "Return a file's symbols plus their 1-hop graph neighborhood "
            "(callers + callees). USE FIRST when starting to edit a file you "
            "haven't touched recently — it tells you the blast radius."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_neighbors": {
                    "type": "integer", "default": 25, "minimum": 1, "maximum": 200,
                    "description": "Cap callers/callees returned per symbol. Defaults to 25 so hot symbols (e.g. OnBarUpdate) don't blow the response.",
                },
            },
            "required": ["path"],
        },
    ),
    Tool(
        name="index_stats",
        description="Return index metadata, chunk counts, and last-updated timestamp.",
        inputSchema={"type": "object", "properties": {}},
    ),
    Tool(
        name="search_code_anchored",
        description=(
            "Phase 21: graph-augmented retrieval. Like search_code, but results "
            "are RE-RANKED by graph proximity to an anchor symbol you provide. "
            "Use this when the user's question has TWO parts: a semantic part "
            "(\"functions that touch session phase\") AND a structural part "
            "(\"...called by OnBarUpdate\"). Set `anchor_symbol` to the "
            "structural anchor and `hops` to the radius of related code you "
            "want boosted (1 = direct callers/callees only). Falls back to "
            "plain hybrid search if the anchor doesn't resolve."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query":         {"type": "string"},
                "anchor_symbol": {"type": "string"},
                "anchor_path":   {"type": "string", "description": "Optional: pin anchor to this file."},
                "hops":          {"type": "integer", "default": 1, "minimum": 1, "maximum": 4},
                "k":             {"type": "integer", "default": 8, "minimum": 1, "maximum": 50},
                "path_glob":     {"type": "string"},
                "lang":          {"type": "string", "enum": ["python", "c_sharp", "typescript", "tsx", "javascript"]},
            },
            "required": ["query", "anchor_symbol"],
        },
    ),
    Tool(
        name="ensure_workspace_indexed",
        description=(
            "Auto-register a repository for indexing. CALL THIS ONCE PER "
            "SESSION the first time you need to search a repo that isn't "
            "already indexed — subsequent calls for the same path are no-ops. "
            "Pass the repo's ABSOLUTE root directory (the one containing the "
            "`.git` folder or top-level project files). The tool:\n"
            "  1. Checks if the path is already under an indexed root — returns immediately if so.\n"
            "  2. Registers the path in the dynamic-roots file (survives reboots).\n"
            "  3. Kicks off a background `code-rag index --path <path>` subprocess.\n"
            "  4. Returns a status object with `already_indexed`, `registered`, and `background_pid`.\n"
            "The live watcher picks up the new root on its next restart; until then the "
            "background subprocess owns the initial ingest. `search_code` continues to "
            "work against whatever is already indexed while this runs. Safe to call "
            "repeatedly — idempotent."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the repo root to auto-index.",
                },
            },
            "required": ["path"],
        },
    ),
]


# ---- Resource holder ------------------------------------------------------


class ServerResources:
    """All long-lived handles the server needs across tool calls.

    Opened once at startup, closed once at shutdown. Tool handlers assume
    these are live — if a handle is missing it's a programmer error.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedder: Embedder | None = None
        self.vec: VectorStore | None = None
        self.lex: LexicalStore | None = None
        self.graph: GraphStore | None = None
        self.reranker: Reranker | None = None
        # Tests still wire an Indexer onto the resources to populate the corpus
        # before driving the handlers. In production the MCP server is purely
        # a reader (see .open()).
        self.indexer: Any = None
        self.searcher: HybridSearcher | None = None

    async def open(self) -> None:
        s = self.settings
        self.embedder = build_embedder(s)
        self.vec = build_vector_store(s)
        self.lex = build_lexical_store(s)
        # The MCP server is always a reader — the autostart watcher is the
        # single writer. Open Kuzu read-only so both can coexist.
        self.graph = build_graph_store(s, read_only=True)
        self.reranker = build_reranker(s)

        # Phase 36-G: retry embedder.health() with backoff. A single transient
        # LM Studio 500 (which happens during model swaps, GPU pressure spikes,
        # or post-crash auto-restart) would otherwise abort the entire MCP
        # boot, surface to Claude Code as "Server disconnected", and require
        # the user to manually /mcp reconnect. Three attempts with 2-4-8s
        # backoff covers the 30s window LM Studio needs to recover from
        # most transient hiccups, while still failing fast if LM Studio is
        # genuinely down.
        last_err: Exception | None = None
        for attempt, delay in enumerate((0, 2, 4, 8), start=1):
            if delay > 0:
                await asyncio.sleep(delay)
            try:
                await self.embedder.health()
                last_err = None
                break
            except Exception as e:
                last_err = e
                log.warning(
                    "mcp.embedder_health_retry",
                    attempt=attempt, err=f"{type(e).__name__}: {e}",
                )
        if last_err is not None:
            raise RuntimeError(
                f"embedder.health() failed after retries: {last_err}"
            ) from last_err
        meta = ChromaVectorStore.build_meta(
            embedder_kind=s.embedder.kind,
            embedder_model=self.embedder.model,
            embedder_dim=self.embedder.dim,
        )
        self.vec.open(meta)
        self.lex.open()
        self.graph.open()

        # Phase 35 (D1): pre-warm the reranker. With kind="cross_encoder"
        # the first search query would otherwise pay the model's lazy-load
        # cost (~3s warm, ~16s cold) — a P99 outlier that catches users on
        # their very first MCP tool call. health() forces the model into
        # memory at server-start time. Best-effort: a reranker init failure
        # falls back to no-op rerank rather than blocking the server.
        try:
            await self.reranker.health()
        except Exception as e:
            log.warning("mcp.reranker_health_failed",
                        err=f"{type(e).__name__}: {e}")

        # Phase 19: HyDE retriever plan. Built lazily so the import doesn't
        # fail when the user doesn't have a chat-completion-capable model
        # loaded in LM Studio. The plan internally probes /v1/models on
        # first use and silently falls back to literal-only search if no
        # suitable LLM is present — so this is safe to wire ALWAYS-ON.
        # The intent classifier inside the plan ensures identifier queries
        # never spend an LLM call.
        from code_rag.retrieval.hyde import HydeRetrieverPlan, LMStudioHyDEGenerator
        # Resolve from config; if no hyde_model is set, run literal-only.
        # Don't fabricate a default model id — mismatching what's loaded in
        # LM Studio just makes every HyDE call 404 and adds latency.
        hyde_model = getattr(s.embedder, "hyde_model", None)
        try:
            hyde_gen = (
                LMStudioHyDEGenerator(
                    base_url=s.embedder.base_url,
                    model=str(hyde_model),
                    timeout_s=15.0,
                ) if hyde_model else None
            )
            hyde_plan = HydeRetrieverPlan(generator=hyde_gen)
        except Exception as e:
            log.warning("mcp.hyde_disabled", err=str(e))
            hyde_plan = HydeRetrieverPlan(generator=None)

        # Phase 30 query rewriter (built only if enabled in config). Free
        # local rewrites by default; opt into LLM expansions by setting
        # `[query_rewriter].model` in config.toml. Returns None when
        # disabled, so the searcher silently falls back to literal-only.
        from code_rag.factory import (
            build_query_decomposer,
            build_query_rewriter,
            build_reflector,
        )
        self.rewriter = build_query_rewriter(s)
        # Phase 37-A: query decomposition (multi-part question splitting).
        # Disabled by default; opt in via `[decompose].enabled = true` plus
        # a chat model. Best-effort — null = pure single-arm search.
        self.decomposer = build_query_decomposer(s)
        # Phase 37-B: post-rerank reflection. Disabled by default; opt in
        # via `[reflection].enabled = true` plus a chat model. Confidence-
        # gated so confident queries pay no LLM cost.
        self.reflector = build_reflector(s)

        # MCP server is purely a READER. Writes (reindex, ingest) happen via
        # the live watcher or the CLI. This avoids lock contention with the
        # watcher on Kuzu, which only permits a single writer.
        self.searcher = HybridSearcher(
            self.embedder, self.vec, self.lex, self.reranker,
            graph_store=self.graph,
            hyde_plan=hyde_plan,
            rewriter=self.rewriter,
            decomposer=self.decomposer,
            reflector=self.reflector,
        )
        log.info("mcp.resources.open",
                 embedder=self.embedder.model, dim=self.embedder.dim,
                 chunks=self.vec.count(),
                 hyde="on" if hyde_plan is not None else "off",
                 rewriter="on" if self.rewriter is not None else "off",
                 decomposer="on" if self.decomposer is not None else "off",
                 reflector="on" if self.reflector is not None else "off")

    async def close(self) -> None:
        if self.vec is not None:
            self.vec.close()
        if self.lex is not None:
            self.lex.close()
        if self.graph is not None:
            self.graph.close()
        if self.embedder is not None:
            await self.embedder.aclose()
        if isinstance(self.reranker, LMStudioReranker):
            await self.reranker.aclose()
        # Phase 30: tear down rewriter (cache + httpx client).
        rewriter = getattr(self, "rewriter", None)
        if rewriter is not None:
            from code_rag.retrieval.query_rewriter import LMStudioQueryRewriter
            if isinstance(rewriter, LMStudioQueryRewriter):
                await rewriter.aclose()
                if rewriter._cache is not None:  # pyright: ignore[reportPrivateUsage]
                    rewriter._cache.close()  # pyright: ignore[reportPrivateUsage]
        # Phase 37: tear down decomposer + reflector httpx clients.
        decomposer = getattr(self, "decomposer", None)
        if decomposer is not None:
            from code_rag.retrieval.decompose import LMStudioQueryDecomposer
            if isinstance(decomposer, LMStudioQueryDecomposer):
                await decomposer.aclose()
        reflector = getattr(self, "reflector", None)
        if reflector is not None:
            from code_rag.retrieval.reflection import LMStudioReflector
            if isinstance(reflector, LMStudioReflector):
                await reflector.aclose()


# ---- Handler dispatch -----------------------------------------------------


async def _tool_search_code(res: ServerResources, args: dict[str, Any]) -> dict[str, Any]:
    assert res.searcher is not None
    # Phase 37-K: optional recency filter + root-preference boost. None
    # means "not provided" so SearchParams uses the dataclass default
    # (no filter / no boost).
    rfo_raw = args.get("recent_files_only_days")
    recent_days = int(rfo_raw) if rfo_raw is not None else None
    params = SearchParams(
        k_final=int(args.get("k", 8)),
        k_vector=res.settings.reranker.top_k_in,
        k_lexical=res.settings.reranker.top_k_in,
        k_rerank_in=res.settings.reranker.top_k_in,
        path_glob=args.get("path_glob"),
        language=args.get("lang"),
        min_score=float(args.get("min_score", 0.0)),
        max_chars_per_hit=int(args.get("max_chars_per_hit", 1200)),
        attach_neighbors=bool(args.get("attach_neighbors", False)),
        recent_files_only_days=recent_days,
        prefer_root=args.get("prefer_root"),
    )
    resp = await res.searcher.search_full(str(args["query"]), params)
    hits_out = []
    for h in resp.hits:
        entry: dict[str, Any] = {
            "chunk_id": h.chunk.id,
            "path": h.chunk.path,
            "symbol": h.chunk.symbol,
            "kind": h.chunk.kind.value,
            "start_line": h.chunk.start_line,
            "end_line": h.chunk.end_line,
            "language": h.chunk.language,
            "score": round(h.score, 4),
            "source": h.source,
            "match_reason": h.match_reason,
            "text": h.chunk.text,
        }
        if params.attach_neighbors:
            nb = resp.neighborhood.get(h.chunk.id, {"callers": [], "callees": []})
            entry["neighbors"] = {
                "callers": [_ref_to_dict(r) for r in nb["callers"]],
                "callees": [_ref_to_dict(r) for r in nb["callees"]],
            }
        hits_out.append(entry)
    return {
        "hits": hits_out,
        "no_confident_match": resp.no_confident_match,
        "elapsed_ms": round(resp.elapsed_ms, 1),
    }


async def _tool_get_chunk_text(res: ServerResources, args: dict[str, Any]) -> dict[str, Any]:
    """Fetch a chunk's full text by id. Uses the lexical store (SQLite) since
    it has the authoritative full-text copy without the vector overhead."""
    assert res.lex is not None
    chunk_id = str(args["chunk_id"])
    # Reach into the lexical store's connection for a targeted fetch.
    conn = getattr(res.lex, "_conn", None)
    lock = getattr(res.lex, "_lock", None)
    if conn is None:
        return {"error": "lexical store not open"}
    with lock if lock is not None else _NoopCM():
        row = conn.execute(
            "SELECT id, path, symbol, language, kind, start_line, end_line, text "
            "FROM chunks WHERE id = ?",
            (chunk_id,),
        ).fetchone()
    if not row:
        return {"error": f"chunk not found: {chunk_id}"}
    return {
        "chunk_id":   row[0],
        "path":       row[1],
        "symbol":     row[2],
        "language":   row[3],
        "kind":       row[4],
        "start_line": row[5],
        "end_line":   row[6],
        "text":       row[7],
    }


class _NoopCM:
    """Context-manager placeholder when the caller didn't supply a lock."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *a: Any) -> None:
        return None


async def _tool_get_symbol(res: ServerResources, args: dict[str, Any]) -> dict[str, Any]:
    assert res.graph is not None
    refs = res.graph.find_symbol(str(args["name"]), path=args.get("path"))
    return {"symbols": [_ref_to_dict(r) for r in refs]}


async def _tool_get_callers(res: ServerResources, args: dict[str, Any]) -> dict[str, Any]:
    assert res.graph is not None
    refs = res.graph.callers_of(str(args["symbol"]), path=args.get("path"))
    return {"callers": [_ref_to_dict(r) for r in refs]}


async def _tool_get_callees(res: ServerResources, args: dict[str, Any]) -> dict[str, Any]:
    assert res.graph is not None
    refs = res.graph.callees_of(str(args["symbol"]), path=args.get("path"))
    return {"callees": [_ref_to_dict(r) for r in refs]}


async def _tool_get_file_context(res: ServerResources, args: dict[str, Any]) -> dict[str, Any]:
    assert res.graph is not None
    path = str(args["path"])
    max_n = max(1, min(200, int(args.get("max_neighbors", 25))))
    # All symbols defined in the file. We walk once per file-level symbol to
    # get incoming calls — cheap compared to the search pipeline.
    defined = _all_symbols_in_file(res.graph, path)

    context = []
    truncated_any = False
    for s in defined:
        callers = res.graph.callers_of(s.symbol, path=path)
        callees = res.graph.callees_of(s.symbol, path=path)
        callers_trunc = len(callers) > max_n
        callees_trunc = len(callees) > max_n
        if callers_trunc or callees_trunc:
            truncated_any = True
        context.append({
            "symbol": s.symbol,
            "kind": s.kind,
            "start_line": s.start_line,
            "end_line": s.end_line,
            "callers": [_ref_to_dict(r) for r in callers[:max_n]],
            "callers_total": len(callers),
            "callees": [_ref_to_dict(r) for r in callees[:max_n]],
            "callees_total": len(callees),
        })
    return {
        "path": path,
        "symbols": context,
        "truncated": truncated_any,
        "max_neighbors": max_n,
    }


async def _tool_index_stats(res: ServerResources, args: dict[str, Any]) -> dict[str, Any]:
    assert res.vec is not None and res.lex is not None
    meta_path = res.settings.index_meta_path
    meta = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else None
    return {
        "meta": meta,
        "vector_count":  res.vec.count(),
        "lexical_count": res.lex.count(),
    }


async def _tool_search_code_anchored(
    res: ServerResources, args: dict[str, Any],
) -> dict[str, Any]:
    """Phase 21: graph-augmented retrieval — hybrid search re-ranked by
    proximity to an anchor symbol in the call graph."""
    from code_rag.retrieval.graph_augmented import GraphAugmentedSearcher
    assert res.searcher is not None
    params = SearchParams(
        k_final=int(args.get("k", 8)),
        k_vector=res.settings.reranker.top_k_in,
        k_lexical=res.settings.reranker.top_k_in,
        k_rerank_in=res.settings.reranker.top_k_in,
        path_glob=args.get("path_glob"),
        language=args.get("lang"),
    )
    aug = GraphAugmentedSearcher(res.searcher, res.graph)
    result = await aug.search(
        str(args["query"]),
        params,
        anchor_symbol=str(args["anchor_symbol"]),
        anchor_path=args.get("anchor_path"),
        hops=int(args.get("hops", 1)),
    )
    return result.as_dict()


async def _tool_ensure_workspace_indexed(
    res: ServerResources, args: dict[str, Any],
) -> dict[str, Any]:
    """Register `path` as a dynamic root and kick off a background index.

    Idempotent: if the path is already under a configured or dynamic root,
    we short-circuit. Otherwise we append to `dynamic_roots.json` and spawn
    a detached `code-rag index --path <path>` subprocess that writes into
    the same stores — guarded by the existing single-writer semantics on
    Kuzu (caller is expected to stop the live watcher first if running a
    large bulk ingest; incremental add of a single repo usually coexists).
    """
    import subprocess
    import sys

    from code_rag.dynamic_roots import DynamicRoots

    raw = str(args.get("path", "")).strip()
    if not raw:
        return {"error": "path is required"}
    path = Path(raw).resolve()
    if not path.exists() or not path.is_dir():
        return {"error": f"path does not exist or is not a directory: {path}"}

    # Already under any configured or dynamic root? Nothing to do.
    for r in res.settings.all_roots():
        try:
            path.relative_to(r.resolve())
            return {
                "already_indexed": True,
                "registered":      False,
                "path":             path.as_posix(),
                "matched_root":     r.as_posix(),
                "note":             "path is already under an indexed root",
            }
        except ValueError:
            continue

    # Register in the persistent dynamic-roots file.
    dyn = DynamicRoots.load(res.settings.dynamic_roots_path)
    added = dyn.add(path, source="mcp.ensure_workspace_indexed")

    # Phase 38 (audit fix): coordinate concurrent ensure_workspace_indexed
    # invocations from multiple Claude Code sessions. Without this lock,
    # two MCP processes can each spawn `code-rag index --path X` for the
    # SAME path, both opening Kuzu in writer mode → second process fails
    # noisily into auto_index.log that the user never sees. Skip the spawn
    # when an indexer for this path is already in flight.
    indexer_lock_dir = res.settings.paths.data_dir / "ensure_locks"
    indexer_lock_dir.mkdir(parents=True, exist_ok=True)
    # File name = sanitized path. Slashes/colons replaced with underscores.
    lock_name = (
        path.as_posix()
        .replace("/", "_").replace(":", "_").replace("\\", "_")
        .strip("_")[:200] + ".lock"
    )
    lock_file = indexer_lock_dir / lock_name
    if lock_file.exists():
        try:
            existing_pid = int(lock_file.read_text("utf-8").strip())
            from code_rag.util.proc_hygiene import is_process_alive
            if is_process_alive(existing_pid):
                return {
                    "already_indexed":  False,
                    "registered":       added,
                    "path":             path.as_posix(),
                    "background_pid":   existing_pid,
                    "note": ("indexer already running for this path; "
                             "watching the existing pid"),
                }
        except (ValueError, OSError):
            # Stale or unreadable lock — fall through and overwrite.
            pass

    # Spawn a detached indexing subprocess. We use sys.executable to stay
    # inside the same venv; `-m code_rag index --path` handles its own
    # lifecycle (opens stores, reindexes, closes). stdout/err redirected
    # to a dedicated log so the MCP stdio stream stays clean.
    log_file = res.settings.paths.log_dir / "auto_index.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Ensure the subprocess picks up the same config the MCP server did.
    if "CODE_RAG_CONFIG" not in env:
        # The active config may not be advertised via env; best-effort fallback
        # to the default lookup.
        pass

    # On Windows, CREATE_NEW_PROCESS_GROUP + DETACHED_PROCESS lets the child
    # survive independently of this MCP server. On POSIX, start_new_session
    # achieves the same via setsid.
    popen_kwargs: dict[str, Any] = {
        "stdout": open(log_file, "a", encoding="utf-8"),  # noqa: SIM115
        "stderr": subprocess.STDOUT,
        "stdin":  subprocess.DEVNULL,
        "env":    env,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | 0x00000008  # DETACHED_PROCESS
        )
    else:
        popen_kwargs["start_new_session"] = True
    proc = subprocess.Popen(
        [sys.executable, "-m", "code_rag", "index", "--path", str(path)],
        **popen_kwargs,
    )
    # Stamp the singleton lock with the spawned PID so future invocations
    # for the same path can detect the in-flight indexer. Best-effort:
    # disk-full / permission failure just means future calls double-spawn,
    # not a correctness issue.
    with contextlib.suppress(OSError):
        lock_file.write_text(str(proc.pid), encoding="utf-8")
    log.info("mcp.ensure_workspace_indexed.spawned",
             path=path.as_posix(), pid=proc.pid, added=added)
    return {
        "already_indexed": False,
        "registered":      added,
        "path":            path.as_posix(),
        "background_pid":  proc.pid,
        "log_file":        str(log_file),
        "note":            ("indexing started in background; `search_code` "
                            "will see results as chunks land"),
    }


_HANDLERS = {
    "search_code":               _tool_search_code,
    "search_code_anchored":      _tool_search_code_anchored,
    "get_chunk_text":            _tool_get_chunk_text,
    "get_symbol":                _tool_get_symbol,
    "get_callers":               _tool_get_callers,
    "get_callees":               _tool_get_callees,
    "get_file_context":          _tool_get_file_context,
    "index_stats":               _tool_index_stats,
    "ensure_workspace_indexed":  _tool_ensure_workspace_indexed,
}


# ---- helpers --------------------------------------------------------------


def _ref_to_dict(r: Any) -> dict[str, Any]:
    return {
        "path": r.path, "symbol": r.symbol, "kind": r.kind,
        "start_line": r.start_line, "end_line": r.end_line,
    }


def _all_symbols_in_file(graph: GraphStore, path: str) -> list[Any]:
    """Workaround for 'all symbols in file' — call private store if available,
    otherwise fall back to find_symbol with an impossible name + path (returns
    empty) — we'll add a proper API later."""
    # Public API: find_symbol doesn't support a wildcard. We use a Cypher-ish
    # backdoor on KuzuGraphStore via a known method if available, else we
    # return []. This is the one place where we peek through the interface.
    run = getattr(graph, "_run", None)
    if callable(run):
        rows = run(
            "MATCH (s:Symbol {path: $p}) RETURN s.path, s.symbol, s.kind, s.start_line, s.end_line",
            {"p": path},
        )
        out = []
        from code_rag.interfaces.graph_store import SymbolRef
        for row in rows or []:
            if len(row) < 5:
                continue
            out.append(SymbolRef(
                path=str(row[0]), symbol=str(row[1]), kind=str(row[2]),
                start_line=int(row[3]), end_line=int(row[4]),
            ))
        return out
    return []


# ---- server entry ---------------------------------------------------------


def build_server(res: ServerResources) -> Server:
    server: Server = Server(res.settings.mcp.name)

    # The mcp package's decorators are untyped in the published stubs; we
    # silence the narrow per-module warnings rather than weakening strictness
    # globally. See [tool.mypy.overrides] in pyproject.toml.
    @server.list_tools()  # type: ignore[no-untyped-call]
    async def _list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        handler = _HANDLERS.get(name)
        if handler is None:
            payload: dict[str, Any] = {"error": f"unknown tool: {name}"}
        else:
            try:
                payload = await handler(res, arguments or {})
            except Exception as e:
                log.exception("mcp.tool.error", tool=name)
                payload = {"error": f"{type(e).__name__}: {e}"}
        return [TextContent(type="text", text=json.dumps(payload, indent=2, ensure_ascii=False))]

    return server


async def run_stdio(settings: Settings) -> None:
    res = ServerResources(settings)
    await res.open()
    server = build_server(res)
    try:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    finally:
        await res.close()


def main_entry() -> None:
    """Entry point for `code-rag mcp` CLI subcommand.

    Phase 32: starts a parent-death watchdog so this MCP subprocess exits
    cleanly when Claude Code (our parent) dies. Prevents orphan
    accumulation across restarts.
    """
    from code_rag.util.proc_hygiene import start_parent_death_watchdog
    # Capture parent PID at process start. On Windows venv stubs we are the
    # anaconda child of the .venv launcher, but the watchdog walks ancestors
    # of `os.getppid()` correctly because if the venv stub dies (which it
    # does when claude.exe dies), our PPID becomes orphaned and the
    # `is_process_alive` probe returns False.
    start_parent_death_watchdog()
    asyncio.run(run_stdio(load_settings()))

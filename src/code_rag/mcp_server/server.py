"""MCP server exposing the code-rag tools over stdio.

Registered tools (must match `_HANDLERS` and `TOOLS` below):
  * search_code       — hybrid search (vector + lexical + rerank)
  * get_chunk_text    — fetch full text of a truncated chunk by id
  * get_symbol        — locate a symbol by name
  * get_callers       — graph walk: who calls X
  * get_callees       — graph walk: what does X call
  * get_file_context  — file's symbols + immediate graph neighborhood
  * index_stats       — counts + metadata + last-updated

Writes (reindex, doc ingestion) are NOT exposed here — the MCP server is
read-only by design. The autostart watcher is the single writer.

All tools return structured JSON; the client (Claude Code) sees concise text
blocks and can pass them straight into a prompt.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from code_rag.config import Settings, load_settings
from code_rag.embedders.lm_studio import LMStudioEmbedder
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
            "if `no_confident_match: true` comes back."
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

        await self.embedder.health()
        meta = ChromaVectorStore.build_meta(
            embedder_kind=s.embedder.kind,
            embedder_model=self.embedder.model,
            embedder_dim=self.embedder.dim,
        )
        self.vec.open(meta)
        self.lex.open()
        self.graph.open()

        # MCP server is purely a READER. Writes (reindex, ingest) happen via
        # the live watcher or the CLI. This avoids lock contention with the
        # watcher on Kuzu, which only permits a single writer.
        self.searcher = HybridSearcher(
            self.embedder, self.vec, self.lex, self.reranker,
            graph_store=self.graph,
        )
        log.info("mcp.resources.open",
                 embedder=self.embedder.model, dim=self.embedder.dim,
                 chunks=self.vec.count())

    async def close(self) -> None:
        if self.vec is not None:
            self.vec.close()
        if self.lex is not None:
            self.lex.close()
        if self.graph is not None:
            self.graph.close()
        if isinstance(self.embedder, LMStudioEmbedder):
            await self.embedder.aclose()
        if isinstance(self.reranker, LMStudioReranker):
            await self.reranker.aclose()


# ---- Handler dispatch -----------------------------------------------------


async def _tool_search_code(res: ServerResources, args: dict[str, Any]) -> dict[str, Any]:
    assert res.searcher is not None
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


_HANDLERS = {
    "search_code":      _tool_search_code,
    "get_chunk_text":   _tool_get_chunk_text,
    "get_symbol":       _tool_get_symbol,
    "get_callers":      _tool_get_callers,
    "get_callees":      _tool_get_callees,
    "get_file_context": _tool_get_file_context,
    "index_stats":      _tool_index_stats,
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
    """Entry point for `code-rag mcp` CLI subcommand."""
    asyncio.run(run_stdio(load_settings()))

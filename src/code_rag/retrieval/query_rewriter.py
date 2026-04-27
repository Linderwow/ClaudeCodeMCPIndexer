"""Phase 30: query rewriting / identifier expansion.

Why this exists
---------------
Code RAG queries mix natural language with identifiers in awkward ways:

    "how does OnBarUpdate handle gaps"
    "where do we call ensure_lm_studio_ready"
    "the connect_db helper that wraps psycopg2"

A pure dense embedder treats identifiers as opaque tokens; BM25 misses
casing and snake_case/CamelCase variants. This rewriter pre-processes the
query in two cheap, deterministic steps:

  1. **Local extraction** — pull identifier-shaped tokens out of the query
     (camelCase, snake_case, alpha-numeric mixes). Generate the obvious
     casing variants: `OnBarUpdate -> on_bar_update, on bar update`. No
     LLM call needed; fast, free, can't fail.

  2. **Optional LLM expansion** — if a chat model is configured, ask the
     small (qwen/qwen3-1.7b) model to suggest semantic synonyms / related
     identifiers. Cheap (~80 tokens prompt, ~40 tokens output, ~1s).
     Cached on disk so repeats are free.

The merged query is whatever the searcher fuses with. We DO NOT replace the
original — we add a parallel arm at search time. So even if the rewriter
hallucinates, the literal arm protects the floor.

Cache
-----
SQLite single-table key/value with the query hash as the key. TTL: 24 h.
Cap: 5000 entries. Eviction: LRU on prune.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from code_rag.logging import get
from code_rag.retrieval.fusion import extract_identifiers

log = get(__name__)

# How long to keep a cached rewrite. 24 h is generous: queries CAN drift in
# meaning over time but in practice "where is OnBarUpdate" stays stable.
_CACHE_TTL_S = 24 * 3600
# How many entries to retain. Pruning happens lazily on write.
_CACHE_MAX = 5_000


# Phase 30: hard cap on parallel arms produced by a single rewrite. Each
# arm fires its own embedding call, and LM Studio serializes embeddings on
# a single GPU slot — so 5 arms per query × N concurrent queries piles up
# fast. Two arms (literal + one variant) is the right balance: it covers
# the snake/camel mismatch without melting the embedder.
_MAX_ARMS = 2


@dataclass(frozen=True)
class Rewrite:
    """A rewritten query — the original plus a list of expanded variants.

    `arms` is the list passed to the retriever as parallel query arms. It
    ALWAYS includes the original query at full weight; expansions ride at
    half weight so they can't drown the literal signal. Capped at
    `_MAX_ARMS` to avoid overloading the embedder.
    """

    original: str
    expansions: list[str] = field(default_factory=list)
    via: str = "literal"     # "literal" | "local" | "llm" | "cache"

    @property
    def arms(self) -> list[tuple[str, float]]:
        """Search-arm list. Original at weight 1.0; first expansion at 0.5.
        Cap = `_MAX_ARMS` (default 2) so we never fire more than 1 extra
        embedding call per search."""
        out: list[tuple[str, float]] = [(self.original, 1.0)]
        for e in self.expansions:
            if e and e != self.original:
                out.append((e, 0.5))
                if len(out) >= _MAX_ARMS:
                    break
        return out


# ---- local (no LLM) expansion ---------------------------------------------


_CAMEL_RE = re.compile(r"(?<!^)(?=[A-Z])")


def _camel_to_snake(s: str) -> str:
    return _CAMEL_RE.sub("_", s).lower()


def _snake_to_camel(s: str) -> str:
    parts = s.split("_")
    if len(parts) == 1:
        return s
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _identifier_variants(idents: list[str]) -> list[str]:
    """Generate snake/camel/spaced variants for each identifier.

    Returns deduped variants that DIFFER from the originals.
    """
    seen: set[str] = {x.lower() for x in idents}
    out: list[str] = []
    for tok in idents:
        # snake_case → camelCase + spaced.
        if "_" in tok:
            cam = _snake_to_camel(tok)
            if cam.lower() not in seen:
                seen.add(cam.lower())
                out.append(cam)
            spaced = tok.replace("_", " ")
            if spaced.lower() not in seen:
                seen.add(spaced.lower())
                out.append(spaced)
        # CamelCase → snake_case + spaced.
        elif any(ch.isupper() for ch in tok[1:]):
            sn = _camel_to_snake(tok)
            if sn.lower() not in seen:
                seen.add(sn.lower())
                out.append(sn)
            # "OnBarUpdate" → "on bar update"
            spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", tok).lower()
            if spaced.lower() not in seen:
                seen.add(spaced.lower())
                out.append(spaced)
    return out


def local_rewrite(query: str) -> Rewrite:
    """Cheap, deterministic rewrite. No LLM. Always safe to call."""
    idents = extract_identifiers(query)
    if not idents:
        return Rewrite(original=query, expansions=[], via="literal")
    variants = _identifier_variants(idents)
    if not variants:
        return Rewrite(original=query, expansions=[], via="literal")
    # We DON'T inject the variants into the same string; we send them as
    # parallel arms. Each variant becomes its own retrieval arm.
    return Rewrite(original=query, expansions=variants, via="local")


# ---- cache -----------------------------------------------------------------


class RewriteCache:
    """Tiny SQLite-backed cache for query rewrites.

    Schema (one table, no migrations needed):
        rewrites(hash TEXT PRIMARY KEY, query TEXT, payload TEXT,
                 created_at INT, last_used_at INT)

    `payload` is the JSON-encoded `Rewrite` minus the original (the original
    is stored separately for debug). On read, we update last_used_at so LRU
    pruning has something to bite on.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._con: sqlite3.Connection | None = None
        # SQLite is thread-safe enough for our concurrency, but we still
        # serialize via a lock to avoid `database is locked` flares.
        self._lock = asyncio.Lock()

    def open(self) -> None:
        if self._con is not None:
            return
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(self._db_path), isolation_level=None)
        self._con.execute("PRAGMA journal_mode=WAL")
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS rewrites ("
            "hash TEXT PRIMARY KEY,"
            "query TEXT,"
            "payload TEXT,"
            "created_at INTEGER,"
            "last_used_at INTEGER)"
        )

    def close(self) -> None:
        if self._con is not None:
            with contextlib.suppress(sqlite3.Error):
                self._con.close()
            self._con = None

    @staticmethod
    def _hash(query: str) -> str:
        return hashlib.blake2b(query.encode("utf-8"), digest_size=8).hexdigest()

    async def get(self, query: str) -> Rewrite | None:
        async with self._lock:
            con = self._require()
            now = int(time.time())
            row = con.execute(
                "SELECT payload, created_at FROM rewrites WHERE hash=?",
                (self._hash(query),),
            ).fetchone()
            if not row:
                return None
            payload, created = row
            if now - int(created) > _CACHE_TTL_S:
                con.execute("DELETE FROM rewrites WHERE hash=?",
                            (self._hash(query),))
                return None
            con.execute(
                "UPDATE rewrites SET last_used_at=? WHERE hash=?",
                (now, self._hash(query)),
            )
            try:
                d = json.loads(payload)
                return Rewrite(
                    original=query,
                    expansions=list(d.get("expansions", [])),
                    via="cache",
                )
            except (json.JSONDecodeError, TypeError):
                return None

    async def put(self, rewrite: Rewrite) -> None:
        async with self._lock:
            con = self._require()
            now = int(time.time())
            payload = json.dumps({"expansions": rewrite.expansions})
            con.execute(
                "INSERT OR REPLACE INTO rewrites "
                "(hash, query, payload, created_at, last_used_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (self._hash(rewrite.original), rewrite.original, payload, now, now),
            )
            # Lazy pruning: bound size by deleting the oldest-used entries
            # whenever we cross the cap.
            count = con.execute("SELECT COUNT(*) FROM rewrites").fetchone()[0]
            if count > _CACHE_MAX:
                con.execute(
                    "DELETE FROM rewrites WHERE hash IN ("
                    "SELECT hash FROM rewrites ORDER BY last_used_at ASC LIMIT ?"
                    ")",
                    (count - _CACHE_MAX,),
                )

    def _require(self) -> sqlite3.Connection:
        if self._con is None:
            raise RuntimeError("RewriteCache not open; call .open() first")
        return self._con


# ---- LLM expansion (optional) ---------------------------------------------


class LMStudioQueryRewriter:
    """Calls a small chat model (qwen3-1.7b) to suggest related identifiers
    or rephrasings. Local-rewrite expansions are merged in regardless.

    NEVER raises into the caller — any failure (network, parse, timeout)
    falls back to local rewrite.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        cache: RewriteCache | None = None,
        timeout_s: float = 8.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = httpx.Timeout(timeout_s, connect=3.0)
        self._cache = cache
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(
                        base_url=self._base_url, timeout=self._timeout,
                    )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def rewrite(self, query: str) -> Rewrite:
        if self._cache is not None:
            cached = await self._cache.get(query)
            if cached is not None:
                return cached

        # Always start from the local rewrite — it's free and never wrong.
        local = local_rewrite(query)

        # Skip the LLM call if no model is configured (local-only mode) or
        # there are no identifiers to expand. The LLM rarely helps for pure
        # prose questions and adds latency.
        if not self._model or not extract_identifiers(query):
            if self._cache is not None:
                await self._cache.put(local)
            return local

        llm_expansions = await self._call_llm(query)
        if llm_expansions is None:
            if self._cache is not None:
                await self._cache.put(local)
            return local

        merged_set = {e.lower(): e for e in local.expansions}
        for e in llm_expansions:
            if e and e.lower() != query.lower() and e.lower() not in merged_set:
                merged_set[e.lower()] = e
        merged = list(merged_set.values())[:8]  # cap so we don't fan out wildly
        rewrite = Rewrite(original=query, expansions=merged, via="llm")
        if self._cache is not None:
            await self._cache.put(rewrite)
        return rewrite

    async def _call_llm(self, query: str) -> list[str] | None:
        prompt = (
            "You expand a code search query with closely-related identifiers "
            "and short phrasings.\n"
            "Output ONE JSON object only. No prose, no markdown.\n\n"
            f"Query: {query}\n\n"
            'Return: {"expansions": [<short related queries / identifiers>]}\n'
            "Rules:\n"
            "- 1 to 6 expansions.\n"
            "- Each expansion is a short string (≤60 chars) that a developer "
            "might type to find the same code.\n"
            "- Prefer SPECIFIC over GENERIC. No 'documentation', no 'code'.\n"
            "- Do NOT repeat the original query."
        )
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "query_expansions",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "expansions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 0,
                            "maxItems": 6,
                        },
                    },
                    "required": ["expansions"],
                    "additionalProperties": False,
                },
            },
        }
        try:
            client = await self._ensure_client()
            r = await client.post("/chat/completions", json={
                "model": self._model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 200,
                "response_format": schema,
            })
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
        except Exception as e:
            # Broad except: this is best-effort. Anything wrong (network,
            # malformed response, runtime quirks) falls back to local.
            log.warning("rewriter.llm_fail", err=f"{type(e).__name__}: {e}")
            return None

        try:
            obj = json.loads(content) if isinstance(content, str) else None
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        arr = obj.get("expansions")
        if not isinstance(arr, list):
            return None
        out = [str(x).strip() for x in arr if isinstance(x, str) and x.strip()]
        return out[:6] or None


# ---- convenience -----------------------------------------------------------


async def rewrite_or_local(
    query: str, rewriter: LMStudioQueryRewriter | None,
) -> Rewrite:
    """Use the LLM rewriter if configured, otherwise just local rewrite.

    The MCP server / CLI both call this so they don't have to repeat the
    None-check.
    """
    if rewriter is not None:
        return await rewriter.rewrite(query)
    return local_rewrite(query)

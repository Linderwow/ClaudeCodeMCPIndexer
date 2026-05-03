"""Phase 37-A: query decomposition.

Why this exists
---------------
NVIDIA's RAG blueprint highlights "Query decomposition" as a Query
Processing service feature. The motivating case in code search:

    "How does authentication flow from login through session refresh
     to logout?"

A single embedding will find chunks that vaguely talk about all three
phases but won't pin down any one of them. Decomposing into:

    - "login flow"
    - "session refresh"
    - "logout"

…gives each phase its own retrieval arm so the fused top-k contains
specific chunks for every leg of the question.

This complements (does not replace) Phase 30's query rewriter:

    Phase 30 (rewriter) : casing/synonym variants of the SAME query.
    Phase 37-A (decompose): break ONE multi-part query into N sub-queries.

Both add arms to the same plan. We cap the total arm count downstream
to keep embedder load bounded.

Design
------
- Heuristic gate first: bail without an LLM call when the query clearly
  isn't multi-part (short, no conjunctions, single sentence).
- LLM call shape mirrors the rewriter: chat completions with
  json_schema, conservative temperature, broad fallback to literal.
- Output is `Decomposition`, a dataclass that exposes `arms` matching
  the rewriter's interface so the searcher composes them uniformly.
- NEVER raises into the searcher. Anything wrong falls back to a
  one-arm plan with the literal query.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field

import httpx

from code_rag.logging import get

log = get(__name__)


# Cap on sub-queries we accept from the LLM. More arms => more embedding
# calls and bigger fan-out into the fusion stage. Three covers the
# overwhelming majority of multi-part questions ("X and Y and Z") without
# exploding the search budget.
_MAX_SUBQUERIES = 3

# Sub-query weight relative to the literal (1.0). Lower than the rewriter's
# casing-variant weight (0.5) because a sub-query is a different question;
# we don't want it to outvote the original on hits that satisfy the whole
# query.
_SUBQUERY_WEIGHT = 0.6


# ---- heuristic gate -------------------------------------------------------

# Cheap pre-filter: if the query is short or has no obvious conjunction,
# skip the LLM call entirely. Saves ~1s per non-multipart query.
_CONJUNCTION_RE = re.compile(
    r"\b(and|then|after|before|while|plus|also|as well as|both|including|"
    r"followed by|leading to|through to)\b",
    re.IGNORECASE,
)
_QUESTION_PARTS_RE = re.compile(r"[?]")


def looks_multipart(query: str) -> bool:
    """Cheap heuristic: should we even consider decomposing?

    True iff:
      - The query is at least 8 words (anything shorter rarely benefits).
      - AND it contains a coordinating conjunction OR multiple
        question marks.

    Both signals together filter out most identifier lookups and short
    NL questions ("where does X live?") so the LLM gate fires only when
    decomposition has a real chance of helping.
    """
    q = query.strip()
    if not q:
        return False
    word_count = len(q.split())
    if word_count < 8:
        return False
    has_conj = bool(_CONJUNCTION_RE.search(q))
    multi_q = len(_QUESTION_PARTS_RE.findall(q)) >= 2
    return has_conj or multi_q


# ---- decomposition result -------------------------------------------------


@dataclass(frozen=True)
class Decomposition:
    """Result of a decomposition pass.

    `subqueries` is empty when the query wasn't multi-part (or the LLM
    declined to split it). `arms` always includes the original at full
    weight; sub-queries ride at `_SUBQUERY_WEIGHT` so they augment, not
    replace, the literal signal.
    """

    original: str
    subqueries: list[str] = field(default_factory=list)
    via: str = "skip"   # "skip" | "llm" | "fallback"

    @property
    def arms(self) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = [(self.original, 1.0)]
        for sq in self.subqueries:
            if sq and sq.strip() and sq.strip().lower() != self.original.strip().lower():
                out.append((sq, _SUBQUERY_WEIGHT))
        return out


# ---- LLM-backed decomposer ------------------------------------------------


class LMStudioQueryDecomposer:
    """Calls a chat model to split a multi-part question into sub-queries.

    Pattern mirrors `LMStudioQueryRewriter`:
      - Best-effort. Any failure (network, parse, timeout) returns an
        empty Decomposition so the searcher keeps the original arm.
      - JSON-schema response_format pins the model to a structured array.
      - Heuristic gate skips the LLM for queries that clearly aren't
        multi-part.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_s: float = 8.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = httpx.Timeout(timeout_s, connect=3.0)
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

    async def decompose(self, query: str) -> Decomposition:
        if not self._model:
            return Decomposition(original=query, via="skip")
        if not looks_multipart(query):
            return Decomposition(original=query, via="skip")

        subs = await self._call_llm(query)
        if not subs:
            return Decomposition(original=query, via="fallback")
        return Decomposition(original=query, subqueries=subs, via="llm")

    async def _call_llm(self, query: str) -> list[str]:
        prompt = (
            "You decompose a developer's multi-part question into 2-3 "
            "self-contained sub-questions that, taken together, fully cover "
            "the original. Each sub-question should be searchable on its own.\n"
            "Output ONE JSON object only. No prose, no markdown.\n\n"
            f"Question: {query}\n\n"
            'Return: {"subqueries": [<short self-contained sub-questions>]}\n'
            "Rules:\n"
            "- 2 to 3 sub-queries (or empty list if the question is already atomic).\n"
            "- Each sub-query is short (<= 80 chars) and standalone.\n"
            "- Sub-queries must NOT just paraphrase the original — they must "
            "split it into distinct retrieval targets.\n"
            "- If the question is already a single concept, return an empty list."
        )
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "query_decomposition",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "subqueries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 0,
                            "maxItems": _MAX_SUBQUERIES,
                        },
                    },
                    "required": ["subqueries"],
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
            # Best-effort: never break the search pipeline.
            log.warning("decompose.llm_fail", err=f"{type(e).__name__}: {e}")
            return []

        try:
            obj = json.loads(content) if isinstance(content, str) else None
        except json.JSONDecodeError:
            return []
        if not isinstance(obj, dict):
            return []
        arr = obj.get("subqueries")
        if not isinstance(arr, list):
            return []
        out = [
            str(x).strip() for x in arr
            if isinstance(x, str) and x.strip() and len(x.strip()) <= 200
        ]
        # Drop any sub-query that is just the original re-stated.
        norm_orig = query.strip().lower()
        out = [s for s in out if s.lower() != norm_orig]
        return out[:_MAX_SUBQUERIES]

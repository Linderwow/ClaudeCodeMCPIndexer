"""Phase 19: HyDE (Hypothetical Document Embeddings) + intent classifier.

For natural-language queries like "where does the strategy size positions",
the literal query text often doesn't share many tokens with the actual
answer's chunk text. HyDE (Gao et al., 2022) sidesteps this:

    1. Use a small local LLM to generate a *hypothetical* code chunk that
       would answer the query.
    2. Embed THAT, not the literal query.
    3. Search for chunks similar to the hypothetical answer.
    4. RRF-fuse with the literal-query results to keep recall on identifier
       queries where HyDE would lose signal.

Quality lift: typically +5-15 pp Recall@3 on natural-language queries with
no measurable hit on identifier queries (because we route by intent).

Cost: one extra small-LLM call per query, ~200-500ms with Qwen3-1.7B in
LM Studio. Gated by intent classifier so it ONLY fires on NL queries.

This module is intentionally generator-agnostic: it takes an
`HyDEGenerator` protocol and you can wire any LLM behind it (LM Studio,
HF Transformers, OpenAI-compat) via the factory.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import httpx

from code_rag.logging import get

log = get(__name__)


# ---- intent classifier (heuristic, not ML) --------------------------------
#
# A learned intent classifier would be cleaner but adds a training/eval
# loop. Heuristics get us 90% of the value with zero training cost; if
# the eval harness later shows misrouting hurts recall, we upgrade.

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CAMEL_RE = re.compile(r"^[A-Z][a-z]+([A-Z][a-z0-9]*)+$")
_SNAKE_RE = re.compile(r"^[a-z][a-z0-9_]+_[a-z0-9_]+$")
_SYMBOLIC_CHARS = re.compile(r"[(){}\[\];:.,]")


def classify_intent(query: str) -> str:
    """Return one of:
       - 'identifier'        : exact name lookup; HyDE skipped
       - 'definition_lookup' : "find <thing>" / "where is X defined"; HyDE optional
       - 'natural_language'  : prose question; HyDE on
       - 'mixed'             : has identifier-like tokens AND prose; HyDE on, both flows fused

    Used by `HydeRetriever` to decide whether to spend an LLM call.
    """
    q = query.strip()
    if not q:
        return "natural_language"

    # Pure single-word identifier (CamelCase or snake_case alone).
    if _IDENTIFIER_RE.match(q) and len(q) >= 3:
        if _CAMEL_RE.match(q) or _SNAKE_RE.match(q) or any(c.isupper() for c in q[1:]):
            return "identifier"
        # bare lowercase word -- probably NL ("foo")
        return "natural_language"

    word_count = len(q.split())
    has_symbolic = bool(_SYMBOLIC_CHARS.search(q))
    looks_codey = (
        # CamelCase: lower-then-upper inside a word (e.g. OnBarUpdate)
        bool(re.search(r"\b[A-Z][a-z]+[A-Z]\w*\b", q)) or
        # All-caps acronym + suffix (e.g. MNQAlpha, AWSConfig, NLP_pipeline)
        bool(re.search(r"\b[A-Z]{2,}[A-Za-z0-9_]*\b", q)) or
        # snake_case
        bool(re.search(r"\b\w+_\w+\b", q)) or
        has_symbolic
    )

    # Short prose with codey identifiers: "find PositionSizer", "OnBarUpdate calls".
    if word_count <= 5 and looks_codey:
        return "definition_lookup"

    # Long prose with code refs: "where does the strategy size positions".
    if word_count >= 6 and looks_codey:
        return "mixed"

    if word_count >= 6:
        return "natural_language"

    return "definition_lookup"


# ---- HyDE generator interface ---------------------------------------------


class HyDEGenerator(Protocol):
    """A small-LLM that turns a natural-language question into a hypothetical
    code/doc chunk that WOULD answer it. Output is embedded by the same
    embedder as the index, so the format/length should match real chunks."""

    async def generate(self, query: str, *, max_tokens: int = 256) -> str: ...


_PROMPT = (
    "You are a senior engineer. Given a developer's question, write 4-8 lines "
    "of code (in any language) or a docstring excerpt that would PERFECTLY "
    "answer it -- as if you copy-pasted the canonical answer from a real "
    "codebase. Output ONLY the code/docstring. No explanation, no preamble, "
    "no markdown fences.\n\n"
    "Question: {query}\n\n"
    "Answer (code or docstring only):"
)


class LMStudioHyDEGenerator:
    """HyDEGenerator backed by an LM Studio chat-completions endpoint.

    Designed to use a SMALL model (e.g. Qwen3-1.7B-Instruct) -- HyDE doesn't
    need a frontier reasoner; we're using the OUTPUT as a search query, not
    as a final answer. Latency ~200-500 ms locally.

    Reuses the same `httpx` client style as the embedder/reranker so all
    LM Studio calls share connection pooling and timeout config.
    """

    def __init__(self, base_url: str, model: str, timeout_s: float = 30.0) -> None:
        import httpx
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = httpx.Timeout(timeout_s, connect=5.0)
        self._client: httpx.AsyncClient | None = None
        self._endpoint_ok: bool | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        import httpx
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout)
        return self._client

    async def generate(self, query: str, *, max_tokens: int = 256) -> str:
        """Probe-then-generate. On any LM Studio failure (model not loaded,
        endpoint missing, timeout), we fall back to the query itself -- the
        retriever then gracefully degrades to plain hybrid search.
        """
        if self._endpoint_ok is False:
            return query

        import httpx
        client = await self._ensure_client()

        # Probe /v1/models on first call to fail fast if our model isn't loaded.
        if self._endpoint_ok is None:
            try:
                r = await client.get(
                    "/models", timeout=httpx.Timeout(5.0, connect=2.0),
                )
                r.raise_for_status()
                loaded = {m.get("id") for m in r.json().get("data", [])}
                if self._model not in loaded:
                    log.warning("hyde.model_not_loaded", model=self._model,
                                loaded=sorted(x for x in loaded if x))
                    self._endpoint_ok = False
                    return query
            except Exception as e:
                log.warning("hyde.probe_failed", err=str(e))
                self._endpoint_ok = False
                return query

        payload = {
            "model": self._model,
            "messages": [{"role": "user", "content": _PROMPT.format(query=query)}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "stream": False,
        }
        try:
            r = await client.post("/chat/completions", json=payload)
            r.raise_for_status()
            data = r.json()
            self._endpoint_ok = True
            choices = data.get("choices") or []
            if not choices:
                return query
            content = choices[0].get("message", {}).get("content") or ""
            content = content.strip()
            if not content:
                return query
            # Strip code-fence wrappers the model sometimes inserts despite
            # the prompt asking for none.
            if content.startswith("```"):
                lines = content.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                content = "\n".join(lines)
            return content or query
        except Exception as e:
            log.warning("hyde.generation_failed", err=str(e))
            self._endpoint_ok = False
            return query

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


# ---- retriever wrapper ----------------------------------------------------


class HydeRetrieverPlan:
    """Decision object: given a query, produce the search plan.

    Returns a list of (text, weight) pairs to embed and search over. The
    HybridSearcher fuses results from each, weighted by RRF+rrf_k. Identifier
    queries get a single (literal, 1.0) plan and skip the LLM entirely.

    Keeping this as a plan object rather than wiring it into HybridSearcher
    directly means the eval harness can A/B "hyde on" vs "hyde off" with no
    code change to the search pipeline -- just swap the retriever.
    """

    def __init__(
        self,
        generator: HyDEGenerator | None = None,
        *,
        enable_for: tuple[str, ...] = ("natural_language", "mixed"),
    ) -> None:
        self._gen = generator
        self._enable_for = enable_for

    async def plan(self, query: str) -> list[tuple[str, float]]:
        intent = classify_intent(query)
        if self._gen is None or intent not in self._enable_for:
            return [(query, 1.0)]
        # Generate the hypothetical doc; fall back to literal-only if it
        # comes back empty or the generator errors (it returns the query
        # itself in those cases).
        try:
            hyp = await self._gen.generate(query)
        except Exception as e:
            log.warning("hyde.plan_failed", err=str(e))
            return [(query, 1.0)]
        if hyp.strip() == query.strip():
            return [(query, 1.0)]  # generator fell back; no point fusing
        return [(query, 1.0), (hyp, 0.7)]

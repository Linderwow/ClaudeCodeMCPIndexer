"""Phase 29: cross-encoder reranker via sentence-transformers.

Why this exists alongside `lm_chat`
-----------------------------------
The Phase 24 chat reranker (`LMStudioChatReranker`) works — it lifted R@1 by
+22.6% on the mined eval. But it has costs:
  - Latency: 500-1000 ms per query (the LLM has to read the candidates and
    emit a JSON ranking).
  - Fragility: relies on the small chat model to honor a json_schema;
    occasional fallback to input-order on parse failure.
  - LM Studio dependency: the chat model has to be loaded.

A purpose-built cross-encoder (e.g. `BAAI/bge-reranker-v2-m3`) is the right
tool for this job: it scores each (query, doc) pair numerically with a
single forward pass. ~50-100 ms for 50 candidates on CPU, deterministic,
and no LLM needed.

Why this is OPT-IN
------------------
sentence-transformers + torch adds ~2 GB to the install footprint. Most
users don't need it; the chat reranker is fine. So:
  - sentence-transformers is an OPTIONAL dependency
  - importing this module without the dep raises a clear ImportError
  - factory.build_reranker() gates on `kind="cross_encoder"` in config

To enable:
    pip install -e ".[cross-encoder]"
Then in config.toml:
    [reranker]
    kind  = "cross_encoder"
    model = "BAAI/bge-reranker-v2-m3"

Model is loaded once at server start (lazy: deferred until first rerank)
and reused for every subsequent call. Memory cost: ~1.2 GB for the
default model on CPU.
"""
from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from code_rag.interfaces.reranker import Reranker
from code_rag.logging import get
from code_rag.models import SearchHit

log = get(__name__)


class CrossEncoderReranker(Reranker):
    """Cross-encoder reranker. Pointwise (query, doc) scoring.

    Invariants:
      - NEVER raises into the caller. All failures fall back to input order
        with a logged warning.
      - Loads the model lazily on first call so server startup stays fast
        even if the reranker is configured but never used.
      - Idempotent: same (query, candidate set) → same ranking (deterministic
        forward pass).
    """

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        max_candidates: int = 50,
        max_chars_per_doc: int = 600,
        device: str | None = None,
    ) -> None:
        # Verify the dep is importable up front so the user gets a clear
        # error at config-load time, not at first query.
        try:
            import sentence_transformers  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "cross_encoder reranker requires `sentence-transformers`. "
                "Install with `pip install -e \".[cross-encoder]\"` from the "
                "code-rag-mcp repo, then restart the MCP server."
            ) from e

        self._model_name = model
        self._max_candidates = max_candidates
        self._max_chars = max_chars_per_doc
        # `device=None` lets sentence-transformers auto-pick (cuda → mps → cpu).
        self._device = device
        self._model: Any | None = None
        self._lock = asyncio.Lock()

    @property
    def model(self) -> str:
        return self._model_name

    async def health(self) -> None:
        """Lazy-load the model. Called by `code-rag doctor` and the MCP
        bootstrap so failures surface early."""
        await self._ensure_model()

    async def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is not None:
                return self._model
            # Run the heavy load on a thread so we don't block the event loop.
            self._model = await asyncio.to_thread(self._load_model)
            log.info(
                "rerank.cross_encoder.loaded",
                model=self._model_name,
                device=self._device or "auto",
            )
            return self._model

    def _load_model(self) -> Any:
        # Imported here so the module-level import stays cheap.
        from sentence_transformers import CrossEncoder
        return CrossEncoder(self._model_name, device=self._device)

    async def rerank(
        self, query: str, hits: Sequence[SearchHit], top_k: int,
    ) -> list[SearchHit]:
        if not hits or top_k <= 0:
            return []
        if len(hits) == 1:
            return [self._stamp(hits[0], 0, score=1.0)]

        window = list(hits[: self._max_candidates])

        try:
            model = await self._ensure_model()
        except Exception as e:  # pragma: no cover — defensive
            log.warning("rerank.cross_encoder.load_fail",
                        err=f"{type(e).__name__}: {e}")
            return [self._stamp(h, i, score=float(h.score), fallback=True)
                    for i, h in enumerate(window[:top_k])]

        # Build (query, doc) pairs. Truncate doc text for predictable latency.
        pairs: list[tuple[str, str]] = []
        for h in window:
            text = (h.chunk.text or "").replace("\r", " ").strip()
            text = text[: self._max_chars]
            # Prefix with path::symbol so the cross-encoder has the same
            # context the chat reranker sees.
            sym = h.chunk.symbol or ""
            doc = f"{h.chunk.path} :: {sym}\n{text}" if sym else \
                  f"{h.chunk.path}\n{text}"
            pairs.append((query, doc))

        try:
            scores = await asyncio.to_thread(self._predict_scores, model, pairs)
        except Exception as e:  # pragma: no cover — defensive
            log.warning("rerank.cross_encoder.predict_fail",
                        err=f"{type(e).__name__}: {e}")
            return [self._stamp(h, i, score=float(h.score), fallback=True)
                    for i, h in enumerate(window[:top_k])]

        # Sort by score desc, take top_k. Stable on ties via input rank.
        ranked = sorted(
            range(len(window)),
            key=lambda i: (-float(scores[i]), i),
        )
        out: list[SearchHit] = []
        for new_rank, idx in enumerate(ranked[:top_k]):
            out.append(self._stamp(
                window[idx], new_rank, score=float(scores[idx]),
            ))
        return out

    @staticmethod
    def _predict_scores(model: Any, pairs: list[tuple[str, str]]) -> list[float]:
        # `predict` returns a numpy array; cast to plain floats for safety.
        arr = model.predict(pairs, show_progress_bar=False)
        try:
            return [float(x) for x in arr]
        except TypeError:
            # Some models return a 2D array (logits over labels). Take col 0
            # which is the relevance score for the standard cross-encoder API.
            return [float(x[0]) for x in arr]

    @staticmethod
    def _stamp(
        h: SearchHit, new_rank: int, *, score: float, fallback: bool = False,
    ) -> SearchHit:
        prev = h.match_reason or ""
        tag = (f"rerank cross-encoder (fallback) r{new_rank + 1}" if fallback
               else f"rerank cross-encoder {score:.3f} r{new_rank + 1}")
        reason = f"{prev} | {tag}" if prev else tag
        return SearchHit(
            chunk=h.chunk, score=score, source="rerank", match_reason=reason,
        )

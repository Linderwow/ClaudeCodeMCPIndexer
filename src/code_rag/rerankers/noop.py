from __future__ import annotations

from collections.abc import Sequence

from code_rag.interfaces.reranker import Reranker
from code_rag.models import SearchHit


class NoopReranker(Reranker):
    """Identity reranker — preserves input order, truncates to top_k.

    Used in tests and as the fallback when no reranker backend is available.
    """

    @property
    def model(self) -> str:
        return "noop"

    async def rerank(
        self, query: str, hits: Sequence[SearchHit], top_k: int,
    ) -> list[SearchHit]:
        # Preserve earlier match_reason values. No rerank score to add.
        return [h.model_copy() for h in hits[:top_k]]

    async def health(self) -> None:
        return None

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from code_rag.models import SearchHit


class Reranker(ABC):
    """Cross-encoder style reranker. Takes a query + candidate hits, returns reordered top-k."""

    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    async def rerank(self, query: str, hits: Sequence[SearchHit], top_k: int) -> list[SearchHit]:
        """Reorder and truncate. Scores in returned hits are rerank scores, source='rerank'."""

    @abstractmethod
    async def health(self) -> None: ...

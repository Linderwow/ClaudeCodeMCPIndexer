from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from code_rag.models import Chunk, SearchHit


class LexicalStore(ABC):
    """BM25/FTS5-backed exact-identifier recall.

    Complements the vector store for queries like "MNQAlpha_V91" or
    "OnBarUpdate" where literal tokens matter more than semantics.
    """

    @abstractmethod
    def open(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk]) -> None: ...

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> None: ...

    @abstractmethod
    def delete_by_path(self, path: str) -> int: ...

    @abstractmethod
    def query(self, text: str, k: int) -> list[SearchHit]:
        """Top-k by BM25. source='lexical'."""

    @abstractmethod
    def count(self) -> int: ...

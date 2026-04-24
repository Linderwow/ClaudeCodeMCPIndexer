from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from code_rag.models import Chunk, IndexMeta, SearchHit


class VectorStore(ABC):
    """Persistent vector index keyed by Chunk.id.

    Implementations MUST be idempotent on upsert by id and MUST atomically
    replace the vector if the same id is upserted with a new embedding.
    """

    @abstractmethod
    def open(self, meta: IndexMeta) -> None:
        """Open or create the store. Validates meta against on-disk stamp; raises on mismatch."""

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def upsert(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None: ...

    @abstractmethod
    def delete(self, ids: Sequence[str]) -> None: ...

    @abstractmethod
    def delete_by_path(self, path: str) -> int:
        """Remove every chunk whose path == given path. Returns count deleted."""

    @abstractmethod
    def query(
        self,
        embedding: Sequence[float],
        k: int,
        where: dict[str, object] | None = None,
    ) -> list[SearchHit]:
        """Top-k nearest neighbors. where is a metadata filter (path_glob, lang, etc.)."""

    @abstractmethod
    def count(self) -> int: ...

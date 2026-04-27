from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence


class Embedder(ABC):
    """Produces dense vector embeddings for strings.

    Must be deterministic for a given (model, text). The concrete impl is
    pinned at index time; querying with a different model/dim is refused
    (see IndexMeta in models.py).
    """

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier stamped into index metadata."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Vector dimension. May be probed lazily on first call."""

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Batch embed. Order preserved; returns len(texts) vectors of length self.dim."""

    @abstractmethod
    async def health(self) -> None:
        """Raise if backend is unreachable or model unavailable. Cheap; called on startup."""

    async def aclose(self) -> None:
        """Release any backend resources (HTTP clients, GPU model handles, etc.).

        Default: no-op. Concrete backends may override. Callers can invoke
        unconditionally; the default makes adding new backends safe without
        having to update every cleanup site.
        """
        return None

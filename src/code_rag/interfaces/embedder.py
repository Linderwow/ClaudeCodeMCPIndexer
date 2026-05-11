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
        """Batch embed for DOCUMENTS (raw, no prefix). Order preserved;
        returns len(texts) vectors of length self.dim.

        Document semantics matter for asymmetric retrieval models like
        Qwen3-Embedding -- the indexer pipeline calls this, the chunks
        land raw, and they should STAY raw. Use `embed_query` for
        anything embedded for SEARCH.
        """

    async def embed_query(
        self,
        texts: Sequence[str],
        *,
        instruct: str | None = None,
    ) -> list[list[float]]:
        """Batch embed for QUERIES (Qwen3-Embedding asymmetric retrieval).

        Phase 60-R: Qwen3-Embedding is asymmetric. Queries are wrapped
        with `Instruct: {task}\\nQuery: {text}` before embedding;
        documents stay raw. This default implementation formats with
        `format_query` then defers to `embed()` -- backends that don't
        care about asymmetry (FakeEmbedder for tests, dense BERT-style
        models without prompt training) inherit it transparently.

        Args:
            texts: one or more query strings (e.g. original query +
                rewriter variants + HyDE arms). All receive the same
                `instruct`.
            instruct: optional task-specific prefix; falls back to
                `QUERY_INSTRUCT_DEFAULT`. Phase 2 uses this to route
                per query type (code / config / docs / error / concept).
        """
        if not texts:
            return []
        from code_rag.embedders.prompts import format_query
        formatted = [format_query(t, instruct=instruct) for t in texts]
        return await self.embed(formatted)

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

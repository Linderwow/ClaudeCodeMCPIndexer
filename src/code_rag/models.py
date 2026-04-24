from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class ChunkKind(StrEnum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    STRUCT = "struct"
    INTERFACE = "interface"
    ENUM = "enum"
    NAMESPACE = "namespace"
    MODULE = "module"
    DOC = "doc"
    OTHER = "other"


class Chunk(BaseModel):
    """A unit of indexed content.

    id is content-addressed (blake3 over repo|path|symbol|text), so re-indexing
    identical content is a no-op. Changing any of those fields produces a new id,
    and the stale chunk is garbage-collected by the delete-then-insert pass.
    """

    id: str
    repo: str
    path: str
    language: str
    symbol: str | None
    kind: ChunkKind
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    text: str
    # Convenience cache; recomputed on load if missing.
    n_chars: int = 0

    def model_post_init(self, _: object) -> None:
        if self.n_chars == 0:
            object.__setattr__(self, "n_chars", len(self.text))


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    source: Literal["vector", "lexical", "hybrid", "rerank"] = "hybrid"
    # Human-readable explanation of why this hit was returned. Each stage
    # (lexical, vector, fusion, rerank) stamps a phrase; the final value is a
    # breadcrumb of the stages that touched this chunk.
    #   "lexical bm25 matched tokens ['MNQAlpha','V91']"
    #   "vector cosine 0.84"
    #   "hybrid rrf from [vector r2, lexical r1]"
    #   "rerank cross-encoder 0.91"
    match_reason: str | None = None


class IndexMeta(BaseModel):
    """Persisted alongside the index. Mismatch on open = refuse to query."""

    schema_version: int
    embedder_kind: str
    embedder_model: str
    embedder_dim: int
    created_at: str
    updated_at: str

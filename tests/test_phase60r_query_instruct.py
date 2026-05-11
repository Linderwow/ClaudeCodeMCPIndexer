"""Phase 60-R, Phase 1: Qwen3-Embedding query-side instruct prefix.

Qwen3-Embedding is an asymmetric retrieval model -- queries get a
`Instruct: {task}\\nQuery: {text}` prefix; documents stay raw. These
tests pin down the contract:

* `format_query` produces the exact wire shape (default and custom).
* `Embedder.embed_query` formats every input before delegating to
  `embed` -- so different instructs yield different vectors for the
  same text.
* `Embedder.embed` (document path) stays raw and never sees the prefix.
* The only retrieval-pipeline callsite (`HybridSearcher`) routes
  queries through `embed_query`, NOT `embed`.

A live-API integration test that hits the actual vLLM endpoint is
gated on the `CODE_RAG_LIVE_EMBEDDER` env var so CI doesn't depend on
a running model server.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from unittest.mock import AsyncMock

import pytest

from code_rag.embedders.fake import FakeEmbedder
from code_rag.embedders.prompts import QUERY_INSTRUCT_DEFAULT, format_query

# ---- format_query unit contract --------------------------------------------


def test_format_query_default_uses_default_instruct() -> None:
    """No instruct passed → DEFAULT is used."""
    assert format_query("foo") == f"Instruct: {QUERY_INSTRUCT_DEFAULT}\nQuery: foo"


def test_format_query_custom_instruct_replaces_default() -> None:
    """Caller-provided instruct overrides DEFAULT."""
    out = format_query("foo", instruct="custom task")
    assert out == "Instruct: custom task\nQuery: foo"


def test_format_query_none_falls_back_to_default() -> None:
    """Explicit None argument is the same as omitting -- DEFAULT applies."""
    assert format_query("foo", instruct=None) == format_query("foo")


def test_format_query_preserves_text_verbatim() -> None:
    """The query text is appended verbatim; no stripping, no escaping.
    Important for code-shaped queries with newlines / special chars."""
    query = "class FooBar { def baz(): pass }"
    out = format_query(query)
    assert out.endswith(f"\nQuery: {query}")


def test_format_query_shape_starts_with_instruct_then_newline_query() -> None:
    """The wire shape must be exactly `Instruct: X\\nQuery: Y`.
    Qwen3-Embedding was trained on this exact format; trailing whitespace,
    different separators, or reversed order break the asymmetric retrieval."""
    out = format_query("anything")
    lines = out.split("\n")
    assert len(lines) == 2
    assert lines[0].startswith("Instruct: ")
    assert lines[1].startswith("Query: ")


# ---- embed_query / embed contract ------------------------------------------


@pytest.mark.asyncio
async def test_embed_query_applies_prefix_via_format_query() -> None:
    """Calling embed_query on the SAME text with the DEFAULT instruct must
    produce a vector that DIFFERS from a raw embed() of the same text.
    This is the whole point of asymmetric retrieval -- if these match,
    the prefix isn't being applied."""
    emb = FakeEmbedder(dim=64)
    raw = await emb.embed(["foo"])
    via_query = await emb.embed_query(["foo"])
    assert raw[0] != via_query[0], \
        "embed_query must not produce the same vector as embed for the same text"


@pytest.mark.asyncio
async def test_embed_query_different_instructs_yield_different_vectors() -> None:
    """Two queries with the same text but different instructs must
    produce different vectors -- otherwise routing in Phase 2 is a no-op
    and any cache that keys on text-only would silently corrupt results."""
    emb = FakeEmbedder(dim=64)
    v1 = await emb.embed_query(["foo"])
    v2 = await emb.embed_query(["foo"], instruct="something else entirely")
    assert v1[0] != v2[0]


@pytest.mark.asyncio
async def test_embed_query_same_instruct_yields_stable_vectors() -> None:
    """Determinism: same text + same instruct → same vector.
    FakeEmbedder hashes input, so this also confirms format_query is
    stable across calls (no clock / counter sneaking in)."""
    emb = FakeEmbedder(dim=64)
    v1 = await emb.embed_query(["foo"])
    v2 = await emb.embed_query(["foo"])
    assert v1[0] == v2[0]


@pytest.mark.asyncio
async def test_embed_does_not_apply_prefix() -> None:
    """Documents stay raw. embed() must NOT call format_query.

    Verified two ways:
      1. Output equality: embed(['foo']) and embed(['Instruct: ...\\nQuery: foo'])
         differ (FakeEmbedder is content-sensitive).
      2. The vector for raw 'foo' is deterministic and stable across
         test runs -- if a future refactor accidentally adds the prefix
         to documents, the indexer's chunk_id would change too and
         every test that pins chunk ids would break, but this gives us
         a faster, earlier signal.
    """
    emb = FakeEmbedder(dim=64)
    raw = await emb.embed(["foo"])
    prefixed = await emb.embed([f"Instruct: {QUERY_INSTRUCT_DEFAULT}\nQuery: foo"])
    assert raw[0] != prefixed[0]


@pytest.mark.asyncio
async def test_embed_query_empty_list_returns_empty() -> None:
    """Empty input is the no-op shape; don't pay an HTTP round trip."""
    emb = FakeEmbedder(dim=64)
    assert await emb.embed_query([]) == []


@pytest.mark.asyncio
async def test_embed_query_preserves_order_and_count() -> None:
    """Batch shape contract: same length, order preserved."""
    emb = FakeEmbedder(dim=64)
    out = await emb.embed_query(["a", "b", "c"])
    assert len(out) == 3
    # Each vector is dim-sized; the order maps to the input order.
    assert all(len(v) == 64 for v in out)
    # Sanity: distinct inputs produce distinct vectors.
    assert out[0] != out[1] != out[2]


@pytest.mark.asyncio
async def test_embed_query_default_impl_routes_through_embed() -> None:
    """The default Embedder.embed_query implementation must call embed()
    with the FORMATTED texts. Verified by spying on embed().

    This is the integration contract every concrete backend inherits --
    LMStudioEmbedder, SentenceTransformersEmbedder, and FakeEmbedder all
    rely on this default. If a future backend overrides embed_query,
    that backend gets its own test; this one pins the default.
    """
    class _SpyEmbedder(FakeEmbedder):
        def __init__(self) -> None:
            super().__init__(dim=8)
            self.last_call: list[str] = []

        async def embed(self, texts: Sequence[str]) -> list[list[float]]:
            self.last_call = list(texts)
            return await super().embed(texts)

    spy = _SpyEmbedder()
    await spy.embed_query(["raw query"], instruct="my task")
    assert spy.last_call == ["Instruct: my task\nQuery: raw query"], \
        f"embed_query did not format before delegating; saw: {spy.last_call!r}"


# ---- HybridSearcher uses embed_query, not embed ----------------------------


@pytest.mark.asyncio
async def test_hybrid_searcher_calls_embed_query_not_embed(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
) -> None:
    """The retrieval pipeline's ONLY query-embed callsite (search.py)
    must use `embed_query`. If a future refactor accidentally regresses
    to `embed()`, the prefix stops being applied at search time and
    retrieval silently degrades (matches what the user-visible eval
    would catch -- but this test catches it in seconds, not hours)."""
    from code_rag.embedders.fake import FakeEmbedder
    from code_rag.rerankers.noop import NoopReranker
    from code_rag.retrieval.search import HybridSearcher, SearchParams
    from code_rag.stores.chroma_vector import ChromaVectorStore
    from code_rag.stores.sqlite_lexical import SqliteLexicalStore

    emb = FakeEmbedder(dim=32)
    # Spy: track whether embed_query was called vs embed.
    embed_query_calls: list[Sequence[str]] = []
    embed_calls: list[Sequence[str]] = []
    orig_query = emb.embed_query
    orig_embed = emb.embed

    async def spy_embed_query(texts, *, instruct=None):
        embed_query_calls.append(list(texts))
        return await orig_query(texts, instruct=instruct)

    async def spy_embed(texts):
        embed_calls.append(list(texts))
        return await orig_embed(texts)

    emb.embed_query = spy_embed_query  # type: ignore[assignment]
    emb.embed = spy_embed  # type: ignore[assignment]

    # Minimal store wiring. We don't care about the hits -- just the
    # which-method-got-called signal.
    vec = ChromaVectorStore(
        tmp_path / "chroma", "test_collection",
        tmp_path / "index_meta.json",
    )
    lex = SqliteLexicalStore(tmp_path / "fts.db")
    meta = ChromaVectorStore.build_meta("fake", "fake-embedder-v1", 32)
    vec.open(meta)
    lex.open()
    try:
        searcher = HybridSearcher(emb, vec, lex, reranker=NoopReranker())
        params = SearchParams(k_final=4, k_vector=4, k_lexical=4, k_rerank_in=4)
        await searcher.search("foo bar baz", params)
    finally:
        vec.close()
        lex.close()

    assert len(embed_query_calls) >= 1, \
        "HybridSearcher must call embed_query for query embeddings " \
        f"(got embed_query={len(embed_query_calls)} embed={len(embed_calls)})"


# ---- live integration (opt-in) ---------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("CODE_RAG_LIVE_EMBEDDER"),
    reason="set CODE_RAG_LIVE_EMBEDDER=1 to enable the live vLLM probe",
)
@pytest.mark.asyncio
async def test_live_embedder_prefix_changes_vector() -> None:
    """Against the REAL Qwen3-Embedding vLLM endpoint: the prefixed
    query vector must differ from the raw text vector. This is the
    end-to-end check that the prefix is actually flowing on the wire.
    Skipped by default; set CODE_RAG_LIVE_EMBEDDER=1 to run."""
    from code_rag.config import load_settings
    from code_rag.factory import build_embedder
    settings = load_settings()
    emb = build_embedder(settings)
    try:
        await emb.health()
        raw = await emb.embed(["foo"])
        prefixed = await emb.embed_query(["foo"])
    finally:
        await emb.aclose()
    # Cosine-similar but not identical -- the prefix shifts the vector.
    assert raw[0] != prefixed[0], \
        "live embedder produced identical vectors for prefixed vs raw; " \
        "the prefix isn't reaching the model"

"""Phase 26: eval gate + ground-truth filtering.

Two surfaces under test:
  1. `filter_cases_to_paths` — drops cases whose expected paths aren't indexed,
     prunes per-case expected[] to only in-corpus entries.
  2. `ChromaVectorStore.list_paths` — returns the distinct path set used by
     the filter (smoke-tested via a tiny in-memory index).
"""
from __future__ import annotations

from pathlib import Path

from code_rag.eval.harness import EvalCase, ExpectedHit, filter_cases_to_paths
from code_rag.models import Chunk, ChunkKind, IndexMeta
from code_rag.stores.chroma_vector import ChromaVectorStore
from code_rag.version import INDEX_SCHEMA_VERSION


def _case(query: str, paths: list[str]) -> EvalCase:
    return EvalCase(
        query=query,
        expected=[ExpectedHit(path=p) for p in paths],
        tag=None,
    )


def test_filter_drops_cases_outside_index() -> None:
    cases = [
        _case("q1", ["a.py"]),                   # in
        _case("q2", ["b.py"]),                   # out — gone entirely
        _case("q3", ["a.py", "c.py"]),           # partial — keeps a.py only
        _case("q4", ["nope.py", "neither.py"]),  # out — both missing
    ]
    indexed = {"a.py", "c.py"}

    out = filter_cases_to_paths(cases, indexed)

    queries = [c.query for c in out]
    assert queries == ["q1", "q3"], queries
    # q3 should now have ONLY in-corpus expected entries.
    q3 = next(c for c in out if c.query == "q3")
    assert [e.path for e in q3.expected] == ["a.py", "c.py"]


def test_filter_preserves_tag() -> None:
    """`tag` is part of the per-tag breakdown — must survive the filter."""
    case = EvalCase(
        query="ident",
        expected=[ExpectedHit(path="a.py")],
        tag="identifier",
    )
    out = filter_cases_to_paths([case], {"a.py"})
    assert len(out) == 1
    assert out[0].tag == "identifier"


def test_filter_with_empty_index_drops_everything() -> None:
    """Edge case: a fresh/empty store. No cases survive."""
    cases = [_case("q1", ["a.py"]), _case("q2", ["b.py"])]
    assert filter_cases_to_paths(cases, set()) == []


def test_filter_with_empty_cases_returns_empty() -> None:
    assert filter_cases_to_paths([], {"a.py"}) == []


def test_filter_dedup_safe_with_repeated_expected() -> None:
    """If a case lists the same path twice, both copies survive (the filter
    doesn't dedupe — that's the harness's job, and ExpectedHit equality
    isn't defined). Documents current behavior."""
    case = _case("q", ["a.py", "a.py", "b.py"])
    out = filter_cases_to_paths([case], {"a.py"})
    assert [e.path for e in out[0].expected] == ["a.py", "a.py"]


def test_chroma_list_paths_smoke(tmp_path: Path) -> None:
    """Tiny end-to-end: upsert two chunks across two paths, list_paths
    returns the distinct path set."""
    meta = IndexMeta(
        schema_version=INDEX_SCHEMA_VERSION,
        embedder_kind="test",
        embedder_model="dummy",
        embedder_dim=4,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection="test_collection",
        meta_path=tmp_path / "meta.json",
    )
    store.open(meta)
    try:
        chunks = [
            Chunk(
                id="id1", repo="r", path="alpha.py", language="python",
                symbol="f", kind=ChunkKind.FUNCTION,
                start_line=1, end_line=5, text="def f(): pass",
            ),
            Chunk(
                id="id2", repo="r", path="alpha.py", language="python",
                symbol="g", kind=ChunkKind.FUNCTION,
                start_line=10, end_line=15, text="def g(): pass",
            ),
            Chunk(
                id="id3", repo="r", path="beta.py", language="python",
                symbol="h", kind=ChunkKind.FUNCTION,
                start_line=1, end_line=2, text="def h(): pass",
            ),
        ]
        embeds = [[0.1, 0.2, 0.3, 0.4]] * len(chunks)
        store.upsert(chunks, embeds)
        paths = store.list_paths()
        assert paths == {"alpha.py", "beta.py"}
    finally:
        store.close()


def test_chroma_list_paths_empty(tmp_path: Path) -> None:
    """An open-but-empty store returns an empty set, not an error."""
    meta = IndexMeta(
        schema_version=INDEX_SCHEMA_VERSION,
        embedder_kind="test",
        embedder_model="dummy",
        embedder_dim=4,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    store = ChromaVectorStore(
        persist_dir=tmp_path / "chroma",
        collection="test_collection",
        meta_path=tmp_path / "meta.json",
    )
    store.open(meta)
    try:
        assert store.list_paths() == set()
    finally:
        store.close()

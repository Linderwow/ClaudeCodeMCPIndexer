"""Phase 7: eval harness computes Recall@k and MRR correctly."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from code_rag.config import load_settings
from code_rag.embedders.fake import FakeEmbedder
from code_rag.eval.harness import EvalCase, ExpectedHit, load_cases, run_eval
from code_rag.indexing.indexer import Indexer
from code_rag.models import Chunk, ChunkKind, SearchHit
from code_rag.rerankers.noop import NoopReranker
from code_rag.retrieval.search import HybridSearcher
from code_rag.stores.chroma_vector import ChromaVectorStore
from code_rag.stores.sqlite_lexical import SqliteLexicalStore

# ---- unit: metrics ---------------------------------------------------------


class _StubSearcher:
    """Returns canned results keyed by query — lets us drive the harness
    with deterministic hit lists."""

    def __init__(self, results: dict[str, list[SearchHit]]):
        self._r = results

    async def search(self, query: str, params):
        _ = params
        return list(self._r.get(query, []))


def _hit(path: str, symbol: str | None = None) -> SearchHit:
    return SearchHit(
        chunk=Chunk(id=path + (symbol or ""), repo="r", path=path, language="python",
                    symbol=symbol, kind=ChunkKind.FUNCTION,
                    start_line=1, end_line=1, text=""),
        score=0.0, source="hybrid",
    )


def test_recall_and_mrr_math() -> None:
    # Case A: target at rank 1
    # Case B: target at rank 3
    # Case C: total miss
    stub = _StubSearcher({
        "A": [_hit("foo.py"), _hit("bar.py")],
        "B": [_hit("bar.py"), _hit("baz.py"), _hit("foo.py")],
        "C": [_hit("nope.py")],
    })
    cases = [
        EvalCase("A", [ExpectedHit("foo.py")]),
        EvalCase("B", [ExpectedHit("foo.py")]),
        EvalCase("C", [ExpectedHit("foo.py")]),
    ]
    report = asyncio.run(run_eval(cases, stub, top_k=10))  # type: ignore[arg-type]
    assert report.recall_at_1  == pytest.approx(1 / 3)
    assert report.recall_at_3  == pytest.approx(2 / 3)
    assert report.recall_at_10 == pytest.approx(2 / 3)
    # MRR = (1/1 + 1/3 + 0) / 3
    assert report.mrr == pytest.approx((1.0 + 1/3) / 3)


def test_expected_symbol_filter_is_strict() -> None:
    stub = _StubSearcher({
        "Q": [_hit("a.py", "wrong_sym"), _hit("a.py", "right_sym")],
    })
    cases = [EvalCase("Q", [ExpectedHit("a.py", symbol="right_sym")])]
    report = asyncio.run(run_eval(cases, stub, top_k=10))  # type: ignore[arg-type]
    assert report.cases[0].rank == 2, "path-only would be rank 1; symbol-strict picks rank 2"


def test_load_cases_roundtrip(tmp_path: Path) -> None:
    fixture = tmp_path / "f.json"
    fixture.write_text(json.dumps([
        {"query": "foo", "expected": [{"path": "a.py", "symbol": "f"}]},
        {"query": "bar", "expected": [{"path": "b.py"}]},
        {"query": "baz", "expected": [{"path": "c.py"}], "tag": "natural_language"},
    ]), encoding="utf-8")
    cases = load_cases(fixture)
    assert len(cases) == 3
    assert cases[0].expected[0].symbol == "f"
    assert cases[1].expected[0].symbol is None
    assert cases[2].tag == "natural_language"


def test_ndcg_at_10_math() -> None:
    # rank 1 → 1.0; rank 2 → 1/log2(3) ≈ 0.6309; rank 11 → 0.
    import math
    stub = _StubSearcher({
        "A": [_hit("foo.py")],                                 # rank 1
        "B": [_hit("bar.py"), _hit("foo.py")],                  # rank 2
        "C": [_hit(f"x{i}.py") for i in range(15)] + [_hit("foo.py")],  # rank 16, beyond 10
    })
    cases = [
        EvalCase("A", [ExpectedHit("foo.py")]),
        EvalCase("B", [ExpectedHit("foo.py")]),
        EvalCase("C", [ExpectedHit("foo.py")]),
    ]
    report = asyncio.run(run_eval(cases, stub, top_k=20))  # type: ignore[arg-type]
    expected = (1.0 + (1.0 / math.log2(3)) + 0.0) / 3
    assert report.ndcg_at_10 == pytest.approx(expected, abs=1e-4)


def test_per_tag_breakdown_groups_correctly() -> None:
    stub = _StubSearcher({
        "ID-1":  [_hit("foo.py")],
        "ID-2":  [_hit("foo.py")],
        "NL-1":  [_hit("nope.py")],
    })
    cases = [
        EvalCase("ID-1", [ExpectedHit("foo.py")], tag="identifier"),
        EvalCase("ID-2", [ExpectedHit("foo.py")], tag="identifier"),
        EvalCase("NL-1", [ExpectedHit("foo.py")], tag="natural_language"),
    ]
    report = asyncio.run(run_eval(cases, stub, top_k=10))  # type: ignore[arg-type]
    per_tag = report.per_tag()
    assert per_tag["identifier"]["recall@1"] == 1.0
    assert per_tag["natural_language"]["recall@1"] == 0.0
    assert per_tag["identifier"]["n"] == 2
    assert per_tag["natural_language"]["n"] == 1


def test_p99_latency_handles_small_n() -> None:
    """Don't crash on a 1-case eval — common in CI smoke tests."""
    stub = _StubSearcher({"Q": [_hit("foo.py")]})
    cases = [EvalCase("Q", [ExpectedHit("foo.py")])]
    report = asyncio.run(run_eval(cases, stub, top_k=10))  # type: ignore[arg-type]
    # Whatever the timing, it should be a non-negative finite number.
    assert report.p99_latency_ms >= 0.0
    assert report.p99_latency_ms == report.p50_latency_ms  # n=1 → all percentiles equal


def test_diff_report_pp_deltas() -> None:
    from code_rag.eval.harness import EvalReport, EvalResult
    base = EvalReport(cases=[
        EvalResult("Q", rank=3, latency_ms=100.0, top_paths=["a", "b", "foo"]),
    ], label="baseline")
    cur = EvalReport(cases=[
        EvalResult("Q", rank=1, latency_ms=80.0, top_paths=["foo"]),
    ], label="after-fix")
    diff = cur.diff(base)
    # recall@1 went from 0 → 1 = +1.0 = +100 pp
    assert diff["deltas_pp"]["recall@1"] == pytest.approx(100.0, abs=0.1)
    # latency improved by 20 ms
    assert diff["deltas_ms"]["p50_latency_ms"] == pytest.approx(-20.0, abs=0.1)
    assert "after-fix" in diff["label"] and "baseline" in diff["label"]


# ---- integration: real searcher on a tiny corpus ---------------------------


def test_end_to_end_eval_against_real_searcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity: eval against FakeEmbedder + real FTS. Lexical must hit the rare
    identifier 100%, so recall@3 ≥ 1.0 for that case."""
    root = tmp_path / "repo"
    (root / "a.py").parent.mkdir(parents=True, exist_ok=True)
    (root / "a.py").write_text(
        "def MARKER_ABC_123():\n    return 1\n", encoding="utf-8",
    )
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        f"""
[paths]
roots    = ["{root.as_posix()}"]
data_dir = "{(tmp_path / 'data').as_posix()}"
log_dir  = "{(tmp_path / 'logs').as_posix()}"

[embedder]
kind     = "lm_studio"
base_url = "http://localhost:1234/v1"
model    = "qwen3-embedding-4b"

[reranker]
kind     = "lm_studio"
base_url = "http://localhost:1234/v1"
model    = "qwen3-reranker-4b"

[chunker]
min_chars = 5
max_chars = 2400
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODE_RAG_CONFIG", str(cfg))
    settings = load_settings()
    embedder = FakeEmbedder(dim=32)
    vec = ChromaVectorStore(settings.chroma_dir, settings.vector_store.collection,
                            settings.index_meta_path)
    lex = SqliteLexicalStore(settings.fts_path)
    meta = ChromaVectorStore.build_meta("fake", embedder.model, embedder.dim)
    vec.open(meta)
    lex.open()
    try:
        indexer = Indexer(settings, embedder, vec, lexical_store=lex)
        asyncio.run(indexer.reindex_all())
        searcher = HybridSearcher(embedder, vec, lex, NoopReranker())
        cases = [EvalCase("MARKER_ABC_123", [ExpectedHit("a.py")])]
        report = asyncio.run(run_eval(cases, searcher, top_k=5))
        assert report.recall_at_3 == 1.0
        assert report.mrr > 0.5
    finally:
        vec.close()
        lex.close()

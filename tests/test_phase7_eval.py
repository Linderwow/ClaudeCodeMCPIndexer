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
    ]), encoding="utf-8")
    cases = load_cases(fixture)
    assert len(cases) == 2
    assert cases[0].expected[0].symbol == "f"
    assert cases[1].expected[0].symbol is None


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

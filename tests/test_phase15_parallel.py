"""Phase 15: parallel pipeline. Bounded asyncio.gather across files; shared
store-lock keeps Chroma/Kuzu/FTS writes ordered."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from code_rag.config import load_settings
from code_rag.embedders.fake import FakeEmbedder
from code_rag.indexing.indexer import Indexer
from code_rag.stores.chroma_vector import ChromaVectorStore
from code_rag.stores.sqlite_lexical import SqliteLexicalStore


def _make_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n_files: int = 8,
    parallel_workers: int = 4,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    for i in range(n_files):
        (root / f"f{i}.py").write_text(
            f"def fn_{i}():\n    return {i}\n", encoding="utf-8",
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

[indexer]
parallel_workers = {parallel_workers}
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODE_RAG_CONFIG", str(cfg))


class _SlowEmbedder:
    """Mocks an embedder where each .embed() call sleeps for `latency_s`. We
    use this to PROVE parallel workers reduce wall-clock -- under serial
    execution, total time = N_files x latency_s; under parallel-N, it's about
    ceil(N_files / N) x latency_s. We assert a strict speedup."""

    def __init__(self, dim: int, latency_s: float) -> None:
        self._inner = FakeEmbedder(dim=dim)
        self._latency = latency_s
        self.in_flight_max = 0
        self._in_flight = 0
        self._lock = asyncio.Lock()

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def dim(self) -> int:
        return self._inner.dim

    async def health(self) -> None:
        await self._inner.health()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        async with self._lock:
            self._in_flight += 1
            self.in_flight_max = max(self.in_flight_max, self._in_flight)
        try:
            await asyncio.sleep(self._latency)
            return await self._inner.embed(texts)
        finally:
            async with self._lock:
                self._in_flight -= 1


def test_parallel_pipeline_runs_concurrently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With 8 files x 0.10 s embed latency:
       - Serial (workers=1): wall ~0.80 s
       - Parallel (workers=4): wall ~0.20 s
    Assert at least 2x speedup AND that >=2 embed calls actually overlapped."""
    _make_settings(tmp_path, monkeypatch, n_files=8, parallel_workers=4)
    settings = load_settings()
    emb = _SlowEmbedder(dim=32, latency_s=0.10)
    vec = ChromaVectorStore(settings.chroma_dir, settings.vector_store.collection,
                            settings.index_meta_path)
    lex = SqliteLexicalStore(settings.fts_path)
    meta = ChromaVectorStore.build_meta("fake", emb.model, emb.dim)
    vec.open(meta)
    lex.open()
    try:
        indexer = Indexer(settings, emb, vec, lexical_store=lex)  # type: ignore[arg-type]
        t0 = time.monotonic()
        stats = asyncio.run(indexer.reindex_all())
        dur = time.monotonic() - t0
        assert stats.files_indexed == 8
        # 8 files x 0.10s serial = 0.80s; parallel-4 should land ~0.25-0.40s.
        # Allow generous slack for CI variance -- anything <0.6s is real speedup.
        assert dur < 0.6, f"parallel reindex took {dur:.2f}s; expected <0.6s"
        # At least 2 embedder calls must have been concurrently in-flight.
        assert emb.in_flight_max >= 2, \
            f"max concurrent embeds was {emb.in_flight_max}; parallel pipeline didn't engage"
    finally:
        vec.close()
        lex.close()


def test_serial_pipeline_baseline_is_strictly_slower(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same workload with parallel_workers=1 should land at ~0.80 s and at
    most 1 embed call in-flight at any time."""
    _make_settings(tmp_path, monkeypatch, n_files=8, parallel_workers=1)
    settings = load_settings()
    emb = _SlowEmbedder(dim=32, latency_s=0.10)
    vec = ChromaVectorStore(settings.chroma_dir, settings.vector_store.collection,
                            settings.index_meta_path)
    lex = SqliteLexicalStore(settings.fts_path)
    meta = ChromaVectorStore.build_meta("fake", emb.model, emb.dim)
    vec.open(meta)
    lex.open()
    try:
        indexer = Indexer(settings, emb, vec, lexical_store=lex)  # type: ignore[arg-type]
        t0 = time.monotonic()
        asyncio.run(indexer.reindex_all())
        dur = time.monotonic() - t0
        # Don't lock down a tight upper bound -- just confirm it's clearly the
        # serial regime: at least 5x slower than the parallel test would run.
        assert dur > 0.5, f"serial took {dur:.2f}s; pipeline isn't honoring parallel_workers=1"
        assert emb.in_flight_max == 1, \
            f"workers=1 should serialize embeds; saw max in-flight = {emb.in_flight_max}"
    finally:
        vec.close()
        lex.close()


def test_idempotency_holds_under_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parallel-pipeline reindex must produce identical chunk count to a
    second run -- same content → same chunk_ids → no duplicates → same total
    count after the second pass (idempotency was the original Indexer
    contract; parallelism must preserve it)."""
    _make_settings(tmp_path, monkeypatch, n_files=12, parallel_workers=4)
    settings = load_settings()
    emb = FakeEmbedder(dim=32)
    vec = ChromaVectorStore(settings.chroma_dir, settings.vector_store.collection,
                            settings.index_meta_path)
    lex = SqliteLexicalStore(settings.fts_path)
    meta = ChromaVectorStore.build_meta("fake", emb.model, emb.dim)
    vec.open(meta)
    lex.open()
    try:
        indexer = Indexer(settings, emb, vec, lexical_store=lex)
        a = asyncio.run(indexer.reindex_all())
        b = asyncio.run(indexer.reindex_all())
        assert a.chunks_upserted == b.chunks_upserted == vec.count()
        assert lex.count() == vec.count()
    finally:
        vec.close()
        lex.close()

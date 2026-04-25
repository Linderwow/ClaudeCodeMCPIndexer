"""Phase 14: per-file content hash registry skips parse+embed when content
hasn't changed. Saves the bulk of the indexer's wall-clock and embedder
budget on no-op reindex passes."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from code_rag.config import load_settings
from code_rag.embedders.fake import FakeEmbedder
from code_rag.indexing.file_hash import FileHashRegistry, hash_bytes
from code_rag.indexing.indexer import Indexer
from code_rag.stores.chroma_vector import ChromaVectorStore
from code_rag.stores.sqlite_lexical import SqliteLexicalStore

# ---- unit: registry --------------------------------------------------------


def test_hash_bytes_is_deterministic_and_short() -> None:
    h1 = hash_bytes(b"hello world")
    h2 = hash_bytes(b"hello world")
    assert h1 == h2
    assert len(h1) == 32  # blake3 hex truncated to 16 bytes = 32 hex chars
    assert hash_bytes(b"hello world ") != h1


def test_registry_roundtrip(tmp_path: Path) -> None:
    reg = FileHashRegistry(tmp_path / "h.db")
    reg.open()
    try:
        assert reg.count() == 0
        assert reg.get("foo.py") is None
        reg.upsert("foo.py", b"def x(): pass\n")
        assert reg.count() == 1
        # Same content → unchanged
        assert reg.is_unchanged("foo.py", b"def x(): pass\n") is True
        # Different content → changed
        assert reg.is_unchanged("foo.py", b"def x(): pass\n# edit\n") is False
        # Unknown path → changed (treat as new)
        assert reg.is_unchanged("never.py", b"") is False
    finally:
        reg.close()


def test_registry_delete(tmp_path: Path) -> None:
    reg = FileHashRegistry(tmp_path / "h.db")
    reg.open()
    try:
        reg.upsert("a.py", b"x")
        reg.upsert("b.py", b"y")
        assert reg.count() == 2
        reg.delete("a.py")
        assert reg.count() == 1
        assert reg.get("a.py") is None
        assert reg.get("b.py") == hash_bytes(b"y")
    finally:
        reg.close()


def test_registry_list_paths(tmp_path: Path) -> None:
    reg = FileHashRegistry(tmp_path / "h.db")
    reg.open()
    try:
        reg.upsert("a.py", b"x")
        reg.upsert("b.cs", b"y")
        assert reg.list_paths() == {"a.py", "b.cs"}
    finally:
        reg.close()


# ---- integration: indexer skips parse+embed on identical content ----------


def _make_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "a.py").write_text(
        "def foo():\n    return 1\n", encoding="utf-8",
    )
    (root / "b.py").write_text(
        "def bar():\n    return 2\n", encoding="utf-8",
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
min_chars = 10
max_chars = 2400
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("CODE_RAG_CONFIG", str(cfg))


class _CountingEmbedder:
    """Wraps a FakeEmbedder to count how many times .embed() actually runs.
    We use this to PROVE the indexer skipped re-embedding unchanged files."""

    def __init__(self, inner: FakeEmbedder) -> None:
        self._inner = inner
        self.embed_calls = 0
        self.embed_input_count = 0

    @property
    def model(self) -> str:
        return self._inner.model

    @property
    def dim(self) -> int:
        return self._inner.dim

    async def health(self) -> None:
        await self._inner.health()

    async def embed(self, texts):  # type: ignore[no-untyped-def]
        self.embed_calls += 1
        self.embed_input_count += len(texts)
        return await self._inner.embed(texts)

    async def aclose(self) -> None:  # pragma: no cover
        await self._inner.aclose()


def test_indexer_skips_unchanged_files_on_second_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_settings(tmp_path, monkeypatch)
    settings = load_settings()
    inner = FakeEmbedder(dim=32)
    embedder = _CountingEmbedder(inner)
    vec = ChromaVectorStore(settings.chroma_dir, settings.vector_store.collection,
                            settings.index_meta_path)
    lex = SqliteLexicalStore(settings.fts_path)
    hashes = FileHashRegistry(settings.file_hashes_path)
    meta = ChromaVectorStore.build_meta("fake", inner.model, inner.dim)
    vec.open(meta)
    lex.open()
    hashes.open()
    try:
        indexer = Indexer(
            settings, embedder, vec,  # type: ignore[arg-type]
            lexical_store=lex,
            file_hashes=hashes,
        )
        stats_a = asyncio.run(indexer.reindex_all())
        # First pass embeds both files.
        assert stats_a.files_indexed == 2
        assert stats_a.files_skipped_unchanged == 0
        first_call_count = embedder.embed_calls
        assert first_call_count >= 2  # at least one embed call per file

        # Second pass with NO file changes — should skip both.
        stats_b = asyncio.run(indexer.reindex_all())
        assert stats_b.files_skipped_unchanged == 2
        assert stats_b.files_indexed == 0  # nothing was actually re-indexed
        # Crucially: embedder was NOT called at all on the no-change pass.
        assert embedder.embed_calls == first_call_count, \
            f"embedder ran {embedder.embed_calls - first_call_count} extra times — hash skip didn't work"
    finally:
        vec.close()
        lex.close()
        hashes.close()


def test_indexer_re_embeds_when_content_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _make_settings(tmp_path, monkeypatch)
    settings = load_settings()
    inner = FakeEmbedder(dim=32)
    embedder = _CountingEmbedder(inner)
    vec = ChromaVectorStore(settings.chroma_dir, settings.vector_store.collection,
                            settings.index_meta_path)
    lex = SqliteLexicalStore(settings.fts_path)
    hashes = FileHashRegistry(settings.file_hashes_path)
    meta = ChromaVectorStore.build_meta("fake", inner.model, inner.dim)
    vec.open(meta)
    lex.open()
    hashes.open()
    try:
        indexer = Indexer(
            settings, embedder, vec,  # type: ignore[arg-type]
            lexical_store=lex,
            file_hashes=hashes,
        )
        asyncio.run(indexer.reindex_all())
        baseline_calls = embedder.embed_calls

        # Edit one file — must re-embed exactly that one.
        (tmp_path / "repo" / "a.py").write_text(
            "def foo():\n    return 999\n", encoding="utf-8",
        )
        stats = asyncio.run(indexer.reindex_all())
        assert stats.files_indexed == 1, f"only a.py changed, but {stats.files_indexed} files re-indexed"
        assert stats.files_skipped_unchanged == 1, "b.py should have been hash-skipped, was not"
        assert embedder.embed_calls > baseline_calls, "edited file did not re-embed"
    finally:
        vec.close()
        lex.close()
        hashes.close()

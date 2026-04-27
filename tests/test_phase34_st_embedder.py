"""Phase 34: sentence-transformers embedder backend.

Mirrors the test pattern of test_phase29_cross_encoder.py: we don't load
a real ST model in CI (would download hundreds of MB and need GPU).
Instead we patch the lazy `_load_model` to return a fake encoder, then
verify the wiring (interface, dim probing, batching, error path).
"""
from __future__ import annotations

import asyncio
import sys
from typing import Any

import pytest


def _has_sentence_transformers() -> bool:
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def test_construction_without_dep_raises_clear_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the import to fail and verify we get a useful error."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    from code_rag.embedders.sentence_transformers import (
        SentenceTransformersEmbedder,
    )
    with pytest.raises(ImportError) as exc:
        SentenceTransformersEmbedder(model="dummy")
    assert "sentence-transformers" in str(exc.value).lower() or \
           "sentence_transformers" in str(exc.value).lower()
    assert "pip install" in str(exc.value).lower()


@pytest.mark.skipif(
    not _has_sentence_transformers(),
    reason="sentence-transformers not installed (optional dep)",
)
def test_model_property_returns_configured_id() -> None:
    from code_rag.embedders.sentence_transformers import (
        SentenceTransformersEmbedder,
    )
    e = SentenceTransformersEmbedder(model="some-org/some-model")
    assert e.model == "some-org/some-model"


@pytest.mark.skipif(
    not _has_sentence_transformers(),
    reason="sentence-transformers not installed (optional dep)",
)
def test_dim_unset_until_health_called() -> None:
    """`dim` raises before health() runs — index meta build needs to wait
    for the probe."""
    from code_rag.embedders.sentence_transformers import (
        SentenceTransformersEmbedder,
    )
    e = SentenceTransformersEmbedder(model="some-org/some-model")
    with pytest.raises(RuntimeError):
        _ = e.dim


@pytest.mark.skipif(
    not _has_sentence_transformers(),
    reason="sentence-transformers not installed (optional dep)",
)
def test_embed_calls_encode_and_returns_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the loaded model's encode() and verify shape + ordering."""
    import numpy as np

    from code_rag.embedders.sentence_transformers import (
        SentenceTransformersEmbedder,
    )

    class FakeEncoder:
        def encode(self, texts: list[str], **kwargs: Any) -> Any:
            # Return a deterministic 4-d vector per text where the first
            # component encodes the text length (lets us verify ordering).
            return np.array([
                [float(len(t)), 0.1, 0.2, 0.3] for t in texts
            ], dtype=np.float32)

    e = SentenceTransformersEmbedder(model="dummy")

    async def _fake_load(self: Any) -> Any:  # type: ignore[no-untyped-def]
        self._st_model = FakeEncoder()
        return self._st_model

    monkeypatch.setattr(
        SentenceTransformersEmbedder, "_ensure_loaded", _fake_load,
    )

    texts = ["a", "ab", "abc"]
    out = asyncio.run(e.embed(texts))
    assert len(out) == 3
    # Shape: each row is a list of 4 floats.
    assert all(len(row) == 4 for row in out)
    # Ordering preserved — first column matches input length.
    assert out[0][0] == 1.0
    assert out[1][0] == 2.0
    assert out[2][0] == 3.0


@pytest.mark.skipif(
    not _has_sentence_transformers(),
    reason="sentence-transformers not installed (optional dep)",
)
def test_health_probes_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    """health() should embed a probe string and stamp `_dim`."""
    import numpy as np

    from code_rag.embedders.sentence_transformers import (
        SentenceTransformersEmbedder,
    )

    class FakeEncoder1024:
        def encode(self, texts: list[str], **kwargs: Any) -> Any:
            return np.zeros((len(texts), 1024), dtype=np.float32)

    e = SentenceTransformersEmbedder(model="dummy")

    async def _fake_load(self: Any) -> Any:  # type: ignore[no-untyped-def]
        self._st_model = FakeEncoder1024()
        return self._st_model

    monkeypatch.setattr(
        SentenceTransformersEmbedder, "_ensure_loaded", _fake_load,
    )
    asyncio.run(e.health())
    assert e.dim == 1024


@pytest.mark.skipif(
    not _has_sentence_transformers(),
    reason="sentence-transformers not installed (optional dep)",
)
def test_embed_empty_returns_empty() -> None:
    from code_rag.embedders.sentence_transformers import (
        SentenceTransformersEmbedder,
    )
    e = SentenceTransformersEmbedder(model="dummy")
    assert asyncio.run(e.embed([])) == []


@pytest.mark.skipif(
    not _has_sentence_transformers(),
    reason="sentence-transformers not installed (optional dep)",
)
def test_embed_propagates_encode_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """If sentence-transformers raises (e.g. CUDA OOM, model load fail),
    we propagate — there's no fallback embedder, and silent failures
    here would corrupt the index."""
    from code_rag.embedders.sentence_transformers import (
        SentenceTransformersEmbedder,
    )

    class BrokenEncoder:
        def encode(self, texts: list[str], **kwargs: Any) -> Any:
            raise RuntimeError("simulated CUDA OOM")

    e = SentenceTransformersEmbedder(model="dummy")

    async def _fake_load(self: Any) -> Any:  # type: ignore[no-untyped-def]
        self._st_model = BrokenEncoder()
        return self._st_model

    monkeypatch.setattr(
        SentenceTransformersEmbedder, "_ensure_loaded", _fake_load,
    )
    with pytest.raises(RuntimeError, match="simulated CUDA OOM"):
        asyncio.run(e.embed(["x"]))


def test_factory_wires_kind_sentence_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `factory.build_embedder` routes kind='sentence_transformers'
    to the new backend (without actually importing sentence-transformers
    or loading any model)."""
    if not _has_sentence_transformers():
        pytest.skip("sentence-transformers not installed (optional dep)")

    from code_rag.config import EmbedderConfig
    from code_rag.embedders.sentence_transformers import (
        SentenceTransformersEmbedder,
    )
    from code_rag.factory import build_embedder

    # Build a minimal Settings with an ST embedder config.
    class _S:
        embedder = EmbedderConfig(
            kind="sentence_transformers",
            base_url="",
            model="some-org/some-model",
            dim=0,
            batch=16,
            normalize=True,
            trust_remote_code=False,
        )

    e = build_embedder(_S())  # type: ignore[arg-type]
    assert isinstance(e, SentenceTransformersEmbedder)
    assert e.model == "some-org/some-model"

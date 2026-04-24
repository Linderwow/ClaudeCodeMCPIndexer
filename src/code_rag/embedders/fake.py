from __future__ import annotations

import hashlib
from collections.abc import Sequence

from code_rag.interfaces.embedder import Embedder


class FakeEmbedder(Embedder):
    """Deterministic, offline embedder for tests.

    Hashes input to a fixed-dim float vector in [-1, 1]. Same text -> same vector.
    Not semantically meaningful — only for plumbing tests.
    """

    def __init__(self, model: str = "fake-embedder-v1", dim: int = 64) -> None:
        self._model = model
        self._dim = dim

    @property
    def model(self) -> str:
        return self._model

    @property
    def dim(self) -> int:
        return self._dim

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            # Expand hash to dim bytes, map each byte to [-1, 1].
            needed = self._dim
            buf = bytearray()
            i = 0
            while len(buf) < needed:
                buf.extend(hashlib.sha256(h + i.to_bytes(4, "little")).digest())
                i += 1
            out.append([(b / 127.5) - 1.0 for b in buf[:needed]])
        return out

    async def health(self) -> None:
        return None

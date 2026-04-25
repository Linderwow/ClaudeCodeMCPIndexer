"""Phase 20 scaffold: ColBERT v2 late-interaction retrieval adapter.

Bi-encoder retrieval (what our current pipeline uses — single vector per
chunk) loses information on long, multi-concept queries because everything
collapses to a single embedding. ColBERT (Khattab & Zaharia, 2020) keeps
per-token vectors and scores via MaxSim at query time — typically lifts
Recall@10 by 5-10 pp on multi-clause natural-language code queries (per
the ColBERT-Code paper, https://arxiv.org/abs/2310.13683).

What this module ships TONIGHT
------------------------------
The `LateInteractionRetriever` interface + a `NullLateInteraction` no-op.
Real ColBERT integration requires `pip install colbert-ai` (~3-5 GB of
PyTorch + faiss-gpu + CUDA deps), a separate index build (an hour for
100k chunks), and a model checkpoint download. None of that should
happen unattended; user supervises after waking up.

Migration path (free, all on user's machine):

    pip install "colbert-ai[torch,faiss-gpu]"   # heavy install
    code-rag colbert-index                       # builds late-interaction index
    # toggle in config.toml:
    [retrieval]
    use_colbert = true

Then `HybridSearcher` consults the `LateInteractionRetriever` as a third
fusion arm alongside vector + lexical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from code_rag.logging import get
from code_rag.models import SearchHit

log = get(__name__)


class LateInteractionRetriever(Protocol):
    """A retriever that uses per-token vectors + MaxSim scoring instead of
    single-vector cosine. Plugs into `HybridSearcher`'s fusion stage as a
    third candidate list (alongside vector + lexical).

    Implementations:
      * `NullLateInteraction` — no-op fallback; returns empty
      * `ColBERTLateInteraction` — REAL impl (Phase 20 ship), requires
        `colbert-ai` installed
    """

    async def search(self, query: str, k: int) -> list[SearchHit]: ...
    async def close(self) -> None: ...


class NullLateInteraction:
    """Always returns empty. Used when ColBERT isn't installed / configured;
    the fusion stage handles an empty arm transparently (RRF just sees
    one fewer ranked list)."""

    async def search(self, query: str, k: int) -> list[SearchHit]:
        _ = (query, k)
        return []

    async def close(self) -> None:
        return None


@dataclass(frozen=True)
class ColBERTConfig:
    """Where to find the ColBERT index + which checkpoint to use."""
    index_dir: str
    checkpoint_id: str = "colbert-ir/colbertv2.0"
    nbits: int = 2                  # PLAID compression
    doc_maxlen: int = 300
    query_maxlen: int = 64


def colbert_not_installed_error() -> RuntimeError:
    return RuntimeError(
        "ColBERT v2 support is not installed. To enable Phase 20:\n"
        "  pip install 'colbert-ai[torch,faiss-gpu]'   # ~3-5 GB\n"
        "  code-rag colbert-index --out data/colbert    # build index\n"
        "  # then in config.toml:  [retrieval]  use_colbert = true\n"
        "Quality lift on long multi-concept queries: ~5-10 pp Recall@10."
    )


def _is_colbert_installed() -> bool:
    try:
        import colbert  # type: ignore[import-not-found]  # noqa: F401
        return True
    except ImportError:
        return False


def build_colbert_retriever(config: ColBERTConfig) -> LateInteractionRetriever:
    """Construct a real ColBERT retriever, or raise with install instructions.

    Stays a stub until the user runs `pip install colbert-ai` — implementing
    the wrapper without the lib installed would either silently no-op or
    fail unpredictably. Loud errors are correct here.
    """
    if not _is_colbert_installed():
        raise colbert_not_installed_error()
    # When the lib is installed, this branch loads the on-disk index and
    # instantiates a wrapper. Deferred until you're ready to validate the
    # heavy install path.
    raise NotImplementedError(
        "ColBERT wrapper is scaffolded but not enabled. After installing "
        "colbert-ai, replace this NotImplementedError with the real loader."
    )

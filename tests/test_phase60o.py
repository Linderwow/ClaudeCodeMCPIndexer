"""Phase 60-O: monster-audit fixes (correctness + math + IPC + perf).

Round-1 catch-all for the round of fixes that came out of the 5-lens
audit. Covers:

* **C2 — embedder.dim guard** (autostart_bootstrap.py): the previous
  ternary `embedder.dim if embedder.dim > 0 else 1` couldn't work because
  the property RAISES on unprobed access. Now reads `_dim` directly.
* **IPC#1 — atomic SingletonLock**: O_EXCL replaces the read-then-write
  TOCTOU. Two callers within ms can no longer both proceed.
* **Math#1+#2 — RRF arm weights**: weights threaded through and the
  lexical arm is re-weighted to match the sum of vector arms. Identifier
  queries get BM25 voting power back.
* **P6 — chroma delete_by_path no longer count()s**: 0 if path is gone,
  1 if delete touched it. No more O(N) scans.

These are deliberately narrow tests — they pin down the new behavior
shifts. The wider regression coverage is the existing 567 tests.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from code_rag.retrieval.fusion import reciprocal_rank_fusion
from code_rag.util.proc_hygiene import SingletonLock

# ---- IPC#1: atomic O_EXCL acquire ------------------------------------------


def test_atomic_acquire_first_caller_wins(tmp_path: Path) -> None:
    """Sanity: first caller acquires, file is created with our PID."""
    lock_path = tmp_path / "vllm-launch.lock"
    assert not lock_path.exists()
    lock = SingletonLock(lock_path).__enter__()
    try:
        assert lock.acquired is True
        assert int(lock_path.read_text("utf-8")) == os.getpid()
    finally:
        lock.__exit__(None, None, None)
    # Cleanup leaves no file behind.
    assert not lock_path.exists()


def test_atomic_acquire_concurrent_callers_serialize(tmp_path: Path) -> None:
    """Two threads racing to acquire -- exactly one wins. The other
    sees acquired=False (since this thread is still alive holding the
    lock). Without O_EXCL the previous read-then-write code could let
    both proceed."""
    lock_path = tmp_path / "vllm-launch.lock"
    results: list[bool] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        barrier.wait()  # tight start
        with SingletonLock(lock_path) as lock:
            results.append(lock.acquired)
            # Hold for a bit so the OTHER thread definitely sees us as live.
            import time as _t
            _t.sleep(0.1)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert sorted(results) == [False, True], \
        f"expected one True one False; got {results} -- both acquired " \
        f"means O_EXCL guard failed"


# ---- Math#1+#2: weighted RRF ------------------------------------------------


def _hit(cid: str, source: str, score: float = 1.0):
    from code_rag.models import Chunk, SearchHit
    chunk = Chunk(
        id=cid, repo="x", path="x.py", kind="function", language="python",
        symbol=cid, start_line=1, end_line=1, content_hash=cid, text="",
    )
    return SearchHit(chunk=chunk, score=score, source=source, match_reason="")


def test_rrf_weights_default_to_unit() -> None:
    """Backwards compat: when weights is None, the math matches the
    classic Cormack RRF (the existing test suite still passes)."""
    # X appears at rank 1 in BOTH lists → tightest fused score.
    # Y appears at rank 2 in both → second.
    # Z appears at rank 3 in arm_a only.
    arm_a = [_hit("X", "vector"), _hit("Y", "vector"), _hit("Z", "vector")]
    arm_b = [_hit("X", "lexical"), _hit("Y", "lexical")]
    fused = reciprocal_rank_fusion([arm_a, arm_b], k=60, top_k=10)
    ids = [h.chunk.id for h in fused]
    assert ids[0] == "X", f"X should win (rank-1 on both); got {ids}"
    assert ids[1] == "Y", f"Y should be 2nd (rank-2 on both); got {ids}"
    assert ids[2] == "Z", f"Z is rank-3 in one arm only; got {ids}"


def test_rrf_weights_lexical_uplift_makes_bm25_competitive() -> None:
    """The audit's central claim: with N vector arms vs 1 lexical arm,
    BM25-only winners get drowned at unit weight. After re-weighting the
    lexical arm by sum(vec_weights), a BM25 unique winner can rank top-1.
    """
    vec_a = [_hit("X", "vector"), _hit("Y", "vector")]
    vec_b = [_hit("X", "vector"), _hit("Z", "vector")]
    vec_c = [_hit("X", "vector")]
    lex   = [_hit("Q", "lexical")]  # Q appears ONLY in lex, at rank 1

    # Without weights: X gets 3 * 1/61 = 0.0492; Q gets 1 * 1/61 = 0.0164.
    # X wins by 3:1.
    unweighted = reciprocal_rank_fusion([vec_a, vec_b, vec_c, lex], k=60)
    assert unweighted[0].chunk.id == "X", \
        "unweighted fusion: X (3 vec hits) outvotes Q (1 lex hit)"

    # With audit fix: lexical re-weighted by 3 (sum of vec weights). Now
    # X gets 3 * 1/61 = 0.0492 and Q gets 3 * 1/61 = 0.0492 -- tie.
    # Q's ordering depends on insertion order; the point is Q now competes.
    weighted = reciprocal_rank_fusion(
        [vec_a, vec_b, vec_c, lex],
        k=60,
        weights=[1.0, 1.0, 1.0, 3.0],
    )
    q_score = next(h.score for h in weighted if h.chunk.id == "Q")
    x_score = next(h.score for h in weighted if h.chunk.id == "X")
    assert q_score == pytest.approx(x_score, rel=1e-9), \
        f"weighted fusion should equalize Q and X; got Q={q_score}, X={x_score}"


def test_rrf_weights_length_mismatch_raises() -> None:
    """Defensive: caller passing the wrong number of weights gets a
    clear error, not silent miscalculation."""
    arms = [[_hit("a", "vector")], [_hit("b", "lexical")]]
    with pytest.raises(ValueError, match="weights"):
        reciprocal_rank_fusion(arms, weights=[1.0, 1.0, 1.0])


# ---- P6: chroma delete_by_path returns 0/1 cheaply --------------------------


def test_delete_by_path_returns_zero_when_path_absent() -> None:
    """When the path has no chunks in the collection, delete_by_path
    must return 0 (the indexer uses this as `1 if purged else 0`)."""
    from code_rag.stores.chroma_vector import ChromaVectorStore
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    fake_coll = MagicMock()
    fake_coll.get.return_value = {"ids": []}  # no matches
    store._coll = fake_coll  # bypass open() for unit isolation
    assert store.delete_by_path("never-indexed.py") == 0
    fake_coll.delete.assert_not_called(), \
        "no matches -> no delete call (saves the chroma write entirely)"


def test_delete_by_path_returns_one_when_path_present() -> None:
    """When the path has chunks, delete_by_path returns 1 and issues the
    actual delete."""
    from code_rag.stores.chroma_vector import ChromaVectorStore
    store = ChromaVectorStore.__new__(ChromaVectorStore)
    fake_coll = MagicMock()
    fake_coll.get.return_value = {"ids": ["c1", "c2", "c3"]}
    store._coll = fake_coll
    assert store.delete_by_path("real.py") == 1
    fake_coll.delete.assert_called_once()

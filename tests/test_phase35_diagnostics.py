"""Phase 35 (A4): per-miss diagnostic tool.

When eval-gate reports R@3=85%, the question "which 15% missed and
why" is the only signal that drives further improvement. This test
verifies the diagnostic correctly classifies missed cases into the
three actionable buckets.
"""
from __future__ import annotations

from code_rag.eval.harness import (
    EvalCase,
    EvalReport,
    EvalResult,
    ExpectedHit,
    diagnose_misses_with_cases,
)


def _r(query: str, rank: int | None, top_paths: list[str], tag: str | None = None) -> EvalResult:
    return EvalResult(query=query, rank=rank, latency_ms=100, top_paths=top_paths, tag=tag)


def test_classifies_ground_truth_missing_from_index() -> None:
    """Expected path simply isn't in the indexed corpus → wrong category
    of failure (re-mine fixes, not retrieval tuning)."""
    cases = [
        EvalCase(
            query="find frobnicator",
            expected=[ExpectedHit(path="missing/file.py")],
            tag="identifier",
        ),
    ]
    report = EvalReport(cases=[
        _r("find frobnicator", rank=None,
           top_paths=["other/file.py", "another.py"]),
    ])
    indexed = {"other/file.py", "another.py"}  # missing/file.py NOT here

    cats = diagnose_misses_with_cases(cases, report, indexed)
    assert len(cats["ground_truth_not_in_index"]) == 1
    assert cats["ground_truth_not_in_index"][0]["query"] == "find frobnicator"
    assert cats["ground_truth_not_in_index"][0]["missing_paths"] == ["missing/file.py"]
    assert len(cats["retrieved_but_wrong_path"]) == 0


def test_classifies_retrieved_but_wrong_path() -> None:
    """Expected path IS in index but retriever ranked wrong chunks."""
    cases = [
        EvalCase(
            query="find frobnicator",
            expected=[ExpectedHit(path="real/answer.py")],
            tag="natural_language",
        ),
    ]
    report = EvalReport(cases=[
        _r("find frobnicator", rank=None,
           top_paths=["wrong1.py", "wrong2.py", "wrong3.py"]),
    ])
    indexed = {"real/answer.py", "wrong1.py", "wrong2.py", "wrong3.py"}

    cats = diagnose_misses_with_cases(cases, report, indexed)
    assert len(cats["retrieved_but_wrong_path"]) == 1
    assert cats["retrieved_but_wrong_path"][0]["expected_paths"] == ["real/answer.py"]
    assert len(cats["ground_truth_not_in_index"]) == 0


def test_classifies_no_results() -> None:
    """Top-10 was empty (min_score gate killed everything, etc.)."""
    cases = [
        EvalCase(
            query="weird query",
            expected=[ExpectedHit(path="any.py")],
            tag=None,
        ),
    ]
    report = EvalReport(cases=[
        _r("weird query", rank=None, top_paths=[]),
    ])
    indexed = {"any.py"}

    cats = diagnose_misses_with_cases(cases, report, indexed)
    assert len(cats["no_results"]) == 1
    # No-result misses don't go into the other buckets.
    assert len(cats["ground_truth_not_in_index"]) == 0
    assert len(cats["retrieved_but_wrong_path"]) == 0


def test_hits_dont_appear_in_any_bucket() -> None:
    """If a case actually hit the top-N, it shouldn't appear in any
    miss-bucket."""
    cases = [
        EvalCase(
            query="success case",
            expected=[ExpectedHit(path="found.py")],
            tag=None,
        ),
    ]
    report = EvalReport(cases=[
        _r("success case", rank=2, top_paths=["other.py", "found.py"]),
    ])
    cats = diagnose_misses_with_cases(cases, report, {"found.py", "other.py"})
    total = sum(len(v) for v in cats.values())
    assert total == 0


def test_mixed_scenario_full_taxonomy() -> None:
    """Realistic case with one of each miss type plus a hit; verifies
    the diagnostic doesn't bleed cases between buckets."""
    cases = [
        EvalCase(query="hit", expected=[ExpectedHit(path="found.py")], tag=None),
        EvalCase(query="missing-gt", expected=[ExpectedHit(path="ghost.py")], tag=None),
        EvalCase(query="wrong-path", expected=[ExpectedHit(path="real.py")], tag=None),
        EvalCase(query="empty", expected=[ExpectedHit(path="any.py")], tag=None),
    ]
    report = EvalReport(cases=[
        _r("hit", rank=1, top_paths=["found.py"]),
        _r("missing-gt", rank=None, top_paths=["w1.py"]),
        _r("wrong-path", rank=None, top_paths=["wrong1.py", "wrong2.py"]),
        _r("empty", rank=None, top_paths=[]),
    ])
    indexed = {"found.py", "w1.py", "real.py", "wrong1.py",
               "wrong2.py", "any.py"}
    cats = diagnose_misses_with_cases(cases, report, indexed)
    assert len(cats["ground_truth_not_in_index"]) == 1
    assert len(cats["retrieved_but_wrong_path"]) == 1
    assert len(cats["no_results"]) == 1
    queries_in_buckets = {
        e["query"] for v in cats.values() for e in v
    }
    assert "hit" not in queries_in_buckets   # hits never appear

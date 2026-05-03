"""Tests for Phase 37-C continuous telemetry."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from code_rag.eval.harness import EvalReport, EvalResult
from code_rag.eval.telemetry import (
    append_run,
    history_path,
    read_history,
    trend_summary,
)


def _report(label: str, ranks: list[int | None]) -> EvalReport:
    cases = [
        EvalResult(query=f"q{i}", rank=r, latency_ms=10.0 * (i + 1),
                   top_paths=[], tag=None)
        for i, r in enumerate(ranks)
    ]
    return EvalReport(cases=cases, label=label, index_meta=None)


def test_history_path_returns_eval_history_under_data_dir(tmp_path: Path) -> None:
    p = history_path(tmp_path)
    assert p == tmp_path / "eval" / "history.jsonl"


def test_append_run_creates_file_with_one_jsonl_row(tmp_path: Path) -> None:
    p = history_path(tmp_path)
    rep = _report("v1", [1, 2, None])
    append_run(p, rep, label="run-1")

    rows = read_history(p)
    assert len(rows) == 1
    row = rows[0]
    assert row["label"] == "run-1"
    assert row["summary"]["n"] == 3
    assert "ts" in row


def test_append_run_appends_multiple_rows(tmp_path: Path) -> None:
    p = history_path(tmp_path)
    append_run(p, _report("a", [1]),       label="a")
    append_run(p, _report("b", [1, 2]),    label="b")
    append_run(p, _report("c", [1, 2, 3]), label="c")

    rows = read_history(p)
    assert [r["label"] for r in rows] == ["a", "b", "c"]


def test_read_history_skips_corrupt_lines(tmp_path: Path) -> None:
    p = history_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Mix valid + corrupt lines. Reader should ignore the bad ones.
    p.write_text(
        json.dumps({"label": "ok-1", "summary": {}}) + "\n"
        "this is not json\n"
        "\n"  # blank line
        + json.dumps({"label": "ok-2", "summary": {}}) + "\n",
        encoding="utf-8",
    )
    rows = read_history(p)
    assert [r["label"] for r in rows] == ["ok-1", "ok-2"]


def test_read_history_tail_keeps_only_last_n(tmp_path: Path) -> None:
    p = history_path(tmp_path)
    for i in range(5):
        append_run(p, _report(f"r{i}", [1]), label=f"r{i}")
    rows = read_history(p, tail=2)
    assert [r["label"] for r in rows] == ["r3", "r4"]


def test_read_history_returns_empty_when_file_missing(tmp_path: Path) -> None:
    p = history_path(tmp_path)
    assert not p.exists()
    assert read_history(p) == []


def test_trend_summary_empty() -> None:
    assert trend_summary([]) == {"n_runs": 0}


def test_trend_summary_single_run() -> None:
    rows = [{"summary": {"recall@1": 0.5, "p50_latency_ms": 100.0}, "ts": "2025-01-01T00:00:00+00:00"}]
    out = trend_summary(rows)
    assert out["n_runs"] == 1
    # latest == oldest with one row → all deltas are 0.0
    assert all(v == 0.0 for v in out["deltas_pp"].values())


def test_trend_summary_computes_pp_and_ms_deltas() -> None:
    rows = [
        {"summary": {"recall@1": 0.50, "recall@3": 0.70, "p50_latency_ms": 100.0,
                     "p95_latency_ms": 200.0, "p99_latency_ms": 300.0,
                     "recall@10": 0.0, "mrr": 0.0, "ndcg@10": 0.0},
         "ts": "2025-01-01"},
        {"summary": {"recall@1": 0.60, "recall@3": 0.65, "p50_latency_ms": 110.0,
                     "p95_latency_ms": 220.0, "p99_latency_ms": 330.0,
                     "recall@10": 0.0, "mrr": 0.0, "ndcg@10": 0.0},
         "ts": "2025-01-02"},
    ]
    out = trend_summary(rows)
    assert out["n_runs"] == 2
    assert out["deltas_pp"]["recall@1"] == 10.0   # 0.60 - 0.50 = 10.00 pp
    assert out["deltas_pp"]["recall@3"] == -5.0   # regression
    assert out["deltas_ms"]["p50_latency_ms"] == 10.0
    assert out["deltas_ms"]["p95_latency_ms"] == 20.0
    assert out["deltas_ms"]["p99_latency_ms"] == 30.0


def test_append_run_records_index_summary(tmp_path: Path) -> None:
    p = history_path(tmp_path)
    rep = _report("v1", [1])
    append_run(p, rep, index_summary={
        "embedder": {"kind": "lm_studio", "model": "qwen", "dim": 2560},
        "reranker": {"kind": "cross_encoder", "model": "bge"},
    }, ts=datetime(2025, 1, 1, tzinfo=UTC))
    rows = read_history(p)
    assert rows[0]["index"]["embedder"]["dim"] == 2560
    assert rows[0]["ts"] == "2025-01-01T00:00:00+00:00"


def test_append_run_swallows_io_error(tmp_path: Path) -> None:
    """A flaky disk shouldn't break the eval-gate exit code."""
    # Point at a path whose parent is impossible to create on Windows: a
    # NUL-byte-containing name. mkdir() will raise OSError; append_run
    # should log + swallow.
    bad = Path(str(tmp_path / "no\x00way"))
    rep = _report("v1", [1])
    # This must not raise.
    append_run(bad, rep, label="x")

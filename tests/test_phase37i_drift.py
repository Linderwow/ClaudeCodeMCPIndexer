"""Phase 37-I: eval-gate drift detection.

Pure-function tests over `detect_drift`. The CLI command + Task Scheduler
installer are covered by the existing CLI test patterns; the detection
logic is the part that needs proof.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from code_rag.eval.telemetry import detect_drift


def _row(ts: datetime, **summary: float) -> dict[str, Any]:
    return {
        "ts": ts.isoformat(),
        "label": "scheduled",
        "summary": summary,
    }


def test_insufficient_data_no_rows() -> None:
    out = detect_drift([])
    assert out["status"] == "insufficient_data"
    assert out["regressions"] == []


def test_insufficient_data_single_row() -> None:
    out = detect_drift([_row(datetime.now(UTC), recall_at_1=0.65)])
    assert out["status"] == "insufficient_data"


def test_no_regression_returns_ok() -> None:
    """Latest run matches the median baseline → status ok, no regressions."""
    base = datetime(2026, 5, 1, tzinfo=UTC)
    rows = [
        _row(base + timedelta(days=i),
             **{"recall@1": 0.65, "recall@3": 0.80, "mrr": 0.71})
        for i in range(8)
    ]
    out = detect_drift(rows, baseline_window=7, threshold_pp=2.0)
    assert out["status"] == "ok"
    assert out["regressions"] == []
    assert out["n_rows"] == 8


def test_regression_below_threshold_still_ok() -> None:
    """A 1pp drop is below the 2pp threshold; not a regression."""
    base = datetime(2026, 5, 1, tzinfo=UTC)
    rows = [
        _row(base + timedelta(days=i), **{"recall@1": 0.65, "recall@3": 0.80})
        for i in range(7)
    ]
    rows.append(_row(base + timedelta(days=7),
                     **{"recall@1": 0.64, "recall@3": 0.79}))   # -1pp each
    out = detect_drift(rows, threshold_pp=2.0)
    assert out["status"] == "ok"


def test_regression_above_threshold_is_flagged() -> None:
    """A 5pp drop on recall@1 fires."""
    base = datetime(2026, 5, 1, tzinfo=UTC)
    rows = [
        _row(base + timedelta(days=i), **{"recall@1": 0.70, "recall@3": 0.85})
        for i in range(7)
    ]
    rows.append(_row(base + timedelta(days=7),
                     **{"recall@1": 0.65, "recall@3": 0.85}))   # -5pp r@1
    out = detect_drift(rows, threshold_pp=2.0)
    assert out["status"] == "regressed"
    assert len(out["regressions"]) == 1
    reg = out["regressions"][0]
    assert reg["metric"] == "recall@1"
    assert reg["delta_pp"] == -5.0
    assert reg["latest"] == 0.65
    assert reg["baseline"] == 0.70


def test_regression_uses_median_not_mean() -> None:
    """Median is robust to outliers. Two great-day outliers shouldn't pull
    the baseline up enough to make the latest look like a regression
    when it's actually within the baseline window's middle band."""
    base = datetime(2026, 5, 1, tzinfo=UTC)
    rows = [
        _row(base + timedelta(days=0), **{"recall@1": 0.65}),
        _row(base + timedelta(days=1), **{"recall@1": 0.66}),
        _row(base + timedelta(days=2), **{"recall@1": 0.67}),
        _row(base + timedelta(days=3), **{"recall@1": 0.95}),  # outlier
        _row(base + timedelta(days=4), **{"recall@1": 0.93}),  # outlier
        _row(base + timedelta(days=5), **{"recall@1": 0.66}),
        _row(base + timedelta(days=6), **{"recall@1": 0.67}),
        # latest matches the typical band; median of window is ~0.67
        _row(base + timedelta(days=7), **{"recall@1": 0.66}),
    ]
    out = detect_drift(rows, baseline_window=7, threshold_pp=2.0)
    # Median of [.65,.66,.67,.95,.93,.66,.67] is .67. Latest .66 is -1pp,
    # below threshold → ok. (If we had used mean, baseline would be ~.74
    # and latest would falsely flag as -8pp.)
    assert out["status"] == "ok"


def test_multiple_metric_regressions_all_listed() -> None:
    base = datetime(2026, 5, 1, tzinfo=UTC)
    rows = [
        _row(base + timedelta(days=i),
             **{"recall@1": 0.70, "recall@3": 0.85, "mrr": 0.78})
        for i in range(5)
    ]
    rows.append(_row(base + timedelta(days=5),
                     **{"recall@1": 0.62, "recall@3": 0.80, "mrr": 0.70}))
    out = detect_drift(rows, threshold_pp=2.0)
    assert out["status"] == "regressed"
    metrics = {r["metric"] for r in out["regressions"]}
    # All three drop > 2pp.
    assert metrics == {"recall@1", "recall@3", "mrr"}


def test_threshold_is_respected_per_metric() -> None:
    """A 4pp drop on r@1 + a 1pp drop on mrr → only r@1 flagged."""
    base = datetime(2026, 5, 1, tzinfo=UTC)
    rows = [
        _row(base + timedelta(days=i), **{"recall@1": 0.70, "mrr": 0.75})
        for i in range(5)
    ]
    rows.append(_row(base + timedelta(days=5),
                     **{"recall@1": 0.66, "mrr": 0.74}))
    out = detect_drift(rows, threshold_pp=3.0)
    assert out["status"] == "regressed"
    assert len(out["regressions"]) == 1
    assert out["regressions"][0]["metric"] == "recall@1"


def test_handles_missing_metric_in_baseline_gracefully() -> None:
    """If older rows lack a metric (schema evolution), we skip it instead of
    treating absence as zero (which would always flag a regression)."""
    base = datetime(2026, 5, 1, tzinfo=UTC)
    rows = [
        _row(base + timedelta(days=i), recall_at_1=0.70)   # no recall@1
        for i in range(5)
    ]
    rows.append(_row(base + timedelta(days=5), **{"recall@1": 0.69}))
    out = detect_drift(rows, threshold_pp=1.0)
    # Baseline has no `recall@1` data → nothing to compare → no regressions.
    assert out["status"] == "ok"


def test_threshold_pp_echoed_in_output() -> None:
    """Caller can verify the threshold the detector actually used (sanity
    when reading drift-state.json from the dashboard)."""
    base = datetime(2026, 5, 1, tzinfo=UTC)
    rows = [_row(base, **{"recall@1": 0.7}), _row(base, **{"recall@1": 0.7})]
    out = detect_drift(rows, threshold_pp=3.5)
    assert out["threshold_pp"] == 3.5

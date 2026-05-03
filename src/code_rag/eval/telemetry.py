"""Phase 37-C: continuous retrieval telemetry.

Why this exists
---------------
NVIDIA's RAG blueprint advertises "Telemetry and observability" + "RAGAS
evaluation scripts" as first-class features. We already have an eval-gate
(Phase 26) but it's a one-shot pass/fail check. To spot quality drift
that doesn't trip the gate (e.g. p95 latency creeping up by 30 ms a
week, or a single tag's recall slowly eroding), we need a TIME SERIES.

This module appends one row per eval-gate run to
`data/eval/history.jsonl`. The dashboard reads it back to plot trends.

Schema (one JSON object per line):

    {
      "ts":            ISO 8601 UTC timestamp,
      "label":         "eval-gate" | user-supplied label,
      "summary": {
        "n":             N cases,
        "recall@1":      0.0 .. 1.0,
        "recall@3":      0.0 .. 1.0,
        "recall@10":     0.0 .. 1.0,
        "mrr":           0.0 .. 1.0,
        "ndcg@10":       0.0 .. 1.0,
        "p50_latency_ms": float,
        "p95_latency_ms": float,
        "p99_latency_ms": float
      },
      "per_tag":       {tag: {recall@1, ...}},
      "index": {
        "embedder":  {kind, model, dim},
        "reranker":  {kind, model},
        "n_chunks":  int (best-effort),
      },
    }

Append-only by design. Compaction is the user's problem; the file is
expected to grow at one row per scheduled run, so a year of daily
gates is < 1 MB.
"""
from __future__ import annotations

import contextlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from code_rag.eval.harness import EvalReport
from code_rag.logging import get

log = get(__name__)


def history_path(data_dir: Path) -> Path:
    """Canonical location of the JSONL history file."""
    return data_dir / "eval" / "history.jsonl"


def append_run(
    history_file: Path,
    report: EvalReport,
    *,
    index_summary: dict[str, Any] | None = None,
    label: str | None = None,
    ts: datetime | None = None,
) -> None:
    """Append one row to `history_file`.

    Best-effort: any I/O error is logged but doesn't raise. We don't want
    a flaky disk to break the eval-gate exit code.

    Use `index_summary` to record the embedder/reranker/dim/n_chunks for
    later filtering ("show me only runs on bge-code-v1"). The dashboard
    can group by these fields when plotting.
    """
    history_file.parent.mkdir(parents=True, exist_ok=True)

    row: dict[str, Any] = {
        "ts": (ts or datetime.now(UTC)).isoformat(),
        "label": label or report.label or "eval-gate",
        "summary": report.summary(),
        "per_tag": report.per_tag(),
    }
    if index_summary:
        row["index"] = index_summary

    line = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
    try:
        with history_file.open("a", encoding="utf-8") as f:
            f.write(line)
            f.write("\n")
    except (OSError, ValueError) as e:
        # OSError covers disk-full / permission / ENOENT; ValueError covers
        # path-shaped surprises (e.g. embedded NUL, unwritable Windows path).
        # Either way, telemetry is best-effort — never break the gate.
        log.warning("eval_history.write_fail",
                    path=str(history_file), err=str(e))


def read_history(history_file: Path, *, tail: int | None = None) -> list[dict[str, Any]]:
    """Read history rows in chronological order.

    `tail` keeps only the last N rows. Malformed lines are skipped
    silently — a single corrupt row shouldn't break the dashboard.
    """
    if not history_file.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with history_file.open("r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                with contextlib.suppress(json.JSONDecodeError):
                    obj = json.loads(raw)
                    if isinstance(obj, dict):
                        rows.append(obj)
    except OSError as e:
        log.warning("eval_history.read_fail",
                    path=str(history_file), err=str(e))
        return []
    if tail is not None and tail > 0:
        return rows[-tail:]
    return rows


def detect_drift(
    rows: Iterable[dict[str, Any]],
    *,
    baseline_window: int = 7,
    threshold_pp: float = 2.0,
    metric_keys: tuple[str, ...] = ("recall@1", "recall@3", "recall@10", "mrr"),
) -> dict[str, Any]:
    """Phase 37-I: classify quality drift over the recent run history.

    Compares the LATEST run against the median of the previous
    `baseline_window` runs. Any headline metric that drops by more than
    `threshold_pp` percentage points fires a regression flag.

    Why median, not first-row: history.jsonl accumulates runs across
    embedder/reranker/chunker swaps. The first row may be on a totally
    different config. The recent window is the relevant baseline for
    "did anything regress this week?".

    Returns:
        {
          "status":     "ok" | "regressed" | "insufficient_data",
          "n_rows":     int,
          "latest_ts":  ISO 8601 | None,
          "regressions": [
              {"metric": "recall@1", "latest": 0.65, "baseline": 0.70,
               "delta_pp": -5.0}
          ],
          "threshold_pp": float (echoed for clarity)
        }

    The CLI (`code-rag eval-drift-check`) exits non-zero when status is
    "regressed", letting Task Scheduler treat it as a failure surface
    a Windows toast / status flag.
    """
    rows = list(rows)
    if len(rows) < 2:
        return {
            "status": "insufficient_data",
            "n_rows": len(rows),
            "latest_ts": rows[-1].get("ts") if rows else None,
            "regressions": [],
            "threshold_pp": threshold_pp,
        }

    latest = rows[-1].get("summary", {})
    if not isinstance(latest, dict):
        return {
            "status": "insufficient_data",
            "n_rows": len(rows),
            "latest_ts": rows[-1].get("ts"),
            "regressions": [],
            "threshold_pp": threshold_pp,
        }

    # Baseline = median of previous window. Skip the latest row itself.
    window = rows[-(baseline_window + 1):-1] if baseline_window > 0 else rows[:-1]
    if not window:
        return {
            "status": "insufficient_data",
            "n_rows": len(rows),
            "latest_ts": rows[-1].get("ts"),
            "regressions": [],
            "threshold_pp": threshold_pp,
        }

    def _median(vals: list[float]) -> float | None:
        # Phase 38 (audit fix): exclude bool — it's a subclass of int but
        # `True/False` slipping into a baseline median pollutes the
        # comparison (True==1, False==0 in arithmetic context).
        v = sorted(x for x in vals
                   if isinstance(x, int | float) and not isinstance(x, bool))
        if not v:
            return None
        n = len(v)
        return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2

    regressions: list[dict[str, Any]] = []
    for key in metric_keys:
        try:
            latest_val = float(latest.get(key, 0.0))
        except (TypeError, ValueError):
            continue
        baseline_vals: list[float] = []
        for r in window:
            s = r.get("summary", {})
            if isinstance(s, dict) and key in s:
                try:
                    baseline_vals.append(float(s[key]))
                except (TypeError, ValueError):
                    continue
        baseline = _median(baseline_vals)
        if baseline is None:
            continue
        delta_pp = round((latest_val - baseline) * 100, 2)
        if delta_pp < -abs(threshold_pp):
            regressions.append({
                "metric": key,
                "latest": round(latest_val, 4),
                "baseline": round(baseline, 4),
                "delta_pp": delta_pp,
            })

    return {
        "status": "regressed" if regressions else "ok",
        "n_rows": len(rows),
        "latest_ts": rows[-1].get("ts"),
        "regressions": regressions,
        "threshold_pp": threshold_pp,
    }


def trend_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Reduce a list of history rows to a compact trend summary.

    Returns `{n_runs, latest, oldest, deltas_pp}` where deltas_pp is
    the latest minus oldest in percentage points for each headline
    metric. Useful for "is quality drifting?" at a glance.
    """
    rows = list(rows)
    if not rows:
        return {"n_runs": 0}
    latest = rows[-1].get("summary", {})
    oldest = rows[0].get("summary", {})
    if not isinstance(latest, dict) or not isinstance(oldest, dict):
        return {"n_runs": len(rows)}

    def _delta_pp(key: str) -> float | None:
        try:
            l_val = float(latest.get(key, 0.0))
            o_val = float(oldest.get(key, 0.0))
        except (TypeError, ValueError):
            return None
        return round((l_val - o_val) * 100, 2)

    def _delta_ms(key: str) -> float | None:
        try:
            l_val = float(latest.get(key, 0.0))
            o_val = float(oldest.get(key, 0.0))
        except (TypeError, ValueError):
            return None
        return round(l_val - o_val, 1)

    return {
        "n_runs": len(rows),
        "first_ts": rows[0].get("ts"),
        "latest_ts": rows[-1].get("ts"),
        "latest": latest,
        "oldest": oldest,
        "deltas_pp": {
            k: _delta_pp(k)
            for k in ("recall@1", "recall@3", "recall@10", "mrr", "ndcg@10")
        },
        "deltas_ms": {
            k: _delta_ms(k)
            for k in ("p50_latency_ms", "p95_latency_ms", "p99_latency_ms")
        },
    }

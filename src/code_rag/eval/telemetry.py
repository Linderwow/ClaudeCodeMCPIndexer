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

"""Phase 47 → 52: Start All refuses when there isn't enough RAM/VRAM
to run the code-rag stack without OOM.

Phase 47 originally used a simple "> 50% VRAM = refuse" rule. Phase 52
replaced it with the budget-aware arbitrator (resource_budget.py) which
knows code-rag's actual cost (~14 GB RAM + ~12 GB VRAM) and refuses
when (current_used + cost) would exceed (total - reserve).

This file pins the INTEGRATION (the budget verdict feeds into
start_all and produces a 'budget_guard' step). The Phase 52 tests
in test_phase52_resource_budget.py pin the budget logic in isolation.
"""
from __future__ import annotations

from unittest.mock import patch

from code_rag.dashboard import operations as ops
from code_rag.dashboard import resource_budget as rb


class _StubSettings:
    """Minimal stand-in for code_rag.config.Settings — start_all only
    touches paths.data_dir, embedder.model, reranker.kind/model, and
    indirectly start_lms_server / start_watcher_task / load_model."""

    class _Paths:
        from pathlib import Path
        data_dir = Path("C:/nonexistent_test_dir")

    class _Embedder:
        model = "text-embedding-qwen3-embedding-4b"

    class _Reranker:
        kind = "cross_encoder"
        model = "BAAI/bge-reranker-v2-m3"

    paths = _Paths()
    embedder = _Embedder()
    reranker = _Reranker()


def _verdict_blocked() -> rb.BudgetVerdict:
    """Sample 'refuse' verdict — used to drive the integration test
    without standing up the full nvidia-smi + RAM probe stack."""
    return rb.BudgetVerdict(
        project_id="code-rag", ok=False,
        cost_ram_gb=14.0, cost_vram_gb=12.0,
        available_ram_gb=10.0, available_vram_gb=2.0,
        bottleneck="vram",
        suggestion=(
            "refuse start: code-rag (full stack) needs 14.0 GB RAM + 12.0 GB "
            "VRAM. only 2.0 GB VRAM available (2.0 GB reserved); "
            "need to free 10.0 GB."
        ),
    )


def _verdict_ok() -> rb.BudgetVerdict:
    """Sample 'pass' verdict for happy-path integration."""
    return rb.BudgetVerdict(
        project_id="code-rag", ok=True,
        cost_ram_gb=14.0, cost_vram_gb=12.0,
        available_ram_gb=40.0, available_vram_gb=18.0,
        bottleneck=None,
        suggestion="",
    )


def test_guard_blocks_when_budget_verdict_says_refuse() -> None:
    """The budget arbitrator says 'refuse' (e.g. ComfyUI is hogging
    VRAM). Start All must bail BEFORE clearing the stop marker, starting
    LM Studio, or starting the watcher task."""
    with patch.object(ops, "_budget_verdict_for_code_rag",
                      return_value=_verdict_blocked()), \
         patch.object(ops, "start_lms_server",
                      side_effect=AssertionError("must not start LMS when budget blocks")), \
         patch.object(ops, "start_watcher_task",
                      side_effect=AssertionError("must not start watcher when budget blocks")):
        result = ops.start_all(_StubSettings())

    assert result.ok is False
    assert len(result.steps) == 1
    guard = result.steps[0]
    assert guard.name == "budget_guard"
    assert guard.ok is False
    # Suggestion should mention the bottleneck and how much to free.
    assert "VRAM" in guard.detail
    assert "free" in guard.detail.lower()


def test_guard_passes_when_budget_verdict_says_ok() -> None:
    """Happy path: enough headroom. Guard records PASS and start_all
    proceeds through start_lms_server / load_model / start_watcher."""
    with patch.object(ops, "_budget_verdict_for_code_rag",
                      return_value=_verdict_ok()), \
         patch("code_rag.util.stop_marker.clear_intentionally_stopped",
               return_value=True), \
         patch.object(ops, "start_lms_server",
                      return_value=ops.StepResult("start_lms_server", True, "ready", 1.0)), \
         patch.object(ops, "load_model",
                      return_value=ops.StepResult("load_model(x)", True, "loaded", 1.0)), \
         patch.object(ops, "start_watcher_task",
                      return_value=ops.StepResult("start_watcher_task", True, "OK", 1.0)):
        result = ops.start_all(_StubSettings())

    # First step is the budget guard, recording PASS.
    assert result.steps[0].name == "budget_guard"
    assert result.steps[0].ok is True
    # Subsequent steps ran.
    step_names = [s.name for s in result.steps]
    assert "start_lms_server" in step_names
    assert "start_watcher_task" in step_names


def test_force_override_bypasses_guard() -> None:
    """Even when the budget would refuse, force=True skips the guard."""
    with patch.object(ops, "_budget_verdict_for_code_rag",
                      side_effect=AssertionError("guard must NOT run when force=True")), \
         patch("code_rag.util.stop_marker.clear_intentionally_stopped",
               return_value=True), \
         patch.object(ops, "start_lms_server",
                      return_value=ops.StepResult("start_lms_server", True, "ready", 1.0)), \
         patch.object(ops, "load_model",
                      return_value=ops.StepResult("load_model(x)", True, "loaded", 1.0)), \
         patch.object(ops, "start_watcher_task",
                      return_value=ops.StepResult("start_watcher_task", True, "OK", 1.0)):
        result = ops.start_all(_StubSettings(), force=True)

    # No guard step at all.
    step_names = [s.name for s in result.steps]
    assert "budget_guard" not in step_names
    assert "start_lms_server" in step_names


def test_guard_silent_when_probe_returns_none() -> None:
    """If nvidia-smi or the RAM probe fails (returns None), guard
    silently proceeds — better to risk a load attempt than to leave
    CPU-only / GPU-less setups stuck behind a False refusal."""
    with patch.object(ops, "_budget_verdict_for_code_rag", return_value=None), \
         patch("code_rag.util.stop_marker.clear_intentionally_stopped",
               return_value=True), \
         patch.object(ops, "start_lms_server",
                      return_value=ops.StepResult("start_lms_server", True, "ready", 1.0)), \
         patch.object(ops, "load_model",
                      return_value=ops.StepResult("load_model(x)", True, "loaded", 1.0)), \
         patch.object(ops, "start_watcher_task",
                      return_value=ops.StepResult("start_watcher_task", True, "OK", 1.0)):
        result = ops.start_all(_StubSettings())

    step_names = [s.name for s in result.steps]
    assert "budget_guard" not in step_names
    assert "start_lms_server" in step_names

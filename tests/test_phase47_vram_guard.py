"""Phase 47: Start All refuses when GPU VRAM is already > 50% used.

User context: today the user clicked Start All while ComfyUI was
holding ~17 GB on the 22.5 GB GPU. LM Studio's `lms load` then
hung for 109 seconds and returned "Unknown error" — KV-cache
allocation OOM'd because there wasn't enough VRAM left to load
the embedder cleanly. The user asked for a guard.

Default threshold: 50%. If exceeded, Start All bails BEFORE clearing
the stop marker / starting LM Studio. `force=True` overrides.
Dashboard exposes the override via `?force=1` or `{"force": true}`.

Guard is silent on machines with no GPU (nvidia-smi unavailable
returns None) — we don't block CPU-only setups.
"""
from __future__ import annotations

from unittest.mock import patch

from code_rag.dashboard import operations as ops


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
        kind = "cross_encoder"   # not lm_chat → load_model not called for reranker
        model = "BAAI/bge-reranker-v2-m3"

    paths = _Paths()
    embedder = _Embedder()
    reranker = _Reranker()


def _gpu(used_gb: float, total_gb: float = 22.5) -> dict:
    return {
        "name": "NVIDIA GeForce RTX 4090",
        "vram_used_gb": used_gb,
        "vram_total_gb": total_gb,
        "util_pct": 0,
        "temp_c": 40,
    }


def test_guard_blocks_when_vram_above_50pct() -> None:
    """ComfyUI scenario: 17 GB / 22.5 GB used (~75%). Start All must
    refuse and return BEFORE touching the stop marker, LM Studio, or
    the watcher task."""
    with patch.object(ops, "_gpu_status_via_nvidia_smi",
                      return_value=_gpu(used_gb=17.0)), \
         patch.object(ops, "start_lms_server",
                      side_effect=AssertionError("must not start LMS when VRAM full")), \
         patch.object(ops, "start_watcher_task",
                      side_effect=AssertionError("must not start watcher when VRAM full")):
        result = ops.start_all(_StubSettings())

    assert result.ok is False
    # Single-step result: just the guard.
    assert len(result.steps) == 1
    guard = result.steps[0]
    assert guard.name == "vram_guard"
    assert guard.ok is False
    assert "75%" in guard.detail or "76%" in guard.detail   # 17/22.5 = 75.6%
    assert "force" in guard.detail.lower()


def test_guard_passes_when_vram_below_50pct() -> None:
    """5 GB / 22.5 GB (~22%). Guard records OK and Start All proceeds."""
    with patch.object(ops, "_gpu_status_via_nvidia_smi",
                      return_value=_gpu(used_gb=5.0)), \
         patch.object(ops, "clear_intentionally_stopped" if hasattr(ops, "clear_intentionally_stopped") else "_unused", create=True), \
         patch("code_rag.util.stop_marker.clear_intentionally_stopped",
               return_value=True), \
         patch.object(ops, "start_lms_server",
                      return_value=ops.StepResult("start_lms_server", True, "ready", 1.0)), \
         patch.object(ops, "load_model",
                      return_value=ops.StepResult("load_model(x)", True, "loaded", 1.0)), \
         patch.object(ops, "start_watcher_task",
                      return_value=ops.StepResult("start_watcher_task", True, "OK", 1.0)):
        result = ops.start_all(_StubSettings())

    # First step is the guard, recording PASS.
    assert result.steps[0].name == "vram_guard"
    assert result.steps[0].ok is True
    assert "22%" in result.steps[0].detail
    # Subsequent steps ran.
    step_names = [s.name for s in result.steps]
    assert "start_lms_server" in step_names
    assert "start_watcher_task" in step_names


def test_force_override_bypasses_guard() -> None:
    """Even at 90% VRAM, force=True skips the guard entirely."""
    with patch.object(ops, "_gpu_status_via_nvidia_smi",
                      return_value=_gpu(used_gb=20.0)), \
         patch("code_rag.util.stop_marker.clear_intentionally_stopped",
               return_value=True), \
         patch.object(ops, "start_lms_server",
                      return_value=ops.StepResult("start_lms_server", True, "ready", 1.0)), \
         patch.object(ops, "load_model",
                      return_value=ops.StepResult("load_model(x)", True, "loaded", 1.0)), \
         patch.object(ops, "start_watcher_task",
                      return_value=ops.StepResult("start_watcher_task", True, "OK", 1.0)):
        result = ops.start_all(_StubSettings(), force=True)

    # Guard didn't run at all.
    step_names = [s.name for s in result.steps]
    assert "vram_guard" not in step_names
    assert "start_lms_server" in step_names


def test_guard_silent_when_no_gpu_detected() -> None:
    """nvidia-smi unavailable (CPU-only machine) → guard is a no-op,
    Start All proceeds normally."""
    with patch.object(ops, "_gpu_status_via_nvidia_smi", return_value=None), \
         patch("code_rag.util.stop_marker.clear_intentionally_stopped",
               return_value=True), \
         patch.object(ops, "start_lms_server",
                      return_value=ops.StepResult("start_lms_server", True, "ready", 1.0)), \
         patch.object(ops, "load_model",
                      return_value=ops.StepResult("load_model(x)", True, "loaded", 1.0)), \
         patch.object(ops, "start_watcher_task",
                      return_value=ops.StepResult("start_watcher_task", True, "OK", 1.0)):
        result = ops.start_all(_StubSettings())

    # No vram_guard step recorded — silently proceed.
    step_names = [s.name for s in result.steps]
    assert "vram_guard" not in step_names
    assert "start_lms_server" in step_names


def test_guard_at_exactly_threshold_passes() -> None:
    """Phase 47 contract: > 50%, not >=. Exactly 50% is allowed
    (avoid flapping at the boundary)."""
    with patch.object(ops, "_gpu_status_via_nvidia_smi",
                      return_value=_gpu(used_gb=11.25)), \
         patch("code_rag.util.stop_marker.clear_intentionally_stopped",
               return_value=True), \
         patch.object(ops, "start_lms_server",
                      return_value=ops.StepResult("start_lms_server", True, "ready", 1.0)), \
         patch.object(ops, "load_model",
                      return_value=ops.StepResult("load_model(x)", True, "loaded", 1.0)), \
         patch.object(ops, "start_watcher_task",
                      return_value=ops.StepResult("start_watcher_task", True, "OK", 1.0)):
        # 11.25 / 22.5 = 0.50 exactly — guard should accept.
        result = ops.start_all(_StubSettings())

    guard_steps = [s for s in result.steps if s.name == "vram_guard"]
    assert len(guard_steps) == 1
    assert guard_steps[0].ok is True

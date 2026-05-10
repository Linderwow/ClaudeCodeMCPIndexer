"""Phase 52: resource arbitrator tests.

The 22.5 GB VRAM 4090 isn't big enough to run code-rag (~11 GB after
Phase 60-C centralization) + YouTubeBot with ComfyUI (~14 GB) + COD
simultaneously without careful budgeting. Before any project starts
we need to refuse if there isn't enough headroom.

These tests pin:
  - the cost registry has all three projects (catches accidental
    deletion of a project from the registry)
  - can_start_project returns ok=True when there's headroom
  - can_start_project returns ok=False with the right bottleneck
    label (vram / ram / both) when there isn't
  - the suggestion text names what's short + by how much
  - reserves are deducted before computing 'available'

NO LIVE PowerShell or nvidia-smi calls — every test feeds in a fake
CurrentResources directly.
"""
from __future__ import annotations

from code_rag.dashboard.resource_budget import (
    PROJECT_COSTS,
    RAM_RESERVE_GB,
    VRAM_RESERVE_GB,
    CurrentResources,
    ProjectCost,
    can_start_project,
    get_all_verdicts,
    verdict_to_dict,
)


# ---- registry pins -------------------------------------------------------


def test_registry_has_all_three_projects() -> None:
    """A future refactor that drops a project from the registry must
    not silently break the dashboard's budget panel."""
    assert set(PROJECT_COSTS.keys()) == {"code-rag", "YouTubeBot", "MNQAlpha"}


def test_registry_costs_are_sane_estimates() -> None:
    """Sanity-check the tuned numbers — if a future edit zeroes out a
    project or sets it to 100 GB, this catches it."""
    for pid, cost in PROJECT_COSTS.items():
        assert cost.ram_gb > 0, f"{pid} ram_gb must be positive"
        assert cost.vram_gb >= 0, f"{pid} vram_gb must be non-negative"
        assert cost.ram_gb < 64, f"{pid} ram_gb implausibly large"
        assert cost.vram_gb < 24, f"{pid} vram_gb >= 4090 capacity is wrong"


def test_reserves_are_nonzero() -> None:
    """Reserves carve out headroom for browser/OS spikes. A future
    refactor that zeros these out would let a workload OOM during a
    routine spike."""
    assert RAM_RESERVE_GB > 0
    assert VRAM_RESERVE_GB > 0


# ---- happy path: plenty of headroom --------------------------------------


def test_can_start_when_resources_are_idle() -> None:
    """Idle dev box: 2 GB RAM used, 0 GB VRAM used. All projects pass."""
    cur = CurrentResources(
        ram_used_gb=2.0, ram_total_gb=64.0,
        vram_used_gb=0.0, vram_total_gb=22.5,
    )
    for pid in PROJECT_COSTS:
        v = can_start_project(pid, cur)
        assert v.ok, f"{pid} should pass on idle box; got {v.suggestion}"
        assert v.bottleneck is None
        assert v.suggestion == ""


# ---- VRAM exhaustion (the user's actual case) ----------------------------


def test_refuse_code_rag_when_comfyui_is_holding_vram() -> None:
    """ComfyUI scenario: 18 GB VRAM used. code-rag needs 18 GB more →
    only 22.5 - 2 reserve - 18 = 2.5 GB available → refuse. RAM is
    fine here (64-4-20=40 avail vs 18 needed) so bottleneck is vram."""
    cur = CurrentResources(
        ram_used_gb=20.0, ram_total_gb=64.0,
        vram_used_gb=18.0, vram_total_gb=22.5,
    )
    v = can_start_project("code-rag", cur)
    assert v.ok is False
    assert v.bottleneck == "vram"
    assert "VRAM" in v.suggestion
    assert "code-rag" in v.suggestion or "code-rag" in v.suggestion.lower()


def test_refuse_youtubebot_when_cod_and_code_rag_are_running() -> None:
    """COD + code-rag scenario: 16 GB VRAM used (10 cod + 6 code-rag).
    YouTubeBot needs 14 GB more → only 22.5 - 2 - 16 = 4.5 GB → refuse."""
    cur = CurrentResources(
        ram_used_gb=30.0, ram_total_gb=64.0,
        vram_used_gb=16.0, vram_total_gb=22.5,
    )
    v = can_start_project("YouTubeBot", cur)
    assert v.ok is False
    assert v.bottleneck == "vram"


def test_suggestion_says_how_much_to_free() -> None:
    """The suggestion text must include a concrete 'free X GB' number
    so the user can act without guessing."""
    cur = CurrentResources(
        ram_used_gb=2.0, ram_total_gb=64.0,
        vram_used_gb=18.0, vram_total_gb=22.5,
    )
    v = can_start_project("code-rag", cur)
    assert v.ok is False
    # vram cost 11 - available (22.5 - 2 - 18 = 2.5) = 8.5 GB short
    # (Phase 60-C: cost dropped 18 → 11 GB after rerank centralization
    # + vLLM mem-util tightening; the assertion follows the cost.)
    assert "free" in v.suggestion.lower()
    assert "8.5 GB" in v.suggestion or "8.4 GB" in v.suggestion or "8.6 GB" in v.suggestion


# ---- RAM exhaustion ------------------------------------------------------


def test_refuse_when_only_ram_is_short() -> None:
    """VRAM is fine but RAM is full: bottleneck=ram, not vram."""
    cur = CurrentResources(
        ram_used_gb=58.0, ram_total_gb=64.0,    # only 2 GB free after reserve
        vram_used_gb=0.0, vram_total_gb=22.5,
    )
    v = can_start_project("code-rag", cur)   # needs 12 GB RAM (Phase 60-C)
    assert v.ok is False
    assert v.bottleneck == "ram"
    assert "RAM" in v.suggestion


def test_bottleneck_both_when_both_are_short() -> None:
    """If both RAM and VRAM are stretched, bottleneck label is 'both'."""
    cur = CurrentResources(
        ram_used_gb=58.0, ram_total_gb=64.0,
        vram_used_gb=18.0, vram_total_gb=22.5,
    )
    v = can_start_project("code-rag", cur)
    assert v.ok is False
    assert v.bottleneck == "both"
    # Both RAM and VRAM should be mentioned in the suggestion.
    assert "RAM" in v.suggestion
    assert "VRAM" in v.suggestion


# ---- reserve enforcement -------------------------------------------------


def test_reserve_is_subtracted_from_available() -> None:
    """If total VRAM = 22.5 and reserve = 2.0, the budget treats only
    20.5 GB as usable. A 21 GB workload must refuse even on an empty
    GPU."""
    fake_cost = ProjectCost(
        project_id="fake-big", label="fake", ram_gb=1.0,
        vram_gb=22.5 - VRAM_RESERVE_GB + 0.5,   # 0.5 GB above usable
    )
    cur = CurrentResources(
        ram_used_gb=0.0, ram_total_gb=64.0,
        vram_used_gb=0.0, vram_total_gb=22.5,
    )
    v = can_start_project("any", cur, cost_override=fake_cost)
    assert v.ok is False
    assert v.bottleneck == "vram"


def test_reserve_does_not_block_workloads_that_fit() -> None:
    """A workload that uses exactly (total - reserve) on idle hardware
    must still pass — reserve is a CEILING, not a tax-on-everything."""
    fake_cost = ProjectCost(
        project_id="fake-fit", label="fits", ram_gb=1.0,
        vram_gb=22.5 - VRAM_RESERVE_GB,   # exactly the budget
    )
    cur = CurrentResources(
        ram_used_gb=0.0, ram_total_gb=64.0,
        vram_used_gb=0.0, vram_total_gb=22.5,
    )
    v = can_start_project("any", cur, cost_override=fake_cost)
    assert v.ok is True


# ---- unknown project -----------------------------------------------------


def test_unknown_project_returns_clear_error() -> None:
    cur = CurrentResources(
        ram_used_gb=0.0, ram_total_gb=64.0,
        vram_used_gb=0.0, vram_total_gb=22.5,
    )
    v = can_start_project("not-a-project", cur)
    assert v.ok is False
    assert "unknown" in v.suggestion.lower()


# ---- get_all_verdicts ----------------------------------------------------


def test_get_all_verdicts_returns_one_per_project() -> None:
    cur = CurrentResources(
        ram_used_gb=0.0, ram_total_gb=64.0,
        vram_used_gb=0.0, vram_total_gb=22.5,
    )
    verdicts = get_all_verdicts(cur)
    assert len(verdicts) == len(PROJECT_COSTS)
    project_ids = {v.project_id for v in verdicts}
    assert project_ids == set(PROJECT_COSTS.keys())


# ---- verdict_to_dict (frontend contract) ---------------------------------


def test_verdict_to_dict_shape_matches_frontend_expectations() -> None:
    cur = CurrentResources(
        ram_used_gb=0.0, ram_total_gb=64.0,
        vram_used_gb=0.0, vram_total_gb=22.5,
    )
    v = can_start_project("code-rag", cur)
    d = verdict_to_dict(v)
    # Every field the frontend reads:
    assert set(d.keys()) == {
        "project_id", "label", "ok",
        "cost_ram_gb", "cost_vram_gb",
        "available_ram_gb", "available_vram_gb",
        "bottleneck", "suggestion", "notes",
    }
    assert d["project_id"] == "code-rag"
    assert d["label"] == "code-rag (full stack)"
    assert isinstance(d["ok"], bool)
    assert isinstance(d["cost_ram_gb"], float)
    assert isinstance(d["notes"], str)

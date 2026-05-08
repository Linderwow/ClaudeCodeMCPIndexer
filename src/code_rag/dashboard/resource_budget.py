"""Phase 52: resource arbitrator.

User context: 22.5 GB VRAM on the 4090 isn't enough to run code-rag
+ YouTubeBot (with ComfyUI loaded) + MNQAlpha simultaneously. Trying
to start a workload that would exceed available VRAM/RAM produces
silent failures (LM Studio OOM with "Unknown error" after 100s,
ComfyUI crash, etc.). We need to refuse BEFORE starting.

Design
------
Each project declares a `ProjectCost` — RAM + VRAM it expects to use
when fully running. The numbers are estimates (varies by ComfyUI
checkpoint, whether HyDE is loaded, how many MCP servers Claude Code
spawned, etc.) but they're conservative ceilings tuned from
observation in this session.

Two safety reserves are kept off-limits:
  - VRAM_RESERVE_GB: headroom for browser/GPU compositing spikes,
    Windows desktop, NinjaTrader chart rendering, etc.
  - RAM_RESERVE_GB:  same for system RAM spikes.

`can_start_project(project_id, current)` returns a verdict:
  - ok=True if (current_used + project_cost) <= (total - reserve)
  - ok=False with `suggestion` naming what to close to free room

The verdict is consumed by:
  - dashboard.operations.start_all (refuses with actionable message)
  - GET /api/budget (frontend renders per-project status badges)

NOT a live process tracker — this is a pre-flight check based on
current resource state + declared project cost. We don't try to
predict ComfyUI's actual VRAM usage; we use the declared upper bound
and let the user override with `force=true` if they know better.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---- safety reserves ------------------------------------------------------
# Tuned from observation: GPU compositing for Chrome/Edge/Teams + LM Studio
# CUDA shell + NinjaTrader chart renderers can spike 2-3 GB unexpectedly.
# Leave headroom so a spike doesn't OOM a workload mid-run.
VRAM_RESERVE_GB = 2.0
RAM_RESERVE_GB = 4.0


@dataclass(frozen=True)
class ProjectCost:
    """Estimated peak RAM + VRAM cost when this project is fully running.

    Numbers are CEILINGS — actual usage may be lower. Tuned from observed
    process snapshots in this dev session. Update as workloads change
    (e.g. switching from FLUX to SDXL drops ComfyUI VRAM ~6 GB).
    """
    project_id: str
    label: str
    ram_gb: float
    vram_gb: float
    notes: str = ""


# ---- the registry ---------------------------------------------------------
# Tuned 2026-05-08 from this user's observed running stack.

PROJECT_COSTS: dict[str, ProjectCost] = {
    "code-rag": ProjectCost(
        project_id="code-rag",
        label="code-rag (full stack)",
        ram_gb=14.0,
        vram_gb=12.0,
        notes=(
            "Embedder qwen3-embedding-4b @ ctx=4096 (~3 GB VRAM, ~2 GB RAM) "
            "+ HyDE qwen2.5-coder-7b (~8 GB VRAM, ~7 GB RAM if loaded) "
            "+ 4 MCP servers each holding a cross-encoder (~1.5 GB VRAM, "
            "~2 GB RAM) + watcher (~1 GB RAM) + dashboard (~0.1 GB RAM). "
            "If HyDE isn't actively loaded, drop 8 GB VRAM and 7 GB RAM "
            "from these numbers."
        ),
    ),
    "YouTubeBot": ProjectCost(
        project_id="YouTubeBot",
        label="YouTubeBot (ComfyUI active)",
        ram_gb=18.0,
        vram_gb=14.0,
        notes=(
            "ComfyUI with FLUX checkpoint loaded (~14 GB VRAM, ~18 GB RAM "
            "when model is resident — observed today). With SDXL instead, "
            "drop to ~8 GB VRAM, ~10 GB RAM. With ComfyUI not running but "
            "--watch idle, drop to ~0.5 GB RAM, 0 VRAM."
        ),
    ),
    "MNQAlpha": ProjectCost(
        project_id="MNQAlpha",
        label="MNQAlpha (signals daemons)",
        ram_gb=2.0,
        vram_gb=0.5,
        notes=(
            "Six pythonw daemons (~50-500 MB each), trade_gate is the "
            "heaviest at ~450 MB. 0.5 GB VRAM allowance for the rare "
            "MNQAlpha_DeepRetrain task; idle daemons use 0 VRAM."
        ),
    ),
}


@dataclass
class BudgetVerdict:
    """Outcome of `can_start_project`."""
    project_id: str
    ok: bool
    cost_ram_gb: float
    cost_vram_gb: float
    available_ram_gb: float
    available_vram_gb: float
    bottleneck: str | None  # "ram" | "vram" | "both" | None
    suggestion: str = ""    # actionable text the user can act on


@dataclass
class CurrentResources:
    """Live snapshot of machine resource state. Caller fills these in
    from nvidia-smi + Win32_OperatingSystem so the budget logic stays
    side-effect-free + testable."""
    ram_used_gb: float
    ram_total_gb: float
    vram_used_gb: float
    vram_total_gb: float
    # Optional: name of the heaviest non-project process eating VRAM (e.g.
    # "ComfyUI", "cod.exe"). Used to make "Free X" suggestions concrete.
    heaviest_consumer_label: str | None = None


def _format_gb(g: float) -> str:
    return f"{g:.1f} GB"


def can_start_project(
    project_id: str,
    current: CurrentResources,
    *,
    cost_override: ProjectCost | None = None,
) -> BudgetVerdict:
    """Pre-flight check: can we start `project_id` given the current
    resource state without exceeding (total - reserve)?

    `cost_override` lets callers temporarily downgrade the cost (e.g.
    "I'm starting code-rag without HyDE, cost is just 4 GB VRAM not 12").
    Default uses PROJECT_COSTS[project_id].
    """
    cost = cost_override or PROJECT_COSTS.get(project_id)
    if cost is None:
        return BudgetVerdict(
            project_id=project_id,
            ok=False,
            cost_ram_gb=0.0, cost_vram_gb=0.0,
            available_ram_gb=0.0, available_vram_gb=0.0,
            bottleneck=None,
            suggestion=f"unknown project '{project_id}' — no cost registered",
        )

    # Available = total - reserve - already_used. Reserve carves out
    # headroom for non-project spikes (browser, OS, etc.).
    avail_ram = max(
        0.0, current.ram_total_gb - RAM_RESERVE_GB - current.ram_used_gb,
    )
    avail_vram = max(
        0.0, current.vram_total_gb - VRAM_RESERVE_GB - current.vram_used_gb,
    )

    ram_ok = avail_ram >= cost.ram_gb
    vram_ok = avail_vram >= cost.vram_gb

    if ram_ok and vram_ok:
        return BudgetVerdict(
            project_id=project_id,
            ok=True,
            cost_ram_gb=cost.ram_gb, cost_vram_gb=cost.vram_gb,
            available_ram_gb=avail_ram, available_vram_gb=avail_vram,
            bottleneck=None,
            suggestion="",
        )

    # Determine bottleneck + craft suggestion.
    if not ram_ok and not vram_ok:
        bottleneck = "both"
    elif not vram_ok:
        bottleneck = "vram"
    else:
        bottleneck = "ram"

    # How much we'd need to free.
    ram_short = max(0.0, cost.ram_gb - avail_ram)
    vram_short = max(0.0, cost.vram_gb - avail_vram)

    parts = [
        f"refuse start: {cost.label} needs "
        f"{_format_gb(cost.ram_gb)} RAM + {_format_gb(cost.vram_gb)} VRAM"
    ]
    if not vram_ok:
        parts.append(
            f"only {_format_gb(avail_vram)} VRAM available "
            f"({_format_gb(VRAM_RESERVE_GB)} reserved); "
            f"need to free {_format_gb(vram_short)}"
        )
    if not ram_ok:
        parts.append(
            f"only {_format_gb(avail_ram)} RAM available "
            f"({_format_gb(RAM_RESERVE_GB)} reserved); "
            f"need to free {_format_gb(ram_short)}"
        )
    if current.heaviest_consumer_label:
        parts.append(
            f"heaviest non-project consumer right now: "
            f"{current.heaviest_consumer_label} — closing it would "
            f"likely free enough"
        )
    suggestion = ". ".join(parts) + "."

    return BudgetVerdict(
        project_id=project_id,
        ok=False,
        cost_ram_gb=cost.ram_gb, cost_vram_gb=cost.vram_gb,
        available_ram_gb=avail_ram, available_vram_gb=avail_vram,
        bottleneck=bottleneck,
        suggestion=suggestion,
    )


def get_all_verdicts(current: CurrentResources) -> list[BudgetVerdict]:
    """Return a verdict for every registered project — used by the
    dashboard `/api/budget` endpoint to power per-project badges."""
    return [
        can_start_project(pid, current) for pid in PROJECT_COSTS
    ]


def verdict_to_dict(v: BudgetVerdict) -> dict[str, Any]:
    """JSON-friendly shape for the frontend."""
    cost = PROJECT_COSTS.get(v.project_id)
    return {
        "project_id":          v.project_id,
        "label":               cost.label if cost else v.project_id,
        "ok":                  v.ok,
        "cost_ram_gb":         round(v.cost_ram_gb, 1),
        "cost_vram_gb":        round(v.cost_vram_gb, 1),
        "available_ram_gb":    round(v.available_ram_gb, 1),
        "available_vram_gb":   round(v.available_vram_gb, 1),
        "bottleneck":          v.bottleneck,
        "suggestion":          v.suggestion,
        "notes":               cost.notes if cost else "",
    }

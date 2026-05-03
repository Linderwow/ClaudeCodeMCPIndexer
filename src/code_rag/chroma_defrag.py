"""Phase 36-D: Chroma defrag — periodic preventive maintenance.

Why this exists
---------------
Chroma's HNSW `data_level0.bin` keeps capacity from the peak collection
size and never shrinks. Across many edit cycles + crashes the SQLite
WAL accumulates cruft that's not fully checkpointed. Combined effect:
Chroma's `count()` and `query()` get progressively slower and eventually
deadlock (the May 3 failure).

Defrag strategy: wipe + reindex
-------------------------------
Investigated an in-place migration (read all → write to fresh collection
→ delete old → rename new). Chroma has NO rename API; the
delete-and-recreate dance corrupts data. The only safe, simple path is
the same as the corrupted-Chroma recovery: wipe chroma/ and let the
watcher rebuild against the FTS source-of-truth.

Trigger: oversized `data_level0.bin` (default >5 GB) or weekly cadence,
whichever fires first. Scheduled task installs at install-time.

Tradeoff: a defrag costs ~1-2h of reindex time. Run weekly during a
sleep window. Compared to a corruption-induced 6-hour deadlock + wipe,
this is cheap insurance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from code_rag.logging import get

log = get(__name__)


# Trigger threshold — when data_level0.bin grows past this, schedule a defrag.
# 5 GB chosen because: empty collection ≈ <100 MB, 100K vectors at dim 2560
# ≈ 1 GB. Anything past 5 GB is mostly preallocated waste.
_DEFRAG_THRESHOLD_BYTES = 5 * 1024 * 1024 * 1024


def needs_defrag(chroma_dir: Path) -> tuple[bool, dict[str, Any]]:
    """Return (should_defrag, report) — pure read-only check.

    A defrag is recommended when total `data_level0.bin` size across
    all chroma collections exceeds the threshold. The report captures
    sizes so callers can log the decision.
    """
    sizes: dict[str, int] = {}
    if chroma_dir.exists():
        for p in chroma_dir.rglob("data_level0.bin"):
            try:
                sizes[str(p.relative_to(chroma_dir))] = p.stat().st_size
            except OSError:
                pass
    total = sum(sizes.values())
    return total > _DEFRAG_THRESHOLD_BYTES, {
        "total_bytes": total,
        "threshold_bytes": _DEFRAG_THRESHOLD_BYTES,
        "files": sizes,
    }


def defrag_via_wipe_reindex(
    chroma_dir: Path,
    *,
    index_meta_path: Path | None = None,
) -> dict[str, Any]:
    """Defrag = wipe Chroma + let the watcher reindex.

    Reuses the chroma_watchdog heal path. The watcher's RestartCount
    in Task Scheduler will spawn it back up after we kill it; it'll
    re-embed every chunk currently in FTS.

    Returns the heal report. Caller is responsible for scheduling
    cadence — this function just runs the wipe once when called.
    """
    from code_rag.chroma_watchdog import _kill_chroma_holders, wipe_chroma
    log.info("chroma_defrag.starting", chroma_dir=str(chroma_dir))
    _kill_chroma_holders()
    ok = wipe_chroma(chroma_dir, also_index_meta=index_meta_path)
    log.info("chroma_defrag.done", wipe_ok=ok)
    return {"wipe_ok": ok, "method": "wipe_reindex"}

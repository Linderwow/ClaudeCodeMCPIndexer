"""LM Studio "zombie instance" janitor.

What this fixes
---------------
LM Studio's auto-load / Just-In-Time loading occasionally spawns a SECOND
copy of an already-loaded model, suffixed `:2` (then `:3`, `:4`, ...).
This happens when:
  - the existing instance is busy / saturated, OR
  - an `lms load` is called while the model is already loaded but with
    different parallel/context settings (LM Studio creates a new instance
    instead of reconfiguring the existing one — observed with our autostart
    bootstrap requesting `parallel=4 context=4096` while a user-loaded
    instance was already running with stock settings).

Each duplicate burns the full model size in VRAM (e.g. 2.5 GB for the
qwen3-embedding-4B). Steady-state we only ever need ONE copy; the
duplicate is pure waste.

Detection rule
--------------
A loaded model with id `<base>:<N>` (N >= 2) is a zombie iff a model with
id `<base>` (no suffix) is ALSO currently loaded. If only the suffixed
copy is loaded (e.g. user explicitly loaded `model:2`), it's not a zombie.

The janitor runs every `interval_s` inside the autostart watcher process,
so cleanup is automatic and survives reboots without any extra service.
"""
from __future__ import annotations

import asyncio
import re
import subprocess
from dataclasses import dataclass

from code_rag.lms_ctl import find_lms, list_loaded_models_v0, unload_model
from code_rag.logging import get

log = get(__name__)

# Match identifiers like `text-embedding-qwen3-embedding-4b:2`.
# Capture group 1 = base name, group 2 = suffix index.
_SUFFIX_RE = re.compile(r"^(.+):(\d+)$")


@dataclass(frozen=True)
class Zombie:
    """A duplicate-instance identifier we plan to unload."""

    identifier: str   # e.g. "text-embedding-qwen3-embedding-4b:2"
    base: str         # e.g. "text-embedding-qwen3-embedding-4b"
    suffix: int       # e.g. 2


def find_zombies(loaded_models: list[dict]) -> list[Zombie]:
    """Pure helper: scan `loaded_models` (output of `list_loaded_models_v0`)
    and return every `<base>:<N>` whose matching `<base>` (no suffix) is
    also in the loaded set. Stable ordering by identifier so logs are
    deterministic and tests are easy to assert on."""
    loaded_ids = {m.get("id") for m in loaded_models if m.get("id")}
    zombies: list[Zombie] = []
    for mid in loaded_ids:
        m = _SUFFIX_RE.match(mid or "")
        if not m:
            continue
        base, idx = m.group(1), int(m.group(2))
        if idx < 2:                       # `:0` / `:1` shouldn't happen but skip
            continue
        if base in loaded_ids:            # base also loaded → suffixed is zombie
            zombies.append(Zombie(identifier=mid, base=base, suffix=idx))
    zombies.sort(key=lambda z: z.identifier)
    return zombies


async def cleanup_once(base_url: str) -> int:
    """Run one detection + unload pass. Returns the number of zombies
    successfully unloaded (so callers can log a summary). Best-effort: any
    exception is logged and swallowed — janitor errors must never bring
    down the watcher process."""
    loaded = list_loaded_models_v0(base_url)
    zombies = find_zombies(loaded)
    if not zombies:
        return 0

    loc = find_lms()
    if loc.path is None:
        log.warning("lm_janitor.lms_not_found",
                    zombies=[z.identifier for z in zombies])
        return 0

    unloaded = 0
    for z in zombies:
        try:
            # Phase 38 (audit fix): bumped timeout from default 30s to 90s.
            # 7B-class rerankers can legitimately take 30-60s to flush KV
            # cache + GPU buffers under VRAM pressure; killing mid-unload
            # leaves LM Studio in undefined state and the zombie often
            # remains. 90s gives enough headroom for the slowest realistic
            # case while still bounding the janitor cycle.
            r = await asyncio.to_thread(
                unload_model, loc.path, z.identifier, timeout_s=90.0,
            )
        except subprocess.TimeoutExpired:
            log.warning("lm_janitor.unload_timeout", identifier=z.identifier)
            continue
        except Exception as e:  # pragma: no cover — defensive
            log.warning("lm_janitor.unload_error",
                        identifier=z.identifier,
                        err=f"{type(e).__name__}: {e}")
            continue
        if r.returncode == 0:
            unloaded += 1
            log.info("lm_janitor.unloaded",
                     identifier=z.identifier, base=z.base, suffix=z.suffix)
        else:
            log.warning("lm_janitor.unload_nonzero",
                        identifier=z.identifier,
                        rc=r.returncode,
                        stderr=(r.stderr or "")[:200])
    return unloaded


async def janitor_loop(base_url: str, interval_s: float = 60.0) -> None:
    """Long-lived task: every `interval_s` seconds, scan for zombie
    instances and unload them. Designed to be `asyncio.gather`-ed with the
    main watcher loop in `autostart_bootstrap`. Never raises — errors are
    logged and the loop continues."""
    log.info("lm_janitor.started", base_url=base_url, interval_s=interval_s)
    while True:
        try:
            n = await cleanup_once(base_url)
            if n:
                log.info("lm_janitor.cycle", unloaded=n)
        except Exception as e:  # pragma: no cover — defensive
            log.warning("lm_janitor.cycle_error",
                        err=f"{type(e).__name__}: {e}")
        await asyncio.sleep(interval_s)

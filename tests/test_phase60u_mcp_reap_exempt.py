"""Phase 60-U: MCP processes are exempt from the orphan reaper.

WHY THIS EXISTS

The reaper's ancestor-walk heuristic worked fine for watcher / dashboard
processes (which always launch from Task Scheduler with a stable
svchost.exe / taskeng.exe / explorer.exe parent). It broke catastrophically
for MCP servers, which Claude Code spawns through a node-mediated
venv-stub chain:

    claude.exe (node)
      → npx (node) [exits fast]
        → venv launcher stub (exits fast)
          → python.exe -m code_rag mcp   ← gets reparented to System

When the intermediate launchers exit, the python child gets reparented
to PID 4 (System). The reaper walks the ancestor chain looking for
"claude.exe" or "claude", hits System without finding either, and
classifies the LIVE MCP as orphan → kills it. Visible symptom:
"MCP code-rag: Server disconnected" every 10 minutes on this system.

THE FIX

`classify_orphans` now skips processes where `kind == "mcp"`. The MCP
server runs its OWN parent-death watchdog (see
`start_parent_death_watchdog` at module top) that polls os.getppid()
every 10s and self-exits when claude.exe dies. That is the canonical
authority on when MCP should die; the reaper's ancestor-walk is
redundant AND wrong for this kind.

The reaper still inventories MCPs (so operator tools that read
`list_code_rag_processes()` see them) — they just always have
`is_orphan=False` regardless of ancestor topology.
"""
from __future__ import annotations

from code_rag.util.proc_hygiene import (
    ProcessInfo,
    classify_orphans,
)


def _make(pid: int, ppid: int, kind: str, cmdline: str = "") -> ProcessInfo:
    return ProcessInfo(
        pid=pid, ppid=ppid, name="python.exe", cmdline=cmdline,
        create_time=0.0, kind=kind,
    )


def test_mcp_with_dead_ancestor_chain_NOT_orphaned() -> None:
    """Simulate the exact bug: an MCP whose direct parent is PID 4
    (System, dead-ancestor case) -- the OLD reaper called this orphan
    and killed it. New code must leave it alone."""
    mcp = _make(pid=12345, ppid=4, kind="mcp", cmdline="-m code_rag mcp")
    classify_orphans([mcp])
    assert mcp.is_orphan is False, \
        "MCP with no claude.exe ancestor must NOT be flagged orphan -- " \
        "the parent-death watchdog inside the MCP process is the canonical " \
        "authority on when MCP should die"


def test_mcp_with_random_ancestor_NOT_orphaned() -> None:
    """An MCP reparented to some random process (e.g. an old explorer
    instance) ALSO must not be flagged."""
    mcp = _make(pid=12345, ppid=99999, kind="mcp",
                cmdline="-m code_rag mcp")
    classify_orphans([mcp])
    assert mcp.is_orphan is False


def test_watcher_with_dead_ancestor_STILL_orphaned() -> None:
    """Sanity: the exemption is MCP-SPECIFIC. Watcher / dashboard /
    indexer kinds keep their existing orphan classification. A watcher
    with no valid ancestor IS still an orphan (no internal watchdog
    on that path)."""
    watcher = _make(pid=12345, ppid=4, kind="watcher",
                    cmdline="-m code_rag.autostart_bootstrap")
    classify_orphans([watcher])
    # The ancestor walk hits System (PID 4) immediately -> orphan.
    assert watcher.is_orphan is True
    assert "System" in (watcher.reason or "")


def test_unclassified_process_unchanged() -> None:
    """Processes without a recognized kind (returned None by
    _classify_kind) are left at the default is_orphan=False."""
    unknown = _make(pid=12345, ppid=4, kind=None,
                    cmdline="some-unrelated-python-script.py")
    classify_orphans([unknown])
    assert unknown.is_orphan is False

"""Phase 42: stop-marker boot-aware clearing.

Bug context: the autostart_bootstrap was clearing the Stop All marker
on every fire, assuming AtLogon was the only trigger. Task Scheduler's
RestartCount=5 policy meant any time `Stop-ScheduledTask` terminated
the python action, the action got auto-respawned within a minute and
the marker was wiped — Stop All didn't stick.

Fix: `marker_predates_last_boot` distinguishes "marker from a previous
boot session" (clear OK) from "marker created in the current boot"
(stay stopped).
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

from code_rag.util import stop_marker
from code_rag.util.stop_marker import (
    clear_intentionally_stopped,
    is_intentionally_stopped,
    mark_intentionally_stopped,
    marker_path,
    marker_predates_last_boot,
)


def test_predates_returns_none_when_marker_missing(tmp_path: Path) -> None:
    assert marker_predates_last_boot(tmp_path) is None


def test_predates_true_when_marker_older_than_boot(tmp_path: Path) -> None:
    """Marker created last week, system booted today → safe to clear."""
    mark_intentionally_stopped(tmp_path, reason="prior session")
    p = marker_path(tmp_path)
    # Backdate the marker by 7 days.
    week_ago = time.time() - 7 * 86400
    os.utime(p, (week_ago, week_ago))
    # Pretend the system booted 1 hour ago.
    boot_time = time.time() - 3600
    with patch.object(stop_marker, "_system_boot_time_unix",
                      return_value=boot_time):
        assert marker_predates_last_boot(tmp_path) is True


def test_predates_false_when_marker_newer_than_boot(tmp_path: Path) -> None:
    """Marker created 5 minutes ago, system booted yesterday → user
    pressed Stop All in this session, stay stopped."""
    mark_intentionally_stopped(tmp_path, reason="this session")
    boot_time = time.time() - 86400  # 24 h ago
    with patch.object(stop_marker, "_system_boot_time_unix",
                      return_value=boot_time):
        assert marker_predates_last_boot(tmp_path) is False


def test_predates_false_when_boot_time_unknown(tmp_path: Path) -> None:
    """Defensive: if we can't determine boot time on this platform,
    default to 'stay stopped' rather than risk resuming against the
    user's intent."""
    mark_intentionally_stopped(tmp_path, reason="this session")
    with patch.object(stop_marker, "_system_boot_time_unix",
                      return_value=None):
        assert marker_predates_last_boot(tmp_path) is False


def test_full_round_trip(tmp_path: Path) -> None:
    """Sanity check the existing mark / is_stopped / clear flow still
    works alongside the new helper."""
    assert not is_intentionally_stopped(tmp_path)
    assert mark_intentionally_stopped(tmp_path, reason="test")
    assert is_intentionally_stopped(tmp_path)
    assert clear_intentionally_stopped(tmp_path, reason="test")
    assert not is_intentionally_stopped(tmp_path)
    assert marker_predates_last_boot(tmp_path) is None


def test_system_boot_time_returns_plausible_value() -> None:
    """Smoke test the live boot-time probe — should give a unix ts
    within the last ~year on any reasonable Windows/Linux box, or
    None on unsupported platforms (macOS, etc).

    NOT asserting the exact value; only that if it returns a number,
    it's plausibly a recent unix timestamp.
    """
    bt = stop_marker._system_boot_time_unix()
    if bt is None:
        return  # platform doesn't support — the contract allows this
    now = time.time()
    one_year_ago = now - 365 * 86400
    assert one_year_ago <= bt <= now

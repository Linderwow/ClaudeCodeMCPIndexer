"""Process-wide CREATE_NO_WINDOW monkey-patch for subprocess.Popen.

Why this exists
---------------
On Windows, every `subprocess.Popen(...)` (and therefore every
`subprocess.run(...)`, `.check_output()`, etc.) without an explicit
`creationflags` argument allocates a fresh console window. For a
daemon launched by Task Scheduler that's invisible — but for any
subprocess we spawn (lms.exe, schtasks, taskkill, powershell,
chroma_watchdog probes, git, nvidia-smi, …) the OS pops a CMD/console
window in the foreground. The user reported "terminal windows popping
out of nowhere" today.

Auditing every individual subprocess.run call site to add
`creationflags=0x08000000` is fragile (easy to miss new ones). The
cleaner pattern is a process-wide monkey-patch: subclass Popen and
default `creationflags=CREATE_NO_WINDOW` whenever the caller didn't
already specify creationflags or startupinfo.

Auto-installed from `code_rag/__init__.py` so every entry point
(cli, mcp_server, dashboard, autostart_bootstrap, watcher) gets the
patch for free. Idempotent + safe to import twice + no-op on
non-Windows.
"""
from __future__ import annotations

import subprocess
import sys

_CREATE_NO_WINDOW = 0x08000000


def install() -> bool:
    """Apply the patch. Returns True iff a fresh patch was installed
    (False if already installed or non-Windows)."""
    if sys.platform != "win32":
        return False
    if getattr(subprocess.Popen, "_silenced", False):
        return False

    _orig_Popen = subprocess.Popen

    class _SilentPopen(_orig_Popen):
        _silenced = True

        def __init__(self, *args, **kwargs):
            # Respect callers that explicitly set creationflags or
            # startupinfo — they presumably want the console (e.g.
            # interactive prompts) or have already opted into a
            # window-hiding strategy.
            if "creationflags" not in kwargs and "startupinfo" not in kwargs:
                kwargs["creationflags"] = _CREATE_NO_WINDOW
            super().__init__(*args, **kwargs)

    subprocess.Popen = _SilentPopen   # type: ignore[misc]
    return True


# Auto-install on import.
install()

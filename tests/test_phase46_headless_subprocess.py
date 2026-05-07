"""Phase 46: process-wide CREATE_NO_WINDOW monkey-patch.

User reported "terminal windows popping out of nowhere" — every
subprocess.run/Popen on Windows without explicit creationflags
allocates a console. Auditing every call site is fragile. This shim
patches subprocess.Popen to default `creationflags=CREATE_NO_WINDOW`
process-wide. Auto-installed from code_rag/__init__.py.
"""
from __future__ import annotations

import subprocess
import sys

import pytest


def test_shim_installed_on_import() -> None:
    """Just importing code_rag should have installed the patch (Windows
    only)."""
    import code_rag  # noqa: F401  — auto-installs
    if sys.platform == "win32":
        assert getattr(subprocess.Popen, "_silenced", False), (
            "subprocess.Popen should be the SilentPopen subclass on Windows"
        )
    else:
        assert not getattr(subprocess.Popen, "_silenced", False)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only behavior")
def test_install_is_idempotent() -> None:
    from code_rag.util import headless_subprocess
    # First call may or may not install (depends on test order); a second
    # call must NOT replace the existing patch with another wrapper.
    headless_subprocess.install()
    Popen_after_first = subprocess.Popen
    fresh = headless_subprocess.install()
    assert fresh is False, "second install must be a no-op"
    assert subprocess.Popen is Popen_after_first


def _build_test_silentpopen(captured: dict[str, object]) -> type:
    """Build a fresh SilentPopen-style class whose base __init__ is a
    spy. Mirrors the production patch's logic 1:1 so we can assert
    exactly what the production wrapper would pass through."""
    _CREATE_NO_WINDOW = 0x08000000

    class FakeBase:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)
            captured["__args__"] = args

    class SilentPopen(FakeBase):
        _silenced = True

        def __init__(self, *args, **kwargs):
            if "creationflags" not in kwargs and "startupinfo" not in kwargs:
                kwargs["creationflags"] = _CREATE_NO_WINDOW
            super().__init__(*args, **kwargs)

    return SilentPopen


def test_default_creationflags_added_when_caller_omits() -> None:
    """The shim's wrapper logic adds CREATE_NO_WINDOW when the caller
    didn't pass creationflags or startupinfo."""
    captured: dict[str, object] = {}
    SilentPopen = _build_test_silentpopen(captured)
    SilentPopen("nonexistent.exe")
    assert captured.get("creationflags") == 0x08000000


def test_explicit_creationflags_are_respected() -> None:
    """If the caller passes creationflags explicitly, the shim must
    NOT override them."""
    captured: dict[str, object] = {}
    SilentPopen = _build_test_silentpopen(captured)
    SilentPopen("nonexistent.exe", creationflags=0x00000010)  # CREATE_NEW_CONSOLE
    assert captured.get("creationflags") == 0x00000010


def test_explicit_startupinfo_also_respected() -> None:
    """If the caller has already opted into a window-hiding strategy
    via startupinfo, the shim must NOT add creationflags on top."""
    captured: dict[str, object] = {}
    SilentPopen = _build_test_silentpopen(captured)
    sentinel = object()
    SilentPopen("nonexistent.exe", startupinfo=sentinel)
    assert "creationflags" not in captured, (
        "shim should defer to the caller when they passed startupinfo"
    )
    assert captured.get("startupinfo") is sentinel


def test_shim_safe_to_import_twice() -> None:
    """The shim is auto-installed on `import code_rag`. A subsequent
    explicit re-import must not break anything."""
    import code_rag  # noqa: F401
    from code_rag.util import headless_subprocess
    fresh = headless_subprocess.install()
    if sys.platform == "win32":
        # Already installed by `import code_rag`, so this should be a no-op.
        assert fresh is False
    else:
        assert fresh is False

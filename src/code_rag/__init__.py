# Auto-install the CREATE_NO_WINDOW monkey-patch on subprocess.Popen so
# every later subprocess call across cli, mcp_server, dashboard,
# autostart_bootstrap, watcher, etc. runs headless. No-op on non-Windows.
# Imported FIRST so it patches subprocess before any other code-rag
# module gets a chance to spawn a child process.
from code_rag.util import headless_subprocess  # noqa: F401

from code_rag.version import __version__

__all__ = ["__version__"]

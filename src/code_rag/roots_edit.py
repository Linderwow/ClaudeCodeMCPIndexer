"""Read/modify the `[paths].roots` list in config.toml without disturbing
comments or the rest of the file.

We deliberately avoid pulling in a round-trip TOML library (tomlkit, tomli-w)
for one edit point. Instead: stdlib tomllib to parse, a targeted regex to
locate the `roots = [ ... ]` block, and a clean rewrite of just that block.
"""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

# Matches the `roots = [ ... ]` block inside [paths]. Tolerates:
#   - multi-line arrays
#   - trailing comments on each line
#   - the whole block being on one line
_ROOTS_BLOCK_RE = re.compile(
    r"(^\s*roots\s*=\s*)\[(.*?)\]",
    flags=re.DOTALL | re.MULTILINE,
)


class ConfigEditError(RuntimeError):
    pass


def read_roots(config_path: Path) -> list[Path]:
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    raw = (data.get("paths") or {}).get("roots") or []
    return [Path(str(r)) for r in raw]


def _paths_equal(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve()
    except OSError:
        return str(a).lower() == str(b).lower()


def add_root(config_path: Path, new_root: Path) -> tuple[list[Path], bool]:
    """Add `new_root` if not already present. Returns (updated_roots, added)."""
    new_root = new_root.resolve()
    if not new_root.exists():
        raise ConfigEditError(f"path does not exist: {new_root}")
    if not new_root.is_dir():
        raise ConfigEditError(f"not a directory: {new_root}")

    current = read_roots(config_path)
    if any(_paths_equal(r, new_root) for r in current):
        return current, False
    updated = [*current, new_root]
    _rewrite_roots(config_path, updated)
    return updated, True


def remove_root(config_path: Path, target: Path) -> tuple[list[Path], bool]:
    """Remove `target` if present. Returns (updated_roots, removed)."""
    current = read_roots(config_path)
    kept = [r for r in current if not _paths_equal(r, target)]
    if len(kept) == len(current):
        return current, False
    _rewrite_roots(config_path, kept)
    return kept, True


def _rewrite_roots(config_path: Path, roots: list[Path]) -> None:
    text = config_path.read_text("utf-8")
    m = _ROOTS_BLOCK_RE.search(text)
    if m is None:
        raise ConfigEditError(
            "Could not locate `roots = [...]` in config file. "
            "Edit manually or restore from template."
        )
    block = _format_roots(roots)
    new_text = text[: m.start()] + m.group(1) + block + text[m.end() :]
    tmp = config_path.with_suffix(config_path.suffix + ".tmp")
    tmp.write_text(new_text, encoding="utf-8")
    tmp.replace(config_path)


def _format_roots(roots: list[Path]) -> str:
    """Emit a multi-line TOML array with forward-slash paths on Windows."""
    if not roots:
        return "[]"
    lines = []
    for r in roots:
        posix = r.as_posix()
        lines.append(f'    "{posix}",')
    return "[\n" + "\n".join(lines) + "\n]"

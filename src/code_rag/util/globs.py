from __future__ import annotations

import fnmatch
from pathlib import Path, PurePosixPath


def _posix(p: Path | str) -> str:
    return str(PurePosixPath(Path(p).as_posix()))


def matches_any(path: Path, patterns: list[str]) -> bool:
    """Case-insensitive gitignore-ish match. Patterns are posix globs.

    We intentionally keep this dumb — no per-directory .gitignore inheritance,
    no negation. The config's ignore list is absolute/glob-against-full-path;
    walker layers per-root .gitignore on top using pathspec.
    """
    sp = _posix(path).lower()
    return any(fnmatch.fnmatchcase(sp, pat.lower()) for pat in patterns)

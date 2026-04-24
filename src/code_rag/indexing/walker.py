from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from code_rag.logging import get
from code_rag.util.globs import matches_any

log = get(__name__)

# Extension -> tree-sitter language id. Anything not listed is skipped for code.
CODE_EXT: dict[str, str] = {
    ".py":   "python",
    ".cs":   "c_sharp",
    ".ts":   "typescript",
    ".tsx":  "tsx",
    ".js":   "javascript",
    ".jsx":  "javascript",
    ".mjs":  "javascript",
    ".cjs":  "javascript",
}

DOC_EXT: dict[str, str] = {
    ".md":   "markdown",
    ".pdf":  "pdf",
    ".docx": "docx",
    # Non-AST text files carried through the doc chunker. Angular templates
    # and stylesheets go here — we index them for substring recall, not
    # semantic structure.
    ".html": "html",
    ".scss": "scss",
    ".css":  "css",
}

MAX_FILE_BYTES = 2_000_000  # 2 MB — anything bigger is almost certainly generated


class Walker:
    """Yields (root, absolute_path, language) for every indexable file under the configured roots.

    Ignore layers (checked in order):
      1. config.ignore.globs (absolute path globs)
      2. Extension allowlist (CODE_EXT or DOC_EXT)
      3. Size cap (MAX_FILE_BYTES)
    """

    def __init__(self, roots: list[Path], ignore_globs: list[str]) -> None:
        self._roots = [r.resolve() for r in roots]
        self._ignore = ignore_globs

    def _skip(self, p: Path) -> str | None:
        if matches_any(p, self._ignore):
            return "ignored"
        try:
            if p.stat().st_size > MAX_FILE_BYTES:
                return "too_large"
        except OSError as e:
            return f"stat_error:{e.__class__.__name__}"
        return None

    def iter_code(self) -> Iterator[tuple[Path, Path, str]]:
        for root in self._roots:
            if not root.exists():
                log.warning("walker.root_missing", root=str(root))
                continue
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                lang = CODE_EXT.get(p.suffix.lower())
                if lang is None:
                    continue
                reason = self._skip(p)
                if reason:
                    log.debug("walker.skip", path=str(p), reason=reason)
                    continue
                yield root, p, lang

    def iter_docs(self) -> Iterator[tuple[Path, Path, str]]:
        for root in self._roots:
            if not root.exists():
                continue
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                kind = DOC_EXT.get(p.suffix.lower())
                if kind is None:
                    continue
                reason = self._skip(p)
                if reason:
                    continue
                yield root, p, kind

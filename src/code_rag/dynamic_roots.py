"""Persistent registry of dynamically-added indexing roots.

Design
------
`config.toml` holds the **curated** roots — the ones the user explicitly
pinned. This module maintains a **separate** `data/dynamic_roots.json` that
accrues roots auto-added at runtime, typically when the MCP server sees
Claude Code working in a repo that isn't already indexed.

Keeping the two files separate means:

* The user's hand-edited `config.toml` is never mutated by automation.
* Wiping the index (`rm -rf data/`) also wipes the dynamic roots, which is
  the right semantics — the auto-discovered set is index-scoped, not config-
  scoped.
* `git status` on the tool's repo never shows churn from automation.

Schema (JSON)::

    {
      "version": 1,
      "roots": [
        {
          "path": "C:/Users/Alex/RiderProjects/foo",
          "added_at":     "2026-04-24T22:10:00+00:00",
          "last_used_at": "2026-04-24T22:10:00+00:00",
          "source":       "mcp.ensure_workspace_indexed"
        }
      ]
    }

Concurrency
-----------
Writes are atomic (tmp + rename). Multiple readers are fine. Two writers
would race (MCP adding a root while CLI prunes) — unlikely in practice;
the worst case is one of the writes wins and the loser's change is
re-applied on next call. No file-lock gymnastics needed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from code_rag.logging import get

log = get(__name__)

SCHEMA_VERSION = 1


@dataclass
class DynamicRoot:
    path: Path
    added_at: str
    last_used_at: str
    source: str = "mcp.ensure_workspace_indexed"

    def as_dict(self) -> dict[str, str]:
        return {
            "path":         self.path.as_posix(),
            "added_at":     self.added_at,
            "last_used_at": self.last_used_at,
            "source":       self.source,
        }

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> DynamicRoot:
        return cls(
            path=Path(str(d["path"])),
            added_at=str(d.get("added_at") or _now()),
            last_used_at=str(d.get("last_used_at") or d.get("added_at") or _now()),
            source=str(d.get("source") or "unknown"),
        )


@dataclass
class DynamicRoots:
    store_path: Path
    entries: list[DynamicRoot] = field(default_factory=list)

    @classmethod
    def load(cls, store_path: Path) -> DynamicRoots:
        if not store_path.exists():
            return cls(store_path=store_path)
        try:
            data = json.loads(store_path.read_text("utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            log.warning("dynamic_roots.load_failed", path=str(store_path), err=str(e))
            return cls(store_path=store_path)
        raw = data.get("roots") or []
        entries = [DynamicRoot.from_dict(d) for d in raw if d.get("path")]
        return cls(store_path=store_path, entries=entries)

    def save(self) -> None:
        payload = {
            "version": SCHEMA_VERSION,
            "roots":   [e.as_dict() for e in self.entries],
        }
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.store_path.with_suffix(self.store_path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.store_path)

    def paths(self) -> list[Path]:
        """Absolute, resolved paths for all live entries (existing on disk)."""
        out: list[Path] = []
        for e in self.entries:
            try:
                if e.path.exists() and e.path.is_dir():
                    out.append(e.path.resolve())
            except OSError:
                continue
        return out

    def contains(self, path: Path) -> bool:
        target = path.resolve()
        for e in self.entries:
            try:
                if e.path.resolve() == target:
                    return True
            except OSError:
                continue
        return False

    def add(self, path: Path, source: str = "mcp.ensure_workspace_indexed") -> bool:
        """Add `path` if not already present. Returns True on add, False on noop."""
        if self.contains(path):
            self.touch(path)
            return False
        now = _now()
        self.entries.append(DynamicRoot(
            path=path.resolve(),
            added_at=now,
            last_used_at=now,
            source=source,
        ))
        self.save()
        return True

    def touch(self, path: Path) -> None:
        """Bump last_used_at for the matching entry, if any."""
        target = path.resolve()
        changed = False
        for e in self.entries:
            try:
                if e.path.resolve() == target:
                    e.last_used_at = _now()
                    changed = True
                    break
            except OSError:
                continue
        if changed:
            self.save()

    def remove(self, path: Path) -> bool:
        """Remove `path`. Returns True on remove, False if not found."""
        target = path.resolve()
        before = len(self.entries)
        self.entries = [
            e for e in self.entries
            if _safe_resolve(e.path) != target
        ]
        if len(self.entries) == before:
            return False
        self.save()
        return True

    def prune_stale(self, older_than_days: int) -> list[Path]:
        """Remove entries not used in the last `older_than_days` days.

        Returns the list of pruned paths for caller logging.
        """
        if older_than_days <= 0:
            return []
        cutoff = _parse_iso(_now()).timestamp() - older_than_days * 86400.0
        keep: list[DynamicRoot] = []
        dropped: list[Path] = []
        for e in self.entries:
            try:
                ts = _parse_iso(e.last_used_at).timestamp()
            except ValueError:
                ts = 0.0
            if ts < cutoff:
                dropped.append(e.path)
            else:
                keep.append(e)
        if dropped:
            self.entries = keep
            self.save()
        return dropped


# ---- helpers ----------------------------------------------------------------


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_iso(s: str) -> datetime:
    # Python 3.11 `fromisoformat` handles most ISO strings including offsets.
    return datetime.fromisoformat(s)


def _safe_resolve(p: Path) -> Path:
    try:
        return p.resolve()
    except OSError:
        return p

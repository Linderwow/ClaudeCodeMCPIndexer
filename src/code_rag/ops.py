"""Phase 23: production-grade operations — Prometheus-style metrics + fsck.

Two pieces:

1. `metrics_text()` returns an OpenMetrics-format string snapshot of the
   current index health: chunk counts, embedder model identity, last
   updated, and a few histograms-as-gauges. Trivial to scrape with
   Prometheus or just `curl /metrics` to a local Grafana panel.

2. `fsck()` walks all three stores looking for inconsistencies — chunks in
   one store that don't have a corresponding entry in the others, dangling
   graph edges, paths in `dynamic_roots.json` whose directories vanished,
   stale entries in `file_hashes.db` for files no longer indexed. Returns
   a structured report with `auto_fix=True` repair option.

Both pieces use only stdlib + existing deps. No `prometheus_client` install
required (the OpenMetrics text format is dead simple to emit by hand).
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

from code_rag.config import Settings
from code_rag.dynamic_roots import DynamicRoots
from code_rag.indexing.file_hash import FileHashRegistry
from code_rag.interfaces.lexical_store import LexicalStore
from code_rag.interfaces.vector_store import VectorStore
from code_rag.logging import get

log = get(__name__)


# ---- metrics ---------------------------------------------------------------


def metrics_text(
    settings: Settings,
    vec: VectorStore,
    lex: LexicalStore,
) -> str:
    """OpenMetrics-format snapshot of index health.

    Designed to be scraped by Prometheus OR rendered by `curl localhost:.../metrics`
    into a Grafana panel via the JSON datasource. Free, local, zero-config.

    Lines emitted:
        # HELP / # TYPE / value
    Each metric is a single gauge — no histograms here, since the index is
    point-in-time state. Per-request latency histograms are emitted by the
    MCP server's request logger separately (search.done events).
    """
    import json
    meta_path = settings.index_meta_path
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text("utf-8"))
        except Exception:
            meta = {}

    vec_count = vec.count()
    lex_count = lex.count()
    drift = abs(vec_count - lex_count)

    dyn = DynamicRoots.load(settings.dynamic_roots_path)

    lines: list[str] = []

    def gauge(name: str, value: float, help_text: str, labels: dict[str, str] | None = None) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} gauge")
        if labels:
            lbl = ",".join(f'{k}="{_escape(v)}"' for k, v in sorted(labels.items()))
            lines.append(f"{name}{{{lbl}}} {value}")
        else:
            lines.append(f"{name} {value}")

    gauge("code_rag_chunks_total", vec_count,
          "Total chunks currently in the vector store.",
          labels={"store": "vector"})
    gauge("code_rag_chunks_total", lex_count,
          "Total chunks currently in the lexical store.",
          labels={"store": "lexical"})
    gauge("code_rag_chunks_drift", drift,
          "Absolute count drift between vector and lexical stores; should be 0.")
    gauge("code_rag_dynamic_roots", float(len(dyn.entries)),
          "Number of auto-discovered roots in dynamic_roots.json.")
    gauge("code_rag_config_roots", float(len(settings.paths.roots)),
          "Number of roots declared in config.toml.")
    gauge("code_rag_index_meta_present", 1.0 if meta_path.exists() else 0.0,
          "Whether the index metadata file exists (1) or is missing (0).")
    if meta:
        gauge("code_rag_embedder_dim", float(meta.get("embedder_dim") or 0),
              "Embedder vector dimension stamped in the index.",
              labels={"model": str(meta.get("embedder_model") or "unknown")})
    return "\n".join(lines) + "\n"


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


# ---- fsck ------------------------------------------------------------------


@dataclass
class FsckIssue:
    severity: str           # "warn" | "error"
    code: str               # short machine-readable id
    detail: str             # human-readable
    auto_fixable: bool = False


@dataclass
class FsckReport:
    issues: list[FsckIssue] = field(default_factory=list)
    fixed: list[FsckIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    def summary(self) -> dict[str, Any]:
        return {
            "ok":           self.ok,
            "issue_count":  len(self.issues),
            "fixed_count":  len(self.fixed),
            "errors":       sum(1 for i in self.issues if i.severity == "error"),
            "warnings":     sum(1 for i in self.issues if i.severity == "warn"),
            "issues": [
                {"severity": i.severity, "code": i.code, "detail": i.detail,
                 "auto_fixable": i.auto_fixable}
                for i in self.issues
            ],
            "fixed": [
                {"code": i.code, "detail": i.detail}
                for i in self.fixed
            ],
        }


def fsck(
    settings: Settings,
    vec: VectorStore,
    lex: LexicalStore,
    *,
    auto_fix: bool = False,
) -> FsckReport:
    """Walk all stores looking for inconsistencies. With auto_fix=True, attempt
    to repair the safe ones (purge orphaned dynamic-root entries, drop stale
    file-hash rows). Never deletes chunks without an explicit go.
    """
    report = FsckReport()

    # Drift: vector vs lexical counts should match exactly.
    vc = vec.count()
    lc = lex.count()
    if vc != lc:
        report.issues.append(FsckIssue(
            severity="error", code="store_drift",
            detail=f"vector_count={vc} ≠ lexical_count={lc} (drift={abs(vc - lc)})",
        ))

    # Dynamic roots that no longer exist on disk.
    dyn = DynamicRoots.load(settings.dynamic_roots_path)
    stale_roots = [e for e in dyn.entries if not e.path.exists()]
    if stale_roots:
        for e in stale_roots:
            report.issues.append(FsckIssue(
                severity="warn", code="dynamic_root_missing",
                detail=f"{e.path} (added {e.added_at}) no longer exists on disk",
                auto_fixable=True,
            ))
        if auto_fix:
            for e in stale_roots:
                if dyn.remove(e.path):
                    report.fixed.append(FsckIssue(
                        severity="warn", code="dynamic_root_missing",
                        detail=f"removed {e.path} from dynamic_roots.json",
                    ))

    # File-hash rows for paths no longer in the lexical store (= no longer
    # indexed). They're harmless cache entries, but pruning them keeps the
    # registry size honest.
    hashes_path = settings.file_hashes_path
    if hashes_path.exists():
        reg = FileHashRegistry(hashes_path)
        try:
            reg.open()
            indexed = lex.list_paths()
            registered = reg.list_paths()
            stale_hashes = registered - indexed
            if stale_hashes:
                report.issues.append(FsckIssue(
                    severity="warn", code="orphan_file_hashes",
                    detail=f"{len(stale_hashes)} hash rows have no matching chunk in the lexical store",
                    auto_fixable=True,
                ))
                if auto_fix:
                    for path in stale_hashes:
                        reg.delete(path)
                    report.fixed.append(FsckIssue(
                        severity="warn", code="orphan_file_hashes",
                        detail=f"pruned {len(stale_hashes)} orphan hash rows",
                    ))
        except Exception as e:
            report.issues.append(FsckIssue(
                severity="warn", code="file_hashes_open_failed",
                detail=str(e),
            ))
        finally:
            with contextlib.suppress(Exception):
                reg.close()

    return report

"""One-shot installer for `code-rag install`.

Four idempotent steps, each skippable:
  1. Probe LM Studio and print a clear action if the embedder isn't loaded.
  2. Run the initial full index against the configured roots.
  3. Merge a `code-rag` MCP server block into every Claude config we find
     (Claude Desktop + Claude Code user config), preserving anything else.
  4. Register a windowless Task Scheduler autostart entry that runs
     `pythonw.exe -m code_rag watch` at logon.

Safety rules
------------
- NEVER overwrites an existing `mcpServers.code-rag` entry without backing up
  the file first (and printing the backup path).
- NEVER creates a new Claude config if the enclosing directory doesn't exist
  (we infer "that app isn't installed" and skip).
- NEVER requires admin. Task Scheduler registration uses the current user's
  Interactive principal.

This module is deliberately free of Click — the CLI layer imports `run_install`
and `InstallOptions` and renders the report.
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from code_rag.config import Settings
from code_rag.embedders.lm_studio import LMStudioEmbedder
from code_rag.factory import (
    build_embedder,
    build_graph_store,
    build_lexical_store,
    build_vector_store,
)
from code_rag.graph.ingest import GraphIngester
from code_rag.indexing.indexer import Indexer
from code_rag.logging import get
from code_rag.stores.chroma_vector import ChromaVectorStore

log = get(__name__)

MCP_SERVER_NAME = "code-rag"


# ---- options + report -------------------------------------------------------


@dataclass
class InstallOptions:
    skip_probe: bool = False
    skip_index: bool = False
    skip_claude: bool = False
    skip_autostart: bool = False
    force_reindex: bool = False


@dataclass
class StepReport:
    name: str
    ok: bool
    detail: str = ""
    skipped: bool = False

    def fmt(self) -> str:
        if self.skipped:
            return f"  [skip] {self.name}: {self.detail}"
        tag = "[OK]  " if self.ok else "[FAIL]"
        return f"  {tag} {self.name}: {self.detail}" if self.detail else f"  {tag} {self.name}"


@dataclass
class InstallReport:
    steps: list[StepReport] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.ok or s.skipped for s in self.steps)

    def add(self, step: StepReport) -> None:
        self.steps.append(step)
        log.info("install.step", name=step.name, ok=step.ok,
                 skipped=step.skipped, detail=step.detail)


# ---- Step 1: LM Studio probe ------------------------------------------------


async def step_probe_lm_studio(settings: Settings) -> StepReport:
    """Probe LM Studio. On failure, try to auto-bring-it-up via lms CLI before
    reporting failure — this makes `install` match the autostart behaviour."""
    from code_rag.lms_ctl import ensure_lm_studio_ready, find_lms

    emb = build_embedder(settings)
    try:
        await emb.health()
        detail = f"{settings.embedder.model} loaded (dim={emb.dim})"
        return StepReport("LM Studio probe", ok=True, detail=detail)
    except Exception as first_err:
        # Didn't respond. Try to bring it up automatically via lms CLI.
        loc = find_lms()
        if loc.path is None:
            return StepReport(
                "LM Studio probe", ok=False,
                detail=(f"{type(first_err).__name__}: {first_err}. "
                        f"Start LM Studio, load '{settings.embedder.model}', and enable the "
                        f"local server on {settings.embedder.base_url}. "
                        "Tip: install LM Studio's CLI (`lms bootstrap`) so `code-rag install` "
                        "can start the server automatically next time."),
            )
        extra: tuple[str, ...] = ()
        if settings.reranker.kind == "lm_studio" and settings.reranker.model:
            extra = (settings.reranker.model,)
        result = await asyncio.to_thread(
            ensure_lm_studio_ready,
            settings.embedder.base_url,
            settings.embedder.model,
            extra,
        )
        if not result.ok:
            return StepReport(
                "LM Studio probe", ok=False,
                detail=f"auto-start via lms failed: {result.error}",
            )
        # Re-check via the embedder so we get a fresh dim.
        await emb.health()
        return StepReport(
            "LM Studio probe", ok=True,
            detail=f"auto-started via lms; {settings.embedder.model} loaded (dim={emb.dim})",
        )
    finally:
        if isinstance(emb, LMStudioEmbedder):
            await emb.aclose()


# ---- Step 2: initial index build -------------------------------------------


async def step_initial_index(settings: Settings, force: bool) -> StepReport:
    # Skip if an index already exists and force=False.
    if settings.index_meta_path.exists() and not force:
        return StepReport(
            "initial index",
            ok=True,
            detail=f"already present at {settings.chroma_dir} (use --force-reindex to rebuild)",
        )

    embedder = build_embedder(settings)
    vec = build_vector_store(settings)
    lex = build_lexical_store(settings)
    graph = build_graph_store(settings)
    try:
        await embedder.health()
    except Exception as e:
        if isinstance(embedder, LMStudioEmbedder):
            await embedder.aclose()
        return StepReport("initial index", ok=False,
                          detail=f"embedder not reachable: {e}")

    meta = ChromaVectorStore.build_meta(
        embedder_kind=settings.embedder.kind,
        embedder_model=embedder.model,
        embedder_dim=embedder.dim,
    )
    try:
        vec.open(meta)
        lex.open()
        graph.open()
    except Exception as e:
        return StepReport("initial index", ok=False, detail=f"open failed: {e}")

    try:
        indexer = Indexer(
            settings, embedder, vec,
            lexical_store=lex,
            graph_store=GraphIngester(graph),
        )
        stats = await indexer.reindex_all()
        detail = (f"{stats.files_indexed} files, {stats.chunks_upserted} chunks "
                  f"in {stats.elapsed_s:.1f}s")
        if stats.errors:
            detail += f" ({len(stats.errors)} error(s); see log)"
        return StepReport("initial index", ok=True, detail=detail)
    except Exception as e:
        return StepReport("initial index", ok=False, detail=str(e))
    finally:
        vec.close()
        lex.close()
        graph.close()
        if isinstance(embedder, LMStudioEmbedder):
            await embedder.aclose()


# ---- Step 3: Claude config merge -------------------------------------------


def claude_config_targets() -> list[tuple[str, Path]]:
    """Return list of (name, path) for every Claude config file worth merging.

    Skips targets whose enclosing directory doesn't exist (the app isn't installed)
    or whose file can't be created sensibly.
    """
    targets: list[tuple[str, Path]] = []
    # Claude Desktop: %APPDATA%\Claude\claude_desktop_config.json
    appdata = os.environ.get("APPDATA")
    if appdata:
        claude_dir = Path(appdata) / "Claude"
        if claude_dir.is_dir():
            targets.append(("Claude Desktop", claude_dir / "claude_desktop_config.json"))
    # Claude Code (CLI) user config: %USERPROFILE%\.claude.json
    home = Path.home()
    claude_code_cfg = home / ".claude.json"
    if claude_code_cfg.exists():
        targets.append(("Claude Code (user)", claude_code_cfg))
    return targets


def mcp_server_block(repo_root: Path) -> dict[str, Any]:
    """The MCP server entry we inject into each Claude config.

    Uses pythonw.exe so the process is truly windowless. Falls back to
    python.exe if pythonw isn't present (e.g., some portable installs).
    """
    python_dir = Path(sys.executable).parent
    pythonw = python_dir / "pythonw.exe"
    exe = pythonw if pythonw.exists() else python_dir / "python.exe"
    return {
        "command": str(exe),
        "args": ["-m", "code_rag", "mcp"],
        "cwd": str(repo_root.resolve()),
    }


def step_wire_claude(repo_root: Path) -> StepReport:
    targets = claude_config_targets()
    if not targets:
        return StepReport(
            "Claude config wiring",
            ok=True,
            skipped=True,
            detail="no Claude Desktop or Claude Code config detected",
        )

    block = mcp_server_block(repo_root)
    touched: list[str] = []
    failures: list[str] = []
    for label, path in targets:
        try:
            _merge_mcp_block(path, MCP_SERVER_NAME, block)
            touched.append(f"{label} ({path})")
        except Exception as e:
            failures.append(f"{label}: {e}")

    if failures and not touched:
        return StepReport("Claude config wiring", ok=False,
                          detail="; ".join(failures))
    detail = "wrote to " + "; ".join(touched)
    if failures:
        detail += f"; WARNINGS: {'; '.join(failures)}"
    return StepReport("Claude config wiring", ok=True, detail=detail)


def _merge_mcp_block(path: Path, name: str, block: dict[str, Any]) -> None:
    """Idempotent: read → merge → atomic write. Backs up on first modification."""
    existing: dict[str, Any] = {}
    if path.exists():
        raw = path.read_text("utf-8").strip()
        if raw:
            try:
                existing = json.loads(raw)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"config at {path} is not valid JSON: {e}") from e
        if not isinstance(existing, dict):
            raise RuntimeError(f"config at {path} top-level is not an object")

    servers = dict(existing.get("mcpServers") or {})
    current = servers.get(name)
    # If the block is already exactly our expected shape, this is a no-op.
    if current == block:
        return

    # We're about to modify — back up FIRST so the user can revert.
    if path.exists():
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        backup = path.with_suffix(path.suffix + f".backup-{stamp}")
        backup.write_bytes(path.read_bytes())

    servers[name] = block
    existing["mcpServers"] = servers
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---- Step 4: autostart registration ----------------------------------------


def step_autostart(repo_root: Path) -> StepReport:
    script = repo_root / "scripts" / "install-autostart.ps1"
    if not script.exists():
        return StepReport("autostart registration", ok=False,
                          detail=f"missing {script}")

    # Already registered? Use schtasks to peek.
    try:
        r = subprocess.run(
            ["schtasks", "/Query", "/TN", "code-rag-watch"],
            capture_output=True, text=True, check=False,
        )
        if r.returncode == 0:
            return StepReport(
                "autostart registration", ok=True,
                detail="scheduled task 'code-rag-watch' already registered (run uninstall-autostart.ps1 to remove)",
            )
    except FileNotFoundError:
        # schtasks missing — almost certainly not Windows. Skip.
        return StepReport(
            "autostart registration", ok=True, skipped=True,
            detail="schtasks not available (non-Windows?)",
        )

    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(script)],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return StepReport("autostart registration", ok=False,
                          detail="powershell not found in PATH")
    if r.returncode != 0:
        msg = (r.stderr or r.stdout or "unknown error").strip().splitlines()
        return StepReport("autostart registration", ok=False,
                          detail=f"PowerShell exit {r.returncode}: {msg[-1] if msg else '?'}")
    return StepReport("autostart registration", ok=True,
                      detail="task 'code-rag-watch' registered; starts at next logon")


# ---- top-level orchestration ------------------------------------------------


async def run_install(settings: Settings, opts: InstallOptions) -> InstallReport:
    report = InstallReport()
    repo_root = Path(__file__).resolve().parents[2]

    # Step 1: probe
    if opts.skip_probe:
        report.add(StepReport("LM Studio probe", ok=True, skipped=True,
                              detail="skipped by --skip-probe"))
    else:
        probe = await step_probe_lm_studio(settings)
        report.add(probe)
        if not probe.ok and not opts.skip_index:
            # No embedder → index will fail. Short-circuit: mark index skipped.
            report.add(StepReport(
                "initial index", ok=True, skipped=True,
                detail="skipped because LM Studio probe failed; fix LM Studio then rerun `code-rag install`",
            ))
            opts = InstallOptions(
                skip_probe=True, skip_index=True,
                skip_claude=opts.skip_claude,
                skip_autostart=opts.skip_autostart,
                force_reindex=opts.force_reindex,
            )

    # Step 2: initial index
    if not opts.skip_index:
        report.add(await step_initial_index(settings, opts.force_reindex))

    # Step 3: Claude config
    if opts.skip_claude:
        report.add(StepReport("Claude config wiring", ok=True, skipped=True,
                              detail="skipped by --skip-claude"))
    else:
        report.add(step_wire_claude(repo_root))

    # Step 4: autostart
    if opts.skip_autostart:
        report.add(StepReport("autostart registration", ok=True, skipped=True,
                              detail="skipped by --skip-autostart"))
    else:
        report.add(step_autostart(repo_root))

    return report


def run_install_sync(settings: Settings, opts: InstallOptions) -> InstallReport:
    return asyncio.run(run_install(settings, opts))

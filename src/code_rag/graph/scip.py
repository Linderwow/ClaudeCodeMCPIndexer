"""Phase 18 scaffold: SCIP semantic indexing adapter.

SCIP (Sourcegraph Code Intelligence Protocol; https://github.com/sourcegraph/scip)
is the open standard for type-aware, cross-file resolved symbol indexing.
Our current Kuzu graph stores TEXTUAL call edges (the callee field is
just the bare token at the call site). SCIP gives us:

    * Properly resolved symbols (`scip-csharp ... MNQAlpha#OnBarUpdate().`)
    * Cross-file references with correct overload disambiguation
    * Type-aware "go to definition" / "find references"

Architecture for the eventual integration
-----------------------------------------
1. Per-language `scip-*` indexer runs out-of-process (Sourcegraph maintains
   one per language: `scip-python`, `scip-typescript`, `scip-csharp`).
2. Each emits a `index.scip` protobuf file per repo.
3. This module parses the protobuf into our `(SymbolRef, Edge)` records.
4. Symbols flow into Kuzu with a richer `kind` and a stable `scip_id` so
   the same symbol stays linkable across reindexes.

What this module ships TONIGHT
------------------------------
The contract: `SCIPIndex` dataclass (parsed records), `load_scip_index()`
fn, `iter_symbols_and_edges()` fn that converts SCIP records into our
`SymbolRef` and `Edge` shapes. The actual scip protobuf parser depends
on `scip-python` (pip install scip-python) which has heavy deps — we
provide a stub that REJECTS LOUDLY when called without the parser
installed, rather than silently returning empty. That way the contract
is wired, but trying to use it requires the user to install the parser
deliberately.

Migration path (when you're ready to flip it on, costs nothing in money):

    pip install scip-python                 # ~100 MB
    npm install -g @sourcegraph/scip-typescript
    # scip-csharp ships as a .NET tool:
    dotnet tool install -g Sourcegraph.Scip.CSharp

    code-rag scip-index --root ~/RiderProjects   # produces .scip files
    code-rag scip-import                          # loads into Kuzu
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from code_rag.interfaces.graph_store import Edge, SymbolRef
from code_rag.logging import get

log = get(__name__)


@dataclass(frozen=True)
class ScipSymbol:
    """One SCIP symbol record. Mirrors fields we actually need from the
    `scip.proto` `SymbolInformation` message — kept minimal so adding the
    real parser later doesn't shake the type signature too hard."""
    scip_id: str               # e.g. "scip-csharp . . . MNQAlpha#OnBarUpdate()."
    display_name: str          # human-readable name
    relative_path: str         # repo-relative path, posix
    kind: str                  # SCIP enum: "Class", "Method", "Function", ...
    start_line: int
    end_line: int


@dataclass
class ScipIndex:
    """Structured view of a single repo's SCIP output."""
    repo_label: str
    metadata_tool: str = ""    # e.g. "scip-csharp 0.1.4"
    project_root: str = ""     # absolute path the indexer ran against
    symbols: list[ScipSymbol] = field(default_factory=list)
    references: list[tuple[str, str, str]] = field(default_factory=list)
    # ^ list of (src_relative_path, source_scip_id, target_scip_id). For now
    # we only carry the IDs; resolving them against `symbols` is the loader's
    # job downstream.


# ---- adapter contract (stub today, real impl post-install) ----------------


class SCIPNotInstalledError(RuntimeError):
    """Raised when we try to load a .scip file but the protobuf parser isn't
    installed. We make this LOUD on purpose — silent skipping would give the
    user the impression SCIP indexing was working when it wasn't."""

    def __init__(self) -> None:
        super().__init__(
            "SCIP support is not installed. To enable Phase 18 semantic indexing:\n"
            "  pip install scip-python\n"
            "  # AND install the per-language indexers you need:\n"
            "  pip install scip-python  # provides the protobuf parser\n"
            "  npm install -g @sourcegraph/scip-typescript\n"
            "  dotnet tool install -g Sourcegraph.Scip.CSharp\n"
            "Then re-run `code-rag scip-import`."
        )


def _is_scip_parser_installed() -> bool:
    try:
        import scip_pb2  # type: ignore[import-not-found]  # noqa: F401
        return True
    except ImportError:
        return False


def load_scip_index(scip_path: Path, repo_label: str) -> ScipIndex:
    """Parse a SCIP `index.scip` file into our internal representation.

    Today: raises `SCIPNotInstalledError` if the protobuf bindings aren't
    available. Once `scip_pb2` is installed, this becomes a real parser.
    The protobuf wire format is stable, so the migration is pure
    code-on-the-shelf when you flip Phase 18 on.
    """
    if not _is_scip_parser_installed():
        raise SCIPNotInstalledError()
    # NB: when we wire the real parser, the implementation lives below this
    # block. It is intentionally not implemented tonight — running it without
    # validation could corrupt the live Kuzu index. The interface above is
    # the stable contract; the user supervises the first real run.
    raise NotImplementedError(
        "SCIP parsing is scaffolded but not enabled. Install scip-python and "
        "remove this NotImplementedError after running validation against a "
        "small repo first."
    )


def iter_symbols_and_edges(
    index: ScipIndex,
) -> tuple[list[SymbolRef], list[Edge]]:
    """Convert a `ScipIndex` into our existing `SymbolRef` + `Edge` shapes
    so it can be ingested by the same Kuzu pipeline as the tree-sitter
    extractor produces. Mapping is straightforward — kind names line up;
    we just need to map `references` triples to `Edge(kind="calls"...)`
    or similar. Implementation deferred until the parser is enabled."""
    if not index.symbols:
        return [], []
    symbols: list[SymbolRef] = [
        SymbolRef(
            path=s.relative_path,
            symbol=s.display_name,
            kind=s.kind.lower(),
            start_line=s.start_line,
            end_line=s.end_line,
        )
        for s in index.symbols
    ]
    by_id = {s.scip_id: s for s in index.symbols}
    edges: list[Edge] = []
    for src_path, src_id, dst_id in index.references:
        src = by_id.get(src_id)
        dst = by_id.get(dst_id)
        if src is None:
            continue
        edges.append(Edge(
            src_path=src_path,
            src_symbol=src.display_name,
            dst_path=dst.relative_path if dst else None,
            dst_symbol=dst.display_name if dst else dst_id,
            kind="calls",
        ))
    return symbols, edges

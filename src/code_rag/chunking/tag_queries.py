"""Phase 16: tag-query-based symbol extraction.

Tree-sitter ships a stable convention for "what's a symbol in this file":
each grammar's `queries/tags.scm` declares captures named
`@definition.<kind>` / `@reference.<kind>` / `@name`. Using those instead
of hand-maintained `_DECL_KIND` dicts means:

  1. Grammar updates auto-deliver new symbol kinds (e.g. TypeScript adding
     a new `decorator` definition) with no code change.
  2. Adding a new language is dropping in a `.scm` file — no Python edit.
  3. Captures already encode AST structure correctly, so we don't have to
     re-derive node-type-name → kind mappings per language.

This module is intentionally independent of the existing chunker. We ship
it ALONGSIDE the legacy `_SYMBOL_SPEC` so the cutover is gated by config
(`[chunker].use_tag_queries`, default false). After a clean reindex on
the new path validates parity, the legacy dicts can be deleted.

For C# — which does NOT ship a `tags.scm` upstream — we vendor our own at
`scripts/queries/c_sharp/tags.scm`. The runner falls back to that copy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tree_sitter import Language, Node, Parser, Query, QueryCursor

from code_rag.logging import get
from code_rag.models import ChunkKind

log = get(__name__)


# Map tag-query capture names to the ChunkKind we emit. The convention
# `@definition.<kind>` is a Sourcegraph SCIP / nvim-treesitter standard;
# we honor whatever a grammar's tags.scm uses and fall through to OTHER.
_CAPTURE_TO_KIND: dict[str, tuple[ChunkKind, bool]] = {
    # (ChunkKind, is_scope) — `is_scope` means this node contains nested
    # symbols whose names should be prefixed with this one's name.
    "definition.class":     (ChunkKind.CLASS,     True),
    "definition.struct":    (ChunkKind.STRUCT,    True),
    "definition.interface": (ChunkKind.INTERFACE, True),
    "definition.enum":      (ChunkKind.ENUM,      False),
    "definition.module":    (ChunkKind.NAMESPACE, True),
    "definition.namespace": (ChunkKind.NAMESPACE, True),
    "definition.function":  (ChunkKind.FUNCTION,  False),
    "definition.method":    (ChunkKind.METHOD,    False),
    "definition.constant":  (ChunkKind.OTHER,     False),
    "definition.variable":  (ChunkKind.OTHER,     False),
    "definition.macro":     (ChunkKind.OTHER,     False),
    "definition.type":      (ChunkKind.OTHER,     False),
}


def _vendor_dir() -> Path:
    """Local override directory for grammars that don't ship tags.scm or
    where we want a different version. Sits next to this module."""
    return Path(__file__).resolve().parent / "tag_queries_vendor"


def _grammar_query_path(language: str) -> Path | None:
    """Find a tags.scm for `language`. Order:
       1. Vendored override at `tag_queries_vendor/<lang>/tags.scm`
       2. Upstream package's `queries/tags.scm`
    Returns None if neither exists.
    """
    override = _vendor_dir() / language / "tags.scm"
    if override.exists():
        return override
    pkg_map = {
        "python":     "tree_sitter_python",
        "javascript": "tree_sitter_javascript",
        "typescript": "tree_sitter_typescript",
        "tsx":        "tree_sitter_typescript",
    }
    pkg_name = pkg_map.get(language)
    if pkg_name is None:
        return None
    try:
        import importlib
        pkg = importlib.import_module(pkg_name)
        for p in getattr(pkg, "__path__", []):
            cand = Path(p) / "queries" / "tags.scm"
            if cand.exists():
                return cand
    except Exception:
        return None
    return None


@dataclass
class TaggedSymbol:
    """One definition extracted from a file via tag queries."""
    name: str
    capture: str             # e.g. "definition.function"
    kind: ChunkKind
    is_scope: bool
    start_line: int          # 1-based
    end_line: int
    start_byte: int
    end_byte: int


class TagQueryRunner:
    """Loads a grammar's tags.scm and extracts TaggedSymbol's from a parsed
    tree. Stateless across files — one runner per (language, parser)."""

    def __init__(self, language_id: str, language: Language) -> None:
        self._language_id = language_id
        self._language = language
        self._parser = Parser(language)
        scm_path = _grammar_query_path(language_id)
        if scm_path is None:
            raise FileNotFoundError(
                f"no tags.scm available for language {language_id!r}; "
                f"add one at {_vendor_dir() / language_id / 'tags.scm'}"
            )
        scm = scm_path.read_text("utf-8")
        # Some upstream tags.scm use queries we don't need (highlights,
        # references). We only filter to definitions on extract; the parser
        # still loads everything.
        try:
            self._query = Query(language, scm)
        except Exception as e:  # pragma: no cover — bad scm is dev error
            raise RuntimeError(f"failed to compile tags.scm for {language_id}: {e}") from e

    @property
    def language_id(self) -> str:
        return self._language_id

    def extract(self, src: bytes) -> list[TaggedSymbol]:
        """Parse `src`, run the tag query, return all definition symbols.
        Reference captures (calls/imports/types) are NOT returned here —
        the chunker only cares about definitions; the graph extractor has
        its own pass for references."""
        tree = self._parser.parse(src)
        if tree is None or tree.root_node is None:
            return []

        out: list[TaggedSymbol] = []
        # tree-sitter 0.25 Query API: QueryCursor.matches(node).
        cursor = QueryCursor(self._query)
        try:
            matches = cursor.matches(tree.root_node)
        except AttributeError:
            # Older binding: query.captures
            matches = []
            captures = self._query.captures(tree.root_node)  # type: ignore[attr-defined]  # older tree-sitter only
            # `captures` returns dict[capture_name, list[Node]] in 0.24+.
            # Reshape into match-like records.
            if isinstance(captures, dict):
                # Group captures by their pattern_index using start_byte heuristics.
                # Simplest: produce one synthetic match per (definition node, name).
                for cap_name, nodes in captures.items():
                    if not cap_name.startswith("definition."):
                        continue
                    for n in nodes:
                        # Find a name capture that's a child of this node.
                        name_text = self._find_name(n, src)
                        if name_text is None:
                            continue
                        kind, is_scope = _CAPTURE_TO_KIND.get(
                            cap_name, (ChunkKind.OTHER, False),
                        )
                        out.append(TaggedSymbol(
                            name=name_text, capture=cap_name,
                            kind=kind, is_scope=is_scope,
                            start_line=n.start_point[0] + 1,
                            end_line=n.end_point[0] + 1,
                            start_byte=n.start_byte, end_byte=n.end_byte,
                        ))
                return out

        # New-API path: cursor.matches → list of (pattern_idx, dict[capture_name, list[Node]]).
        for _pattern_idx, caps in matches:
            def_node: Node | None = None
            def_capture: str | None = None
            name_node: Node | None = None
            for cap_name, nodes in caps.items():
                if not nodes:
                    continue
                if cap_name.startswith("definition."):
                    def_node = nodes[0]
                    def_capture = cap_name
                elif cap_name == "name":
                    name_node = nodes[0]
            if def_node is None or def_capture is None:
                continue
            if name_node is not None:
                name_text = src[name_node.start_byte:name_node.end_byte].decode("utf-8", "replace")
            else:
                fallback = self._find_name(def_node, src)
                if fallback is None:
                    continue
                name_text = fallback
            kind, is_scope = _CAPTURE_TO_KIND.get(def_capture, (ChunkKind.OTHER, False))
            out.append(TaggedSymbol(
                name=name_text, capture=def_capture,
                kind=kind, is_scope=is_scope,
                start_line=def_node.start_point[0] + 1,
                end_line=def_node.end_point[0] + 1,
                start_byte=def_node.start_byte, end_byte=def_node.end_byte,
            ))
        return out

    @staticmethod
    def _find_name(node: Node, src: bytes) -> str | None:
        """Last-resort name extraction: look for a `name` field, otherwise
        the first identifier-like child."""
        n = node.child_by_field_name("name")
        if n is not None:
            return src[n.start_byte:n.end_byte].decode("utf-8", "replace")
        for c in node.named_children:
            if c.type in ("identifier", "type_identifier", "property_identifier"):
                return src[c.start_byte:c.end_byte].decode("utf-8", "replace")
        return None

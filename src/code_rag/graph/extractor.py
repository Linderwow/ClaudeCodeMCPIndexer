from __future__ import annotations

from pathlib import Path

from tree_sitter import Node, Parser

from code_rag.chunking.treesitter import _load_languages  # reuse loader
from code_rag.interfaces.graph_store import Edge, SymbolRef
from code_rag.logging import get

log = get(__name__)


# Per-language map of declaration-node-type -> (symbol_kind, is_scope).
# Mirrors the chunker spec but expressed in terms the extractor needs.
_TS_DECL: dict[str, tuple[str, bool]] = {
    "class_declaration":              ("class",     True),
    "abstract_class_declaration":     ("class",     True),
    "interface_declaration":          ("interface", True),
    "function_declaration":           ("function",  False),
    "generator_function_declaration": ("function",  False),
    "method_definition":              ("method",    False),
    "abstract_method_signature":      ("method",    False),
    "method_signature":               ("method",    False),
    "enum_declaration":               ("enum",      False),
    "type_alias_declaration":         ("type",      False),
    "module":                         ("namespace", True),
    "internal_module":                ("namespace", True),
}
_JS_DECL: dict[str, tuple[str, bool]] = {
    "class_declaration":              ("class",    True),
    "function_declaration":           ("function", False),
    "generator_function_declaration": ("function", False),
    "method_definition":              ("method",   False),
}
_DECL_KIND: dict[str, dict[str, tuple[str, bool]]] = {
    "python": {
        "function_definition":       ("function", False),
        "async_function_definition": ("function", False),
        "class_definition":          ("class", True),
    },
    "c_sharp": {
        "namespace_declaration":             ("namespace", True),
        "file_scoped_namespace_declaration": ("namespace", True),
        "class_declaration":                 ("class",     True),
        "struct_declaration":                ("struct",    True),
        "record_declaration":                ("class",     True),
        "record_struct_declaration":         ("struct",    True),
        "interface_declaration":             ("interface", True),
        "enum_declaration":                  ("enum",      False),
        "method_declaration":                ("method",    False),
        "constructor_declaration":           ("method",    False),
        "destructor_declaration":            ("method",    False),
        "operator_declaration":              ("method",    False),
        "property_declaration":              ("method",    False),
        "delegate_declaration":              ("function",  False),
    },
    "typescript": _TS_DECL,
    "tsx":        _TS_DECL,
    "javascript": _JS_DECL,
}

# Call-site node types — we treat the node's first-identifier descendant as the
# callee name. It's coarse (resolves only to a name, not a target symbol), but
# good enough for "show me who calls `OnBarUpdate`" queries, which is 90% of
# what developers ask. Fully-qualified resolution is an eventual upgrade path.
_CALL_NODE: dict[str, tuple[str, ...]] = {
    "python":     ("call",),
    "c_sharp":    ("invocation_expression",),
    "typescript": ("call_expression",),
    "tsx":        ("call_expression",),
    "javascript": ("call_expression",),
}

# Import nodes.
_IMPORT_NODE: dict[str, tuple[str, ...]] = {
    "python":     ("import_statement", "import_from_statement"),
    "c_sharp":    ("using_directive",),
    "typescript": ("import_statement",),
    "tsx":        ("import_statement",),
    "javascript": ("import_statement",),
}


class GraphExtractor:
    """Extracts SymbolRefs + Edges from source using tree-sitter.

    Focus is breadth (names, counts, neighborhoods), not full semantic
    resolution. A 'calls' edge's dst_symbol is the TEXTUAL callee name at the
    call site; we don't try to resolve overloads or generic dispatch.
    """

    def __init__(self) -> None:
        # Phase 37 audit fix: per-call Parser. See chunking/treesitter.py for
        # context — tree-sitter Parser is not reentrant; sharing across
        # async-to-thread workers can corrupt the tree or segfault.
        self._langs = _load_languages()

    def extract(
        self, abs_path: Path, rel_path: str, language: str,
    ) -> tuple[list[SymbolRef], list[Edge]]:
        if language not in self._langs:
            return [], []
        try:
            src = abs_path.read_bytes()
        except OSError as e:
            log.warning("graph.read_fail", path=rel_path, err=str(e))
            return [], []
        if not src.strip():
            return [], []

        # Per-call Parser — see __init__ comment.
        parser = Parser(self._langs[language])
        tree = parser.parse(src)
        if tree is None:
            return [], []

        symbols: list[SymbolRef] = []
        edges: list[Edge] = []
        self._walk(
            tree.root_node, src, language,
            parent=None, out_symbols=symbols, out_edges=edges, rel_path=rel_path,
        )
        return symbols, edges

    # ---- walk ---------------------------------------------------------------

    def _walk(
        self,
        node: Node,
        src: bytes,
        language: str,
        parent: str | None,
        out_symbols: list[SymbolRef],
        out_edges: list[Edge],
        *,
        rel_path: str,
    ) -> None:
        decl_spec = _DECL_KIND[language]
        call_types = _CALL_NODE[language]
        import_types = _IMPORT_NODE[language]

        for child in node.named_children:
            t = child.type
            if t in decl_spec:
                kind, is_scope = decl_spec[t]
                name = _name_of(child, src) or f"<anon:{t}>"
                full = f"{parent}.{name}" if parent else name
                out_symbols.append(SymbolRef(
                    path=rel_path,
                    symbol=full,
                    kind=kind,
                    start_line=child.start_point[0] + 1,
                    end_line=child.end_point[0] + 1,
                ))
                if parent is not None:
                    out_edges.append(Edge(
                        src_path=rel_path, src_symbol=parent,
                        dst_path=rel_path, dst_symbol=full,
                        kind="contains",
                    ))
                new_parent = full if is_scope or kind in ("method", "function") else parent
                self._walk(child, src, language, new_parent, out_symbols, out_edges,
                           rel_path=rel_path)

            elif t in call_types and parent is not None:
                callee = _call_target(child, src, language)
                if callee:
                    out_edges.append(Edge(
                        src_path=rel_path, src_symbol=parent,
                        dst_path=None, dst_symbol=callee,
                        kind="calls",
                    ))
                # Continue walking so nested calls inside an arg expression count.
                self._walk(child, src, language, parent, out_symbols, out_edges,
                           rel_path=rel_path)

            elif t in import_types:
                imp = _import_target(child, src, language)
                if imp:
                    out_edges.append(Edge(
                        src_path=rel_path,
                        src_symbol=parent or "<module>",
                        dst_path=None,
                        dst_symbol=imp,
                        kind="imports",
                    ))
                # No recursion into import nodes.

            else:
                self._walk(child, src, language, parent, out_symbols, out_edges,
                           rel_path=rel_path)


# ---- helpers ---------------------------------------------------------------


def _name_of(node: Node, src: bytes) -> str | None:
    n = node.child_by_field_name("name")
    if n is not None:
        return src[n.start_byte : n.end_byte].decode("utf-8", "replace")
    for c in node.named_children:
        if c.type in ("identifier", "qualified_name", "type_identifier"):
            return src[c.start_byte : c.end_byte].decode("utf-8", "replace")
    return None


def _call_target(call_node: Node, src: bytes, language: str) -> str | None:
    """Return the bare callee name at a call site.

    Python `call` has a `function` field; C# `invocation_expression` has a
    `function` field too (member_access_expression or identifier).
    """
    fn = call_node.child_by_field_name("function")
    if fn is None:
        fn = next(iter(call_node.named_children), None)
    if fn is None:
        return None
    # Take the last identifier segment — for `self.foo.bar(...)` return "bar".
    text = src[fn.start_byte : fn.end_byte].decode("utf-8", "replace")
    # Split on `.` and `->`, take the tail, strip generic arguments and brackets.
    tail = text.replace("->", ".").split(".")[-1]
    tail = tail.split("<", 1)[0].split("[", 1)[0].strip()
    return tail or None


def _import_target(imp_node: Node, src: bytes, language: str) -> str | None:
    text = src[imp_node.start_byte : imp_node.end_byte].decode("utf-8", "replace")
    if language == "python":
        # "import x.y" / "from x.y import z" — normalize to the module path.
        t = text.strip().removeprefix("from ").removeprefix("import ").split(";", 1)[0].strip()
        return t.split(" ", 1)[0] or None
    if language == "c_sharp":
        # "using N.M;" or "using alias = N.M;"
        t = text.strip().removeprefix("using ").rstrip(";").strip()
        return t or None
    if language in ("typescript", "tsx", "javascript"):
        # TS/JS import specifier is a string literal after `from`, or a bare
        # string (side-effect import). Pull the quoted specifier directly off
        # the AST — more robust than the varied import clause shapes.
        src_node = imp_node.child_by_field_name("source")
        if src_node is not None:
            raw = src[src_node.start_byte : src_node.end_byte].decode("utf-8", "replace")
            return raw.strip().strip("'\"`") or None
        # Fallback: last string in the node text.
        chunks = text.rsplit("'", 2)
        if len(chunks) >= 3:
            return chunks[-2] or None
        chunks = text.rsplit('"', 2)
        if len(chunks) >= 3:
            return chunks[-2] or None
        return None
    return None

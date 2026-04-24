from __future__ import annotations

from pathlib import Path

from tree_sitter import Language, Node, Parser

from code_rag.logging import get
from code_rag.models import Chunk, ChunkKind
from code_rag.util.hashing import chunk_id

log = get(__name__)


def _load_languages() -> dict[str, Language]:
    import tree_sitter_c_sharp as tscs
    import tree_sitter_javascript as tsjs
    import tree_sitter_python as tspy
    import tree_sitter_typescript as tsts
    return {
        "python":     Language(tspy.language()),
        "c_sharp":    Language(tscs.language()),
        "typescript": Language(tsts.language_typescript()),
        "tsx":        Language(tsts.language_tsx()),
        "javascript": Language(tsjs.language()),
    }


# Symbol-bearing node types per language, with the ChunkKind to emit.
# "scope" means the node contains nested symbol-bearing children we should recurse into
# while prefixing their symbol names with the outer node's name.
_TS_SPEC: dict[str, tuple[ChunkKind, bool]] = {
    "class_declaration":          (ChunkKind.CLASS,     True),
    "abstract_class_declaration": (ChunkKind.CLASS,     True),
    "interface_declaration":      (ChunkKind.INTERFACE, True),
    "function_declaration":       (ChunkKind.FUNCTION,  False),
    "generator_function_declaration": (ChunkKind.FUNCTION, False),
    "method_definition":          (ChunkKind.METHOD,    False),
    "abstract_method_signature":  (ChunkKind.METHOD,    False),
    "method_signature":           (ChunkKind.METHOD,    False),
    "enum_declaration":           (ChunkKind.ENUM,      False),
    "type_alias_declaration":     (ChunkKind.OTHER,     False),
    "module":                     (ChunkKind.NAMESPACE, True),
    "internal_module":            (ChunkKind.NAMESPACE, True),
}
_JS_SPEC: dict[str, tuple[ChunkKind, bool]] = {
    "class_declaration":              (ChunkKind.CLASS,    True),
    "function_declaration":           (ChunkKind.FUNCTION, False),
    "generator_function_declaration": (ChunkKind.FUNCTION, False),
    "method_definition":              (ChunkKind.METHOD,   False),
}
_SYMBOL_SPEC: dict[str, dict[str, tuple[ChunkKind, bool]]] = {
    "python": {
        "function_definition":       (ChunkKind.FUNCTION, False),
        "async_function_definition": (ChunkKind.FUNCTION, False),
        "class_definition":          (ChunkKind.CLASS,    True),
    },
    "c_sharp": {
        "namespace_declaration":     (ChunkKind.NAMESPACE, True),
        "file_scoped_namespace_declaration": (ChunkKind.NAMESPACE, True),
        "class_declaration":         (ChunkKind.CLASS,     True),
        "struct_declaration":        (ChunkKind.STRUCT,    True),
        "record_declaration":        (ChunkKind.CLASS,     True),
        "record_struct_declaration": (ChunkKind.STRUCT,    True),
        "interface_declaration":     (ChunkKind.INTERFACE, True),
        "enum_declaration":          (ChunkKind.ENUM,      False),
        "method_declaration":        (ChunkKind.METHOD,    False),
        "constructor_declaration":   (ChunkKind.METHOD,    False),
        "destructor_declaration":    (ChunkKind.METHOD,    False),
        "operator_declaration":      (ChunkKind.METHOD,    False),
        "property_declaration":      (ChunkKind.METHOD,    False),
        "delegate_declaration":      (ChunkKind.FUNCTION,  False),
    },
    "typescript": _TS_SPEC,
    "tsx":        _TS_SPEC,   # TSX shares the TS declaration grammar
    "javascript": _JS_SPEC,
}


class TreeSitterChunker:
    """AST-aware chunker for Python and C#.

    Emits one Chunk per symbol-bearing node, with hierarchical names like
    "MNQAlpha.OnBarUpdate". Falls back to whole-file windowing if parsing
    yields no symbols (rare — happens for top-level scripts or parse failures).
    """

    def __init__(self, min_chars: int, max_chars: int) -> None:
        self._min = min_chars
        self._max = max_chars
        self._langs = _load_languages()
        # tree-sitter 0.25 Parser takes Language in constructor, but also supports .language = ...
        self._parsers: dict[str, Parser] = {k: Parser(v) for k, v in self._langs.items()}

    # ---- public entry -------------------------------------------------------

    def chunk_file(self, repo: str, abs_path: Path, rel_path: str, language: str) -> list[Chunk]:
        if language not in self._parsers:
            return []
        try:
            src = abs_path.read_bytes()
        except OSError as e:
            log.warning("chunker.read_fail", path=rel_path, err=str(e))
            return []
        if not src.strip():
            return []

        tree = self._parsers[language].parse(src)
        if tree is None or tree.root_node is None:
            return self._fallback(repo, rel_path, language, src)

        chunks: list[Chunk] = []
        self._walk(tree.root_node, src, language, parent=None, out=chunks,
                   repo=repo, rel_path=rel_path)

        if not chunks:
            return self._fallback(repo, rel_path, language, src)
        return chunks

    # ---- AST walk -----------------------------------------------------------

    def _walk(
        self,
        node: Node,
        src: bytes,
        language: str,
        parent: str | None,
        out: list[Chunk],
        *,
        repo: str,
        rel_path: str,
    ) -> None:
        spec = _SYMBOL_SPEC[language]
        for child in node.named_children:
            t = child.type
            if t in spec:
                kind, is_scope = spec[t]
                name = self._symbol_name(child, src) or f"<anon:{t}>"
                full = f"{parent}.{name}" if parent else name
                text = self._clip(src[child.start_byte : child.end_byte].decode("utf-8", "replace"))
                start_line = child.start_point[0] + 1
                if len(text) >= self._min:
                    out.append(Chunk(
                        id=chunk_id(repo, rel_path, full, text, start_line),
                        repo=repo,
                        path=rel_path,
                        language=language,
                        symbol=full,
                        kind=kind,
                        start_line=start_line,
                        end_line=child.end_point[0] + 1,
                        text=text,
                    ))
                # Recurse into scope-bearing nodes with an updated parent prefix.
                if is_scope:
                    self._walk(child, src, language, parent=full, out=out,
                               repo=repo, rel_path=rel_path)
                # Into non-scope nodes (e.g., a method body) we also recurse so nested
                # functions/classes are captured, but parent stays the same.
                else:
                    self._walk(child, src, language, parent=parent, out=out,
                               repo=repo, rel_path=rel_path)
            else:
                self._walk(child, src, language, parent=parent, out=out,
                           repo=repo, rel_path=rel_path)

    # ---- helpers ------------------------------------------------------------

    @staticmethod
    def _symbol_name(node: Node, src: bytes) -> str | None:
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            return src[name_node.start_byte : name_node.end_byte].decode("utf-8", "replace")
        # Fallback: first identifier child.
        for c in node.named_children:
            if c.type in ("identifier", "qualified_name", "type_identifier"):
                return src[c.start_byte : c.end_byte].decode("utf-8", "replace")
        return None

    def _clip(self, text: str) -> str:
        """Hard cap to 2x max_chars with a truncation marker.

        Very large methods still produce ONE chunk (we don't silently drop
        content), but we don't send 50k chars to the embedder either.
        """
        limit = self._max * 2
        if len(text) <= limit:
            return text
        head = text[: limit - 64]
        return head + "\n// ... (truncated by chunker) ...\n"

    def _fallback(self, repo: str, rel_path: str, language: str, src: bytes) -> list[Chunk]:
        """Line-aware windowing when the parser yields no symbols."""
        text = src.decode("utf-8", "replace")
        if len(text) < self._min:
            return []
        lines = text.splitlines(keepends=True)
        out: list[Chunk] = []
        buf: list[str] = []
        buf_len = 0
        start_line = 1
        cur_line = 1
        for ln in lines:
            if buf_len + len(ln) > self._max and buf:
                chunk_text = "".join(buf)
                window_sym = f"window:{start_line}"
                out.append(Chunk(
                    id=chunk_id(repo, rel_path, window_sym, chunk_text, start_line),
                    repo=repo,
                    path=rel_path,
                    language=language,
                    symbol=window_sym,
                    kind=ChunkKind.OTHER,
                    start_line=start_line,
                    end_line=cur_line - 1,
                    text=chunk_text,
                ))
                buf = []
                buf_len = 0
                start_line = cur_line
            buf.append(ln)
            buf_len += len(ln)
            cur_line += 1
        if buf and buf_len >= self._min:
            chunk_text = "".join(buf)
            window_sym = f"window:{start_line}"
            out.append(Chunk(
                id=chunk_id(repo, rel_path, window_sym, chunk_text, start_line),
                repo=repo,
                path=rel_path,
                language=language,
                symbol=window_sym,
                kind=ChunkKind.OTHER,
                start_line=start_line,
                end_line=cur_line - 1,
                text=chunk_text,
            ))
        return out

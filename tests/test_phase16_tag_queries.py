"""Phase 16: tag-query-based symbol extraction parallel to the legacy
hand-written `_SYMBOL_SPEC` dicts. The runner is opt-in; until parity is
demonstrated on a real reindex it ships disabled by default."""
from __future__ import annotations

import pytest
from tree_sitter import Language

from code_rag.chunking.tag_queries import (
    TagQueryRunner,
    _grammar_query_path,
)
from code_rag.models import ChunkKind


def _load_lang(language_id: str) -> Language:
    """Produce a `Language` for `language_id` using the same loader the
    chunker uses. Centralized so we don't drift."""
    if language_id == "python":
        import tree_sitter_python as ts
        return Language(ts.language())
    if language_id == "javascript":
        import tree_sitter_javascript as ts
        return Language(ts.language())
    if language_id == "typescript":
        import tree_sitter_typescript as ts
        return Language(ts.language_typescript())
    if language_id == "c_sharp":
        import tree_sitter_c_sharp as ts
        return Language(ts.language())
    raise ValueError(language_id)


# ---- vendored / packaged file presence -------------------------------------


def test_python_uses_upstream_tags_scm() -> None:
    p = _grammar_query_path("python")
    assert p is not None and p.exists()
    text = p.read_text("utf-8")
    assert "definition.function" in text
    assert "definition.class" in text


def test_typescript_uses_upstream_tags_scm() -> None:
    p = _grammar_query_path("typescript")
    assert p is not None and p.exists()
    text = p.read_text("utf-8")
    assert "definition.method" in text or "definition.function" in text


def test_c_sharp_uses_vendored_tags_scm() -> None:
    """C# upstream doesn't ship tags.scm — we vendor our own. Asserting it's
    found at the expected path keeps the override working when the package
    happens to add one upstream later (vendor wins on tie)."""
    p = _grammar_query_path("c_sharp")
    assert p is not None and p.exists()
    assert "tag_queries_vendor" in str(p)


# ---- extraction ------------------------------------------------------------


def test_python_extraction_finds_class_and_methods() -> None:
    runner = TagQueryRunner("python", _load_lang("python"))
    src = b"""\
def top():
    return 1

class Foo:
    def bar(self):
        return 2

    def baz(self):
        return 3
"""
    syms = runner.extract(src)
    names = {s.name for s in syms}
    assert "top" in names
    assert "Foo" in names
    assert "bar" in names
    assert "baz" in names
    # Class's is_scope flag — the chunker uses this to namespace nested members.
    foo = next(s for s in syms if s.name == "Foo")
    assert foo.is_scope is True
    assert foo.kind == ChunkKind.CLASS


def test_javascript_extraction_finds_classes_and_methods() -> None:
    runner = TagQueryRunner("javascript", _load_lang("javascript"))
    src = b"""\
function hello(n) { return 'Hi ' + n; }

class Counter {
    constructor() { this.n = 0; }
    inc() { this.n += 1; }
    dec() { this.n -= 1; }
}
"""
    syms = runner.extract(src)
    names = {s.name for s in syms}
    assert "hello" in names
    assert "Counter" in names
    assert "inc" in names
    assert "dec" in names


def test_typescript_extraction_finds_interfaces() -> None:
    runner = TagQueryRunner("typescript", _load_lang("typescript"))
    src = b"""\
export interface User {
    id: string;
    name: string;
}

export class UserService {
    getById(id: string): User | undefined { return undefined; }
}
"""
    syms = runner.extract(src)
    names = {s.name for s in syms}
    kinds = {s.kind for s in syms}
    assert "User" in names
    assert "UserService" in names
    assert ChunkKind.INTERFACE in kinds
    assert ChunkKind.CLASS in kinds


def test_csharp_extraction_finds_class_and_methods() -> None:
    runner = TagQueryRunner("c_sharp", _load_lang("c_sharp"))
    src = b"""\
namespace NinjaTrader.Strategies
{
    public class Demo
    {
        public void OnBarUpdate() { }
        public void Helper() { }
    }
}
"""
    syms = runner.extract(src)
    names = {s.name for s in syms}
    # Vendored .scm yields namespace + class + each method.
    assert "Demo" in names
    assert "OnBarUpdate" in names
    assert "Helper" in names
    # namespace `NinjaTrader.Strategies` is a qualified_name; runner picks
    # whichever name node the .scm captures — could be the full qualified
    # name or just `Strategies`. Either is fine; assert SOMETHING captured.
    assert any(s.kind == ChunkKind.NAMESPACE for s in syms)


def test_runner_handles_unparseable_gracefully() -> None:
    """tree-sitter is error-tolerant: even malformed source produces a tree.
    Runner shouldn't crash on weird input."""
    runner = TagQueryRunner("python", _load_lang("python"))
    src = b"def broken( syntax error here\n"
    # Should not raise
    out = runner.extract(src)
    # Could be empty or partial — both are acceptable.
    assert isinstance(out, list)


def test_unknown_language_raises_clearly() -> None:
    """If we ever ask for a language with no tags.scm anywhere, the runner
    fails fast with a friendly message pointing at the override directory."""
    # Force a missing path by passing an invented id.
    fake_lang = _load_lang("python")  # any real Language object will do
    with pytest.raises(FileNotFoundError, match=r"no tags\.scm"):
        TagQueryRunner("ocaml", fake_lang)

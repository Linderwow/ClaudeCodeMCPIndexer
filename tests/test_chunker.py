from __future__ import annotations

from pathlib import Path

from code_rag.chunking.treesitter import TreeSitterChunker
from code_rag.models import ChunkKind


def test_python_chunker_emits_class_and_methods(tmp_path: Path) -> None:
    p = tmp_path / "m.py"
    p.write_text(
        '''\
"""module doc"""

def top_level():
    return 1

class Foo:
    """class doc"""

    def bar(self):
        return 2

    def baz(self):
        return 3
''',
        encoding="utf-8",
    )
    ch = TreeSitterChunker(min_chars=10, max_chars=1000)
    chunks = ch.chunk_file("repo", p, "m.py", "python")
    symbols = {c.symbol for c in chunks}
    kinds = {c.kind for c in chunks}

    assert "top_level" in symbols
    assert "Foo" in symbols
    assert "Foo.bar" in symbols
    assert "Foo.baz" in symbols
    assert ChunkKind.FUNCTION in kinds
    assert ChunkKind.CLASS in kinds


def test_csharp_chunker_emits_class_and_methods(tmp_path: Path) -> None:
    p = tmp_path / "S.cs"
    p.write_text(
        """\
namespace NinjaTrader.Strategies
{
    public class Demo
    {
        public int Field;

        public Demo() { }

        public void OnBarUpdate() { Field = 1; }

        public void OtherMethod() { Field = 2; }
    }
}
""",
        encoding="utf-8",
    )
    ch = TreeSitterChunker(min_chars=10, max_chars=1000)
    chunks = ch.chunk_file("repo", p, "S.cs", "c_sharp")
    symbols = {c.symbol for c in chunks}

    assert "NinjaTrader" in str(symbols) or "NinjaTrader.Strategies" in symbols
    assert any(s and "Demo" in s for s in symbols)
    assert any(s and "OnBarUpdate" in s for s in symbols)
    assert any(s and "OtherMethod" in s for s in symbols)


def test_chunker_ids_are_content_addressed(tmp_path: Path) -> None:
    p = tmp_path / "a.py"
    p.write_text("def f():\n    return 1\n", encoding="utf-8")
    ch = TreeSitterChunker(min_chars=1, max_chars=1000)
    a = ch.chunk_file("repo", p, "a.py", "python")
    b = ch.chunk_file("repo", p, "a.py", "python")
    assert [c.id for c in a] == [c.id for c in b]

    # Edit the function body: ids must change (new content).
    p.write_text("def f():\n    return 2\n", encoding="utf-8")
    c = ch.chunk_file("repo", p, "a.py", "python")
    assert [x.id for x in c] != [x.id for x in a]


def test_fallback_windowing_on_unparseable(tmp_path: Path) -> None:
    # No known symbol-bearing nodes — but parser still produces a tree.
    # Using a text blob with no functions/classes triggers the fallback path.
    p = tmp_path / "notes.py"
    p.write_text("x = 1\n" * 500, encoding="utf-8")
    ch = TreeSitterChunker(min_chars=20, max_chars=200)
    chunks = ch.chunk_file("repo", p, "notes.py", "python")
    assert chunks, "fallback chunker should emit at least one window"
    assert all(c.symbol and c.symbol.startswith("window:") for c in chunks)


def test_typescript_chunker_emits_classes_methods_and_interfaces(tmp_path: Path) -> None:
    p = tmp_path / "m.ts"
    p.write_text(
        """\
export interface User {
    id: string;
    name: string;
}

export function greet(u: User): string {
    return `Hi ${u.name}`;
}

export class UserService {
    private readonly _cache: Map<string, User> = new Map();

    getById(id: string): User | undefined {
        return this._cache.get(id);
    }

    upsert(u: User): void {
        this._cache.set(u.id, u);
    }
}

export type UserId = string;

export enum Role { Admin, User }
""",
        encoding="utf-8",
    )
    ch = TreeSitterChunker(min_chars=10, max_chars=2000)
    chunks = ch.chunk_file("repo", p, "m.ts", "typescript")
    symbols = {c.symbol for c in chunks}
    kinds = {c.kind for c in chunks}

    assert "User" in symbols
    assert "greet" in symbols
    assert "UserService" in symbols
    assert "UserService.getById" in symbols
    assert "UserService.upsert" in symbols
    assert "Role" in symbols
    assert ChunkKind.INTERFACE in kinds
    assert ChunkKind.CLASS in kinds
    assert ChunkKind.METHOD in kinds
    assert ChunkKind.FUNCTION in kinds
    assert ChunkKind.ENUM in kinds


def test_tsx_chunker_handles_jsx(tmp_path: Path) -> None:
    p = tmp_path / "App.tsx"
    p.write_text(
        """\
import React from 'react';

interface Props { label: string; }

export function Button(props: Props) {
    return <button>{props.label}</button>;
}

export class Panel extends React.Component<Props> {
    render() { return <div><Button label="go" /></div>; }
}
""",
        encoding="utf-8",
    )
    ch = TreeSitterChunker(min_chars=10, max_chars=2000)
    chunks = ch.chunk_file("repo", p, "App.tsx", "tsx")
    symbols = {c.symbol for c in chunks}
    assert "Props" in symbols
    assert "Button" in symbols
    assert "Panel" in symbols
    assert "Panel.render" in symbols


def test_duplicate_function_in_one_file_gets_unique_ids(tmp_path: Path) -> None:
    """Real-world bug: a JS file copy-pastes the same `pad()` definition three
    times. Old chunk_id formula used `path|symbol|text` only, so all three
    collided to the same id and Chroma's upsert rejected the batch with
    DuplicateIDError. start_line is now part of the hash so identical content
    at different positions gets unique ids."""
    p = tmp_path / "dup.js"
    p.write_text(
        """\
function before() { return 1; }

function pad(n) {
    return ('0000' + n).slice(-4);
}

function middle() { return 2; }

function pad(n) {
    return ('0000' + n).slice(-4);
}

function after() { return 3; }

function pad(n) {
    return ('0000' + n).slice(-4);
}
""",
        encoding="utf-8",
    )
    ch = TreeSitterChunker(min_chars=10, max_chars=2000)
    chunks = ch.chunk_file("repo", p, "dup.js", "javascript")
    pad_chunks = [c for c in chunks if c.symbol == "pad"]
    assert len(pad_chunks) == 3, f"expected 3 pad chunks, got {len(pad_chunks)}"
    ids = {c.id for c in pad_chunks}
    assert len(ids) == 3, f"all pad chunks should have unique ids, got {ids}"
    # Sanity: same content but different start_line is what makes them unique.
    assert {c.text for c in pad_chunks} == {pad_chunks[0].text}
    assert len({c.start_line for c in pad_chunks}) == 3


def test_javascript_chunker_emits_classes_and_functions(tmp_path: Path) -> None:
    p = tmp_path / "m.js"
    p.write_text(
        """\
function hello(n) {
    return 'Hi ' + n;
}

class Counter {
    constructor() { this.n = 0; }
    inc() { this.n += 1; }
    dec() { this.n -= 1; }
}

export { hello, Counter };
""",
        encoding="utf-8",
    )
    ch = TreeSitterChunker(min_chars=10, max_chars=2000)
    chunks = ch.chunk_file("repo", p, "m.js", "javascript")
    symbols = {c.symbol for c in chunks}
    assert "hello" in symbols
    assert "Counter" in symbols
    assert "Counter.inc" in symbols
    assert "Counter.dec" in symbols

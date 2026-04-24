"""Phase 3: Kuzu graph store + extractor end-to-end."""
from __future__ import annotations

from pathlib import Path

from code_rag.graph.extractor import GraphExtractor
from code_rag.graph.ingest import GraphIngester
from code_rag.stores.kuzu_graph import KuzuGraphStore


def test_extractor_finds_python_calls(tmp_path: Path) -> None:
    p = tmp_path / "m.py"
    p.write_text(
        """\
def helper():
    return 1

def top():
    x = helper()
    y = helper() + 1
    return x + y
""",
        encoding="utf-8",
    )
    ex = GraphExtractor()
    symbols, edges = ex.extract(p, "m.py", "python")
    syms = {s.symbol for s in symbols}
    assert {"helper", "top"}.issubset(syms)
    call_edges = [e for e in edges if e.kind == "calls" and e.src_symbol == "top"]
    assert any(e.dst_symbol == "helper" for e in call_edges), f"edges: {edges}"


def test_extractor_finds_csharp_calls(tmp_path: Path) -> None:
    p = tmp_path / "s.cs"
    p.write_text(
        """\
namespace N {
    public class K {
        public void A() { B(); }
        public void B() { }
    }
}
""",
        encoding="utf-8",
    )
    ex = GraphExtractor()
    _syms, edges = ex.extract(p, "s.cs", "c_sharp")
    calls = [e for e in edges if e.kind == "calls"]
    assert any(e.src_symbol.endswith("A") and e.dst_symbol == "B" for e in calls), f"{calls}"


def test_kuzu_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "m.py"
    p.write_text(
        """\
def helper():
    return 1

def top():
    helper()
    helper()
""",
        encoding="utf-8",
    )
    store = KuzuGraphStore(tmp_path / "graph")
    store.open()
    try:
        ing = GraphIngester(store)
        ing.ingest(p, "m.py", "python")

        callers = store.callers_of("helper")
        assert callers, "expected at least one caller"
        assert any(c.symbol == "top" for c in callers)

        callees = store.callees_of("top", path="m.py")
        assert any(c.symbol == "helper" for c in callees)

        sym = store.find_symbol("top", path="m.py")
        assert len(sym) == 1
        assert sym[0].symbol == "top"
    finally:
        store.close()


def test_extractor_finds_typescript_calls_and_imports(tmp_path: Path) -> None:
    p = tmp_path / "m.ts"
    p.write_text(
        """\
import { helper } from './lib';

export function top(): number {
    const a = helper();
    return a + helper();
}
""",
        encoding="utf-8",
    )
    ex = GraphExtractor()
    symbols, edges = ex.extract(p, "m.ts", "typescript")
    syms = {s.symbol for s in symbols}
    assert "top" in syms
    calls = [e for e in edges if e.kind == "calls" and e.src_symbol == "top"]
    assert any(e.dst_symbol == "helper" for e in calls), f"calls: {calls}"
    imports = [e for e in edges if e.kind == "imports"]
    assert any(e.dst_symbol == "./lib" for e in imports), f"imports: {imports}"


def test_extractor_finds_javascript_calls(tmp_path: Path) -> None:
    p = tmp_path / "m.js"
    p.write_text(
        """\
function helper() { return 1; }
function top() {
    const x = helper();
    return x + helper();
}
""",
        encoding="utf-8",
    )
    ex = GraphExtractor()
    _syms, edges = ex.extract(p, "m.js", "javascript")
    calls = [e for e in edges if e.kind == "calls" and e.src_symbol == "top"]
    assert any(e.dst_symbol == "helper" for e in calls), f"calls: {calls}"


def test_kuzu_delete_by_path_removes_symbols_and_edges(tmp_path: Path) -> None:
    p = tmp_path / "m.py"
    p.write_text(
        """\
def f():
    g()

def g():
    pass
""",
        encoding="utf-8",
    )
    store = KuzuGraphStore(tmp_path / "graph")
    store.open()
    try:
        ing = GraphIngester(store)
        ing.ingest(p, "m.py", "python")
        assert store.callers_of("g"), "sanity"

        store.delete_by_path("m.py")
        # After purge, both the symbol and its inbound edges are gone.
        assert not store.find_symbol("f", path="m.py")
        assert not store.callers_of("g", path="m.py")
    finally:
        store.close()

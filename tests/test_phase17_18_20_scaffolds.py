"""Tests for the Phase 17/18 scaffolds.

These two phases need user-supervised model downloads / system installs
before they can ship for real. Tests here verify the SCAFFOLDING is
correctly wired so the eventual flip-on is config-only:

  * Phase 17 (BGE Code v1 / CodeSage adapter): preset registry is
    discoverable, build_code_embedder routes to the right LMStudioEmbedder.
  * Phase 18 (SCIP semantic indexing): adapter contract is in place; the
    loader fails LOUDLY rather than silently returning empty when the
    parser isn't installed.

Phase 20 (ColBERT late-interaction) was deleted as orphaned scaffolding
in the Phase 37 audit — its module had zero callers and the
`code-rag colbert-index` CLI command its docstring referenced never
existed. Resurrect from git history if late-interaction work resumes.
"""
from __future__ import annotations

import pytest

from code_rag.embedders.code_specialized import (
    CODE_EMBEDDER_PRESETS,
    build_code_embedder,
    list_presets,
)
from code_rag.embedders.lm_studio import LMStudioEmbedder
from code_rag.graph.scip import (
    ScipIndex,
    SCIPNotInstalledError,
    ScipSymbol,
    iter_symbols_and_edges,
    load_scip_index,
)

# ---- P17 -------------------------------------------------------------------


def test_p17_presets_include_baseline_and_two_alternates() -> None:
    presets = {p.name for p in list_presets()}
    assert {"qwen3-embedding-4b", "bge-code-v1", "codesage-large-v2"} <= presets
    # Each preset has a HF repo for `lms get`.
    for p in CODE_EMBEDDER_PRESETS.values():
        assert p.huggingface_repo
        assert p.lms_id


def test_p17_build_code_embedder_returns_lm_studio_embedder() -> None:
    emb = build_code_embedder("bge-code-v1", base_url="http://localhost:1234/v1")
    assert isinstance(emb, LMStudioEmbedder)
    # The model id must match the preset's `lms_id`, NOT the friendly name —
    # otherwise LM Studio's /v1/models lookup would miss.
    assert emb.model == "bge-code-v1"


def test_p17_unknown_preset_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="unknown code embedder preset"):
        build_code_embedder("nonexistent-model")


# ---- P18 -------------------------------------------------------------------


def test_p18_load_scip_raises_loudly_without_parser(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """If `scip_pb2` isn't installed, load_scip_index must fail with a clear
    message — silent skipping would mislead the user into thinking SCIP was
    indexing when it wasn't."""
    fake = tmp_path / "index.scip"
    fake.write_bytes(b"")
    with pytest.raises(SCIPNotInstalledError, match="not installed"):
        load_scip_index(fake, repo_label="demo")


def test_p18_iter_symbols_and_edges_handles_empty_index() -> None:
    idx = ScipIndex(repo_label="demo")
    syms, edges = iter_symbols_and_edges(idx)
    assert syms == []
    assert edges == []


def test_p18_iter_symbols_and_edges_maps_minimal_record() -> None:
    """The sym→Edge mapping function must work on a hand-constructed
    `ScipIndex` so the contract is exercised even without the protobuf
    parser. Once the parser ships, only `load_scip_index` changes."""
    idx = ScipIndex(
        repo_label="demo",
        symbols=[
            ScipSymbol(scip_id="A", display_name="Foo", relative_path="a.cs",
                       kind="Class", start_line=1, end_line=10),
            ScipSymbol(scip_id="B", display_name="bar", relative_path="a.cs",
                       kind="Method", start_line=3, end_line=5),
        ],
        references=[("a.cs", "B", "A")],
    )
    syms, edges = iter_symbols_and_edges(idx)
    assert len(syms) == 2
    assert {s.symbol for s in syms} == {"Foo", "bar"}
    assert len(edges) == 1
    assert edges[0].src_symbol == "bar"
    assert edges[0].dst_symbol == "Foo"
    assert edges[0].kind == "calls"


# Phase 20 (ColBERT) tests deleted — the module they exercised was
# orphaned scaffolding with no production wiring; see file docstring above.

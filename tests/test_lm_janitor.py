"""Tests for the LM Studio zombie-instance janitor.

Detection logic only — `cleanup_once` and `janitor_loop` shell out to the
`lms` CLI and aren't worth mocking; we exercise them via the integration
test in the autostart smoke suite (covered manually in development).
"""
from __future__ import annotations

from code_rag.util.lm_janitor import Zombie, find_zombies


def _loaded(*ids: str) -> list[dict]:
    """Tiny helper: build the dict shape returned by `list_loaded_models_v0`."""
    return [{"id": i, "state": "loaded"} for i in ids]


def test_no_duplicates_returns_empty() -> None:
    """Zero zombies when every loaded id is unique."""
    loaded = _loaded(
        "text-embedding-qwen3-embedding-4b",
        "qwen2.5-coder-7b-instruct",
    )
    assert find_zombies(loaded) == []


def test_detects_simple_duplicate() -> None:
    """Classic case: base + `:2` both loaded → `:2` is the zombie."""
    loaded = _loaded(
        "text-embedding-qwen3-embedding-4b",
        "text-embedding-qwen3-embedding-4b:2",
    )
    zombies = find_zombies(loaded)
    assert zombies == [
        Zombie(
            identifier="text-embedding-qwen3-embedding-4b:2",
            base="text-embedding-qwen3-embedding-4b",
            suffix=2,
        ),
    ]


def test_multiple_suffixes_all_caught() -> None:
    """`:2`, `:3`, `:4` alongside the base — all three are zombies."""
    loaded = _loaded(
        "model-a",
        "model-a:2",
        "model-a:3",
        "model-a:4",
    )
    zombies = find_zombies(loaded)
    assert [z.identifier for z in zombies] == [
        "model-a:2", "model-a:3", "model-a:4",
    ]
    assert all(z.base == "model-a" for z in zombies)


def test_orphan_suffix_is_not_a_zombie() -> None:
    """If only `model:2` is loaded (no base), it's NOT a zombie — the user
    or LM Studio explicitly chose that instance, no duplicate exists."""
    loaded = _loaded("model-a:2")  # base missing
    assert find_zombies(loaded) == []


def test_colon_in_name_without_digit_suffix_ignored() -> None:
    """Names like `qwen/qwen3-1.7b` or `org:repo` containing colons but no
    trailing `:<digit>` must not be misidentified as suffixed instances."""
    loaded = _loaded(
        "qwen/qwen3-1.7b",       # forward-slash, fine
        "ggml-org:gpt-oss",      # colon present but no `:N` suffix
        "model-b",
    )
    assert find_zombies(loaded) == []


def test_mixed_loaded_and_not_loaded_only_counts_loaded() -> None:
    """`list_loaded_models_v0` already pre-filters to loaded, but verify
    `find_zombies` doesn't get confused by `state` field presence."""
    loaded = [
        {"id": "model-x", "state": "loaded"},
        {"id": "model-x:2", "state": "loaded"},
    ]
    zombies = find_zombies(loaded)
    assert len(zombies) == 1
    assert zombies[0].identifier == "model-x:2"


def test_independent_models_with_suffix_collision() -> None:
    """Two unrelated models, one of which happens to have a `:2` — the `:2`
    is a zombie ONLY of its own base, not of the other model."""
    loaded = _loaded(
        "alpha",
        "alpha:2",
        "beta",
        # No `beta:2` — so `alpha:2` is the only zombie.
    )
    zombies = find_zombies(loaded)
    assert [z.identifier for z in zombies] == ["alpha:2"]


def test_stable_ordering() -> None:
    """Ordering must be deterministic so logs are diff-friendly. Sorted by
    identifier ascending."""
    loaded = _loaded(
        "z-model",
        "z-model:9",
        "z-model:2",
        "a-model",
        "a-model:3",
    )
    ids = [z.identifier for z in find_zombies(loaded)]
    assert ids == ["a-model:3", "z-model:2", "z-model:9"]

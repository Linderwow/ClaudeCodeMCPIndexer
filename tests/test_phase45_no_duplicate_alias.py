"""Phase 45: prevent the LM Studio `<model>:N` duplicate-alias cascade.

Two related bugs both produced unbounded duplicate aliases:

1. `lms_ctl.model_loaded_with_correct_ctx` returned False whenever a
   model was in `/v1/models` but absent from `lms ps`. That's the
   normal state for a model that's "configured" (LM Studio knows
   about it) but "idle" (not currently in a CUDA stream). Phase 44's
   fast path read False as "wrong ctx" → triggered the slow path's
   `lms load` for ALL models → LM Studio responded by creating
   <embedder>:2 because the embedder was already loaded. Each
   autostart_bootstrap fire spawned a new alias.

2. `dashboard.operations.load_model` unconditionally issued `lms load`
   in the dashboard's "Start All" flow. Same trap: LM Studio creates
   the alias instead of being idempotent.

Both bugs converged on the same observable symptom: duplicates with
:2 / :3 etc. piling up, each holding ~3 GB of VRAM.

Fix
---
- model_loaded_with_correct_ctx returns True when ctx is unknown
  (None) — the helper now signals "no DEFINITIVE evidence of wrong
  ctx" rather than "wrong ctx".
- dashboard.operations.load_model short-circuits when the model is
  already loaded with the correct Phase 33 ctx.

Both changes are conservative: they pass through to the real load
when in doubt, but they refuse to reload when the model is already
correctly loaded.
"""
from __future__ import annotations

from unittest.mock import patch

from code_rag.dashboard import operations as ops
from code_rag.lms_ctl import (
    _LMS_LOAD_SETTINGS,
    LmsLocation,
    model_loaded_with_correct_ctx,
)


_PS_EMPTY = ""

_PS_EMBEDDER_ONLY_4096 = """\
IDENTIFIER                           MODEL                                STATUS    SIZE       CONTEXT    PARALLEL    DEVICE    TTL
text-embedding-qwen3-embedding-4b    text-embedding-qwen3-embedding-4b    IDLE      2.50 GB    4096       4           Local
"""

_PS_EMBEDDER_ONLY_40960 = """\
IDENTIFIER                           MODEL                                STATUS    SIZE       CONTEXT    PARALLEL    DEVICE    TTL
text-embedding-qwen3-embedding-4b    text-embedding-qwen3-embedding-4b    IDLE      2.50 GB    40960      -           Local
"""


class _FakeProc:
    def __init__(self, stdout: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = ""
        self.returncode = returncode


# ---- Phase 45-A: model_loaded_with_correct_ctx is now lenient on None ----


def test_45a_phase33_model_missing_from_ps_returns_true_not_false() -> None:
    """The HyDE model `qwen2.5-coder-7b-instruct` has a Phase 33 pref
    of (1, 8192) but is often not in `lms ps` (configured but idle).
    Pre-Phase 45 the helper returned False here, triggering an
    unnecessary unload+reload cascade that ended up creating the
    embedder's :2 alias. Post-Phase 45: True (no evidence of wrong
    ctx → leave it alone)."""
    assert "qwen2.5-coder-7b-instruct" in _LMS_LOAD_SETTINGS
    with patch("code_rag.lms_ctl.subprocess.run",
               return_value=_FakeProc(_PS_EMBEDDER_ONLY_4096)):
        # `lms ps` shows only the embedder; the HyDE model has ctx=None.
        assert model_loaded_with_correct_ctx(
            "lms.exe", "qwen2.5-coder-7b-instruct",
        ) is True


def test_45a_definitive_wrong_ctx_still_returns_false() -> None:
    """The fix doesn't make the helper toothless — when `lms ps` shows
    a Phase 33-tracked model loaded with a CONCRETELY wrong ctx, the
    helper still returns False so the boot path will reload it."""
    with patch("code_rag.lms_ctl.subprocess.run",
               return_value=_FakeProc(_PS_EMBEDDER_ONLY_40960)):
        assert model_loaded_with_correct_ctx(
            "lms.exe", "text-embedding-qwen3-embedding-4b",
        ) is False


def test_45a_correct_ctx_still_returns_true() -> None:
    """Sanity: the happy path didn't regress."""
    with patch("code_rag.lms_ctl.subprocess.run",
               return_value=_FakeProc(_PS_EMBEDDER_ONLY_4096)):
        assert model_loaded_with_correct_ctx(
            "lms.exe", "text-embedding-qwen3-embedding-4b",
        ) is True


def test_45a_empty_ps_treated_as_no_evidence() -> None:
    """When `lms ps` is empty (no models loaded at all), we have no
    evidence of wrong ctx → return True. The slow path will then issue
    `lms load` from cold, which is the correct fresh-load path."""
    with patch("code_rag.lms_ctl.subprocess.run",
               return_value=_FakeProc(_PS_EMPTY)):
        assert model_loaded_with_correct_ctx(
            "lms.exe", "text-embedding-qwen3-embedding-4b",
        ) is True


# ---- Phase 45-B: dashboard.operations.load_model short-circuits -----------


def test_45b_load_model_skips_redundant_lms_load_when_ctx_matches() -> None:
    """Dashboard's load_model used to ALWAYS issue `lms load`, which on
    an already-loaded model creates :2 alias. Phase 45-B short-circuits
    when the model is already loaded with the correct Phase 33 ctx.

    Verify: subprocess.run is called ONCE (only `lms ps` for the
    ctx-check) and not a second time for `lms load`."""
    fake_ps_proc = _FakeProc(_PS_EMBEDDER_ONLY_4096)

    calls: list[list[str]] = []

    def fake_run(cmd, *a, **kw):
        calls.append(list(cmd))
        return fake_ps_proc

    with patch("code_rag.dashboard.operations.find_lms",
               return_value=LmsLocation(path="lms.exe", found_via="test")), \
         patch("code_rag.lms_ctl.subprocess.run", side_effect=fake_run), \
         patch("code_rag.dashboard.operations.subprocess.run",
               side_effect=fake_run):
        result = ops.load_model("text-embedding-qwen3-embedding-4b", timeout_s=5.0)

    assert result.ok
    assert "skipped redundant" in result.detail.lower()
    # Only the `lms ps` ctx-check fired; `lms load` did NOT.
    cmds_used = [" ".join(str(x) for x in c) for c in calls]
    assert any("ps" in c for c in cmds_used)
    assert not any(c.endswith("load text-embedding-qwen3-embedding-4b") for c in cmds_used)


def test_45b_load_model_still_loads_when_model_not_present_in_ps() -> None:
    """First-boot path: nothing is loaded yet, `lms ps` is empty.
    load_model must still issue `lms load` so the model actually gets
    loaded — the fast-path skip is keyed off CONFIRMED-correct state,
    not absence of evidence."""
    calls: list[list[str]] = []

    def fake_run(cmd, *a, **kw):
        calls.append(list(cmd))
        if "ps" in cmd:
            return _FakeProc(_PS_EMPTY)
        # `lms load` succeeds.
        return _FakeProc("loaded", returncode=0)

    with patch("code_rag.dashboard.operations.find_lms",
               return_value=LmsLocation(path="lms.exe", found_via="test")), \
         patch("code_rag.lms_ctl.subprocess.run", side_effect=fake_run), \
         patch("code_rag.dashboard.operations.subprocess.run",
               side_effect=fake_run):
        result = ops.load_model("text-embedding-qwen3-embedding-4b", timeout_s=5.0)

    assert result.ok
    cmds_used = [" ".join(str(x) for x in c) for c in calls]
    # Ctx-check ran AND the actual load ran.
    assert any("ps" in c for c in cmds_used)
    assert any("load text-embedding-qwen3-embedding-4b" in c for c in cmds_used)


def test_45b_load_model_for_unknown_model_falls_through_to_real_load() -> None:
    """Models without a Phase 33 pref entry skip the short-circuit
    entirely (we don't know what ctx is correct for them) and just
    issue the load."""
    calls: list[list[str]] = []

    def fake_run(cmd, *a, **kw):
        calls.append(list(cmd))
        return _FakeProc("loaded", returncode=0)

    with patch("code_rag.dashboard.operations.find_lms",
               return_value=LmsLocation(path="lms.exe", found_via="test")), \
         patch("code_rag.dashboard.operations.subprocess.run",
               side_effect=fake_run):
        result = ops.load_model("some-random-model-with-no-pref", timeout_s=5.0)

    assert result.ok
    cmds_used = [" ".join(str(x) for x in c) for c in calls]
    # No `lms ps` short-circuit (no pref to compare against), straight to load.
    assert any("load some-random-model-with-no-pref" in c for c in cmds_used)

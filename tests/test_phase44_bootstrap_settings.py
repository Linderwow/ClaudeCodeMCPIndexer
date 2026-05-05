"""Phase 44: bootstrap fast-path now verifies LM Studio ctx matches prefs.

Bug context: ensure_lm_studio_ready had a fast path that returned
"already-ready" any time the model was loaded — without checking the
load settings. LM Studio's auto-load-on-server-start uses stock
defaults (CONTEXT=40960 for our 4B embedder). At every boot, the
embedder loaded with bad settings, autostart_bootstrap said
"already-ready", and never corrected them. The hourly enforce-settings
task was the only safety net — and that was itself broken until
Phase 43.

Today's user-visible cascade today:
  - Reboot → LM Studio auto-loads with CONTEXT=40960
  - autostart fires → ensure_lm_studio_ready sees "model is loaded"
    → returns immediately
  - Embedder uses ~10 GB extra VRAM on KV-cache buffers
  - VRAM at 96.0% with the embedder visibly idle

Fix: check both `model_is_loaded()` AND
`model_loaded_with_correct_ctx()` before claiming the fast path.
If ctx is wrong, unload so the slow path can reload with our prefs.
"""
from __future__ import annotations

from unittest.mock import patch

from code_rag.lms_ctl import (
    _LMS_LOAD_SETTINGS,
    loaded_context_length,
    model_loaded_with_correct_ctx,
)


_PS_LOADED_WRONG_CTX = """\
IDENTIFIER                           MODEL                                STATUS    SIZE       CONTEXT    PARALLEL    DEVICE    TTL
text-embedding-qwen3-embedding-4b    text-embedding-qwen3-embedding-4b    IDLE      2.50 GB    40960      -           Local
"""

_PS_LOADED_RIGHT_CTX = """\
IDENTIFIER                           MODEL                                STATUS    SIZE       CONTEXT    PARALLEL    DEVICE    TTL
text-embedding-qwen3-embedding-4b    text-embedding-qwen3-embedding-4b    IDLE      2.50 GB    4096       4           Local
"""

_PS_EMPTY = ""

_PS_HEADER_ONLY = (
    "IDENTIFIER  MODEL  STATUS  SIZE  CONTEXT  PARALLEL  DEVICE  TTL\n"
)


class _FakeProc:
    """Stand-in for subprocess.CompletedProcess from `lms ps`."""

    def __init__(self, stdout: str, returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = ""


def test_phase33_pref_for_embedder_is_4096() -> None:
    """The pref this fix is keyed off — pin it so a future tuning pass
    has a test reminding them to also adjust this guard."""
    par, ctx = _LMS_LOAD_SETTINGS["text-embedding-qwen3-embedding-4b"]
    assert par == 4
    assert ctx == 4096


def test_loaded_context_length_returns_int_when_present() -> None:
    with patch("code_rag.lms_ctl.subprocess.run",
               return_value=_FakeProc(_PS_LOADED_WRONG_CTX)):
        ctx = loaded_context_length("lms.exe", "text-embedding-qwen3-embedding-4b")
    assert ctx == 40960


def test_loaded_context_length_returns_none_when_model_not_loaded() -> None:
    """Different model id in the ps output → not loaded → None."""
    with patch("code_rag.lms_ctl.subprocess.run",
               return_value=_FakeProc(_PS_LOADED_WRONG_CTX)):
        ctx = loaded_context_length("lms.exe", "qwen2.5-coder-7b-instruct")
    assert ctx is None


def test_loaded_context_length_returns_none_when_ps_empty() -> None:
    with patch("code_rag.lms_ctl.subprocess.run",
               return_value=_FakeProc(_PS_EMPTY)):
        assert loaded_context_length("lms.exe", "any") is None


def test_loaded_context_length_returns_none_when_only_header() -> None:
    with patch("code_rag.lms_ctl.subprocess.run",
               return_value=_FakeProc(_PS_HEADER_ONLY)):
        assert loaded_context_length("lms.exe", "any") is None


def test_loaded_context_length_returns_none_on_subprocess_failure() -> None:
    with patch("code_rag.lms_ctl.subprocess.run",
               return_value=_FakeProc("", returncode=1)):
        assert loaded_context_length("lms.exe", "any") is None


def test_correct_ctx_true_when_match() -> None:
    """Embedder loaded with ctx=4096 → match → fast path is allowed."""
    with patch("code_rag.lms_ctl.subprocess.run",
               return_value=_FakeProc(_PS_LOADED_RIGHT_CTX)):
        assert model_loaded_with_correct_ctx(
            "lms.exe", "text-embedding-qwen3-embedding-4b",
        ) is True


def test_correct_ctx_false_when_mismatch() -> None:
    """Embedder loaded with ctx=40960 → mismatch → bootstrap should
    unload + reload."""
    with patch("code_rag.lms_ctl.subprocess.run",
               return_value=_FakeProc(_PS_LOADED_WRONG_CTX)):
        assert model_loaded_with_correct_ctx(
            "lms.exe", "text-embedding-qwen3-embedding-4b",
        ) is False


def test_correct_ctx_true_when_model_has_no_pref() -> None:
    """We don't have an opinion about random models the user loaded
    manually — accept whatever ctx they have."""
    sample = (
        "IDENTIFIER                           MODEL                                STATUS    SIZE       CONTEXT    PARALLEL    DEVICE    TTL\n"
        "some-random-model                    some-random-model                    IDLE      1.00 GB    8192       1           Local        \n"
    )
    with patch("code_rag.lms_ctl.subprocess.run",
               return_value=_FakeProc(sample)):
        assert model_loaded_with_correct_ctx(
            "lms.exe", "some-random-model",
        ) is True


def test_correct_ctx_false_when_pref_exists_but_model_not_loaded() -> None:
    """Phase 33-tracked model not in ps output. We have an opinion about
    its ctx; absent value can't match it."""
    with patch("code_rag.lms_ctl.subprocess.run",
               return_value=_FakeProc(_PS_EMPTY)):
        # 4B embedder pref is 4096; nothing loaded → no match.
        assert model_loaded_with_correct_ctx(
            "lms.exe", "text-embedding-qwen3-embedding-4b",
        ) is False

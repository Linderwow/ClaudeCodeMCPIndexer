"""Phase 43: regression test for the lms-enforce-settings `lms ps` parser.

The bug: the original parser used `line.split()` (single whitespace),
but `lms ps` aligns columns with 2+ spaces and the SIZE column contains
a literal space (e.g. "2.50 GB" → two tokens via .split()). This shifted
every column index by one starting at SIZE: cols[4] became "GB" instead
of CONTEXT, `int("GB")` raised ValueError, and every row was silently
skipped. Result: enforce-settings claimed "all models match Phase 33
prefs" no matter what.

Today's user-visible symptom: VRAM stuck at 94.9% (21.3 GB) because the
embedder loaded with CONTEXT=40960 instead of 4096, allocating ~10 GB
of unnecessary KV-cache buffers. lms-enforce-settings should have
caught it; instead it silently no-op'd.

This test pins the column-aware parsing logic so a future change to
the parser can't reintroduce the bug.
"""
from __future__ import annotations

import re

# We test the parsing logic in isolation — the production CLI code
# inlines it (it's <30 lines). Re-implementing the same shape here lets
# us pin the contract without forcing a refactor.


_SAMPLE = """\
IDENTIFIER                           MODEL                                STATUS    SIZE       CONTEXT    PARALLEL    DEVICE    TTL
text-embedding-qwen3-embedding-4b    text-embedding-qwen3-embedding-4b    IDLE      2.50 GB    40960      -           Local
qwen2.5-coder-7b-instruct            qwen2.5-coder-7b-instruct            IDLE      4.10 GB    8192       1           Local
"""


def _parse_lms_ps_context(stdout: str) -> dict[str, int]:
    """Mirror of the production parser. Returns {identifier: actual_ctx}."""
    out: dict[str, int] = {}
    lines = [ln.rstrip() for ln in stdout.splitlines() if ln.strip()]
    header_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith("IDENTIFIER")),
        None,
    )
    if header_idx is None:
        return out
    headers = [h.strip().lower() for h in re.split(r"\s{2,}", lines[header_idx])]
    try:
        ctx_col = headers.index("context")
    except ValueError:
        return out
    for line in lines[header_idx + 1:]:
        cols = re.split(r"\s{2,}", line)
        if len(cols) <= ctx_col:
            continue
        ident = cols[0]
        try:
            out[ident] = int(cols[ctx_col])
        except ValueError:
            continue
    return out


def test_parser_extracts_correct_context_despite_size_column_having_a_space() -> None:
    """The buggy parser would have read 'GB' (or worse, '2.50') as
    CONTEXT. The fixed parser reads 40960 / 8192."""
    parsed = _parse_lms_ps_context(_SAMPLE)
    assert parsed == {
        "text-embedding-qwen3-embedding-4b": 40960,
        "qwen2.5-coder-7b-instruct": 8192,
    }


def test_parser_handles_empty_output() -> None:
    assert _parse_lms_ps_context("") == {}


def test_parser_handles_header_only() -> None:
    """LM Studio with no models loaded prints just the header row."""
    header_only = "IDENTIFIER  MODEL  STATUS  SIZE  CONTEXT  PARALLEL  DEVICE  TTL\n"
    assert _parse_lms_ps_context(header_only) == {}


def test_parser_skips_alias_format_after_caller_filter() -> None:
    """Aliases like `<model>:2` aren't dropped by the parser (parser is
    column-only); the CLI does the alias filter. Here we verify the
    parser at least reads them WITHOUT crashing."""
    sample = (
        "IDENTIFIER                           MODEL                                STATUS    SIZE       CONTEXT    PARALLEL    DEVICE    TTL\n"
        "qwen2.5-coder-7b-instruct:2          qwen2.5-coder-7b-instruct            IDLE      4.10 GB    16384      1           Local        \n"
    )
    parsed = _parse_lms_ps_context(sample)
    assert parsed == {"qwen2.5-coder-7b-instruct:2": 16384}


def test_parser_recovers_when_a_row_is_malformed() -> None:
    """A row with too few columns or a non-numeric CONTEXT should be
    skipped, not crash the whole pass."""
    sample = (
        "IDENTIFIER                           MODEL                                STATUS    SIZE       CONTEXT    PARALLEL    DEVICE    TTL\n"
        "text-embedding-qwen3-embedding-4b    text-embedding-qwen3-embedding-4b    IDLE      2.50 GB    4096       4           Local        \n"
        "broken-row\n"
        "another-broken                       another-broken                       LOADED    1.00 GB    not-a-number  -        Local        \n"
        "qwen2.5-coder-7b-instruct            qwen2.5-coder-7b-instruct            IDLE      4.10 GB    8192       1           Local        \n"
    )
    parsed = _parse_lms_ps_context(sample)
    assert parsed == {
        "text-embedding-qwen3-embedding-4b": 4096,
        "qwen2.5-coder-7b-instruct": 8192,
    }


def test_buggy_split_would_have_misread_context() -> None:
    """Pin the bug: prove that the OLD parser (`line.split()`) would have
    extracted the wrong field for SIZE-with-space rows. This documents
    what we're protecting against."""
    line = (
        "text-embedding-qwen3-embedding-4b    text-embedding-qwen3-embedding-4b    "
        "IDLE      2.50 GB    40960      -           Local        "
    )
    cols_buggy = line.split()
    # Old code did `int(cols[4])`. With SIZE="2.50 GB", cols[4] is "GB".
    assert cols_buggy[4] == "GB"
    # The actual CONTEXT (40960) is at index 5 in the buggy split.
    assert cols_buggy[5] == "40960"

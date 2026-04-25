"""Tests for the transcript miner — heuristics, normalization, dedupe."""
from __future__ import annotations

import json
from pathlib import Path

from code_rag.eval.mine_transcripts import (
    _is_codey_question,
    _normalize_path,
    mine_session,
)

# ---- heuristics -------------------------------------------------------------


def test_codey_question_accepts_real_questions() -> None:
    assert _is_codey_question("Where is OnBarUpdate defined?")
    assert _is_codey_question("how does the strategy size positions")
    assert _is_codey_question("find ReadTradeGate")
    assert _is_codey_question("what calls UpdateDelta")
    assert _is_codey_question("OnBarUpdate")  # camelCase identifier alone


def test_codey_question_rejects_imperatives() -> None:
    assert not _is_codey_question("commit this")
    assert not _is_codey_question("push the changes")
    assert not _is_codey_question("run the tests")
    assert not _is_codey_question("fix the bug")
    assert not _is_codey_question("Add a logging line to MNQAlpha")
    assert not _is_codey_question("ok")
    assert not _is_codey_question("yes")
    assert not _is_codey_question("continue")


def test_codey_question_rejects_too_short_or_long() -> None:
    assert not _is_codey_question("hi")  # too short
    assert not _is_codey_question("x" * 700)  # too long


# ---- path normalization ----------------------------------------------------


def test_normalize_path_picks_longest_matching_root(tmp_path: Path) -> None:
    a = tmp_path / "outer"
    a.mkdir()
    b = a / "inner"
    b.mkdir()
    target = b / "f.cs"
    target.write_text("// content", encoding="utf-8")
    # Both roots could match; longest-prefix wins so the file is attributed
    # to the more-specific one.
    rel = _normalize_path(str(target), [a, b])
    assert rel == "f.cs"


def test_normalize_path_returns_none_if_outside_all_roots(tmp_path: Path) -> None:
    other = tmp_path / "alone"
    other.mkdir()
    f = other / "x.cs"
    f.write_text("// hi", encoding="utf-8")
    assert _normalize_path(str(f), [tmp_path / "elsewhere"]) is None


# ---- end-to-end on a synthetic transcript ----------------------------------


def _write_transcript(path: Path, lines: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")


def test_mine_session_extracts_question_to_path_pair(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "module.py"
    target.write_text("def foo(): ...\n", encoding="utf-8")

    transcript = tmp_path / "session.jsonl"
    _write_transcript(transcript, [
        # User asks a real question
        {"type": "user", "message": {"role": "user", "content": "Where is foo defined?"}},
        # Claude reads a file in response (simulated assistant tool_use)
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Read", "input": {"file_path": str(target)}}
        ]}},
    ])
    pairs = mine_session(transcript, [repo])
    assert len(pairs) == 1
    assert pairs[0].query == "Where is foo defined?"
    assert pairs[0].expected_path == "module.py"


def test_mine_session_skips_when_next_user_msg_intervenes(tmp_path: Path) -> None:
    """If the user follows up with another question before Claude reads
    anything, we shouldn't attribute the FIRST question to whatever came
    later — that's noise."""
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "f.py"
    target.write_text("# x\n", encoding="utf-8")

    transcript = tmp_path / "session.jsonl"
    _write_transcript(transcript, [
        {"type": "user", "message": {"role": "user", "content": "Where is foo?"}},
        {"type": "user", "message": {"role": "user", "content": "actually skip that, where is bar?"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Read", "input": {"file_path": str(target)}}
        ]}},
    ])
    pairs = mine_session(transcript, [repo])
    # First question stops looking at the second user msg → no pair for it.
    # Second question gets the file → 1 pair total.
    assert len(pairs) == 1
    assert pairs[0].query.startswith("actually")


def test_mine_session_ignores_imperative_messages(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "f.py"
    target.write_text("# x\n", encoding="utf-8")

    transcript = tmp_path / "session.jsonl"
    _write_transcript(transcript, [
        {"type": "user", "message": {"role": "user", "content": "commit this for me"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Read", "input": {"file_path": str(target)}}
        ]}},
    ])
    pairs = mine_session(transcript, [repo])
    assert pairs == []


def test_mine_session_skips_files_outside_indexed_roots(tmp_path: Path) -> None:
    repo = tmp_path / "indexed"
    repo.mkdir()
    elsewhere = tmp_path / "untracked"
    elsewhere.mkdir()
    out = elsewhere / "secret.py"
    out.write_text("# hi", encoding="utf-8")

    transcript = tmp_path / "session.jsonl"
    _write_transcript(transcript, [
        {"type": "user", "message": {"role": "user", "content": "where is the helper?"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Read", "input": {"file_path": str(out)}}
        ]}},
    ])
    pairs = mine_session(transcript, [repo])
    assert pairs == []  # Read was outside any indexed root → not ground truth


def test_mine_session_skips_system_reminders(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "f.py"
    target.write_text("# x", encoding="utf-8")

    transcript = tmp_path / "session.jsonl"
    _write_transcript(transcript, [
        {"type": "user", "message": {"role": "user",
                                      "content": "<system-reminder>\nreminder text\n</system-reminder>"}},
        {"type": "assistant", "message": {"role": "assistant", "content": [
            {"type": "tool_use", "name": "Read", "input": {"file_path": str(target)}}
        ]}},
    ])
    pairs = mine_session(transcript, [repo])
    assert pairs == []

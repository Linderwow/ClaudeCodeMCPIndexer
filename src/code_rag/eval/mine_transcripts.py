"""Mine (query, expected_paths) eval pairs from Claude Code transcripts.

Walks every `*.jsonl` under `~/.claude/projects/<project>/` looking for the
pattern:

    user message asking a code-location question
        ↓
    Claude's next tool_use call (Read / Edit / MultiEdit / NotebookEdit)
        ↓
    a concrete file path

The user's message becomes the query; the file path Claude landed on
becomes ground truth. Free, real, in-distribution.

This module is callable both from the CLI (`code-rag eval-mine`) and from
the standalone script (`scripts/mine_eval_from_transcripts.py`). Logic is
heuristic-driven so the eval set never includes commands or boilerplate.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# Heuristic word lists used to bias toward genuine code-location questions
# and away from "do this" commands. Cheap & deterministic; we'd rather miss
# a few good queries than poison the eval set with imperatives.
_QUESTION_PATTERNS = re.compile(
    r"\b(where|how|what|which|why|find|look[ -]?for|show( me)?|search|"
    r"locate|tell me|do (we|you|i)|is there|are there|does|shouldn'?t|"
    r"explain|understand|why .*(?:not|fail|broken)|implementation of|"
    r"definition of|callers? of|callees? of|references? to)\b",
    re.IGNORECASE,
)

# Imperative / command openers that are unlikely to be retrieval questions.
_COMMAND_PATTERNS = re.compile(
    r"^\s*("
    r"commit|push|merge|rebase|stash|"
    r"run( the)?|execute|build|compile|deploy|publish|"
    r"fix|patch|update|upgrade|bump|"
    r"add|create|generate|write|implement|make|"
    r"delete|remove|drop|kill|stop|"
    r"test|verify|check that|ensure|"
    r"refactor|rename|move|"
    r"start|restart|relaunch|"
    r"continue|proceed|go|yes|ok|"
    r"please .*\b(do|run|fix|add|create|make|update|push|commit)\b"
    r")\b",
    re.IGNORECASE,
)

# Tool calls that strongly imply Claude is looking up an answer to the user's
# question. Edit/MultiEdit also count because Claude often opens the canonical
# answer file with Edit when fixing a bug the user described.
_LOCATION_TOOLS = {"Read", "Edit", "MultiEdit", "NotebookEdit"}

_INDEXABLE_SUFFIXES = (
    ".py", ".cs", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".md", ".html", ".css", ".scss",
)


@dataclass
class MinedPair:
    query: str
    expected_path: str
    source_session: str
    raw_path: str  # absolute path from the transcript, before normalization

    def to_case(self) -> dict[str, object]:
        return {
            "query": self.query,
            "expected": [{"path": self.expected_path}],
            "_source": {
                "session": self.source_session,
                "raw_path": self.raw_path,
            },
        }


# ---- transcript parsing ----------------------------------------------------


def _user_text(msg: dict[str, object]) -> str | None:
    """Extract the user's actual text from a transcript line, or None if it's
    not a real user message (system reminders, tool results, etc.)."""
    if msg.get("type") != "user":
        return None
    inner = msg.get("message")
    if not isinstance(inner, dict) or inner.get("role") != "user":
        return None
    content = inner.get("content")
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        # Some transcripts have content as a list of {type, text} blocks.
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                t = block.get("text")
                if isinstance(t, str):
                    parts.append(t)
        text = "\n".join(parts).strip()
    else:
        return None
    if not text:
        return None
    # Skip system-reminder style messages and tool results that masquerade as
    # user-role in the transcript.
    if text.startswith(("<system-reminder>", "<bash-input>", "<command-")):
        return None
    if "<task-notification>" in text:
        return None
    if text.startswith("[") and "Request interrupted" in text[:80]:
        return None
    return text


def _assistant_tool_paths(msg: dict[str, object]) -> list[tuple[str, str]]:
    """For an assistant message, return [(tool_name, file_path), ...] of any
    location-style tool_use blocks. Empty if Claude used non-location tools."""
    if msg.get("type") != "assistant":
        return []
    inner = msg.get("message")
    if not isinstance(inner, dict):
        return []
    content = inner.get("content")
    if not isinstance(content, list):
        return []
    out: list[tuple[str, str]] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        tool = block.get("name")
        if tool not in _LOCATION_TOOLS:
            continue
        inp = block.get("input")
        if not isinstance(inp, dict):
            continue
        fp = inp.get("file_path")
        if isinstance(fp, str) and fp.endswith(_INDEXABLE_SUFFIXES):
            out.append((str(tool), fp))
    return out


def _is_codey_question(text: str) -> bool:
    """Cheap filter: does this user message LOOK like a code-location question?"""
    # Length: too short or too long suggests it's not a focused question.
    if len(text) < 10 or len(text) > 600:
        return False
    # Imperative / command openers are out.
    if _COMMAND_PATTERNS.search(text):
        return False
    # Either an explicit question pattern OR ends with "?" OR contains an
    # identifier-looking word (CamelCase / snake_case).
    if _QUESTION_PATTERNS.search(text):
        return True
    if text.rstrip().endswith("?"):
        return True
    # Identifier-like signal (CamelCase or snake_case word).
    return bool(re.search(r"\b[A-Z][a-z]+[A-Z]\w*\b", text) or re.search(r"\b\w+_\w+\b", text))


def _normalize_path(abs_path: str, roots: list[Path]) -> str | None:
    """Rewrite `abs_path` to the form stored in the index: relative to whichever
    indexed root contains it, posix-style. Returns None if outside every root."""
    try:
        ap = Path(abs_path).resolve()
    except OSError:
        return None
    best: tuple[str, int] | None = None
    for root in roots:
        try:
            rr = root.resolve()
        except OSError:
            continue
        try:
            rel = ap.relative_to(rr)
        except ValueError:
            continue
        depth = len(rr.parts)
        if best is None or depth > best[1]:
            best = (rel.as_posix(), depth)
    return best[0] if best else None


# ---- mining ----------------------------------------------------------------


def mine_session(
    jsonl_path: Path, roots: list[Path], lookahead: int = 12,
) -> list[MinedPair]:
    """Walk one transcript and emit a list of (query, expected_path) pairs."""
    try:
        with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
            messages = [json.loads(line) for line in f if line.strip()]
    except (OSError, json.JSONDecodeError):
        return []

    out: list[MinedPair] = []
    n = len(messages)
    session_id = jsonl_path.stem
    for i, msg in enumerate(messages):
        text = _user_text(msg)
        if not text:
            continue
        if not _is_codey_question(text):
            continue
        # Look ahead for the next assistant tool_use that hits a real file.
        for j in range(i + 1, min(i + 1 + lookahead, n)):
            nxt = messages[j]
            if _user_text(nxt):  # next real user msg — stop looking
                break
            tool_paths = _assistant_tool_paths(nxt)
            if not tool_paths:
                continue
            # Take the FIRST file Claude actually opened.
            _tool, raw = tool_paths[0]
            rel = _normalize_path(raw, roots)
            if rel is None:
                # File outside any indexed root — informational only.
                continue
            out.append(MinedPair(
                query=text,
                expected_path=rel,
                source_session=session_id,
                raw_path=raw,
            ))
            break
    return out


def mine_all(
    transcripts_dir: Path, roots: list[Path], max_pairs: int | None = None,
) -> list[MinedPair]:
    pairs: list[MinedPair] = []
    seen_queries: set[str] = set()
    for jsonl in sorted(transcripts_dir.rglob("*.jsonl")):
        for p in mine_session(jsonl, roots):
            # Dedupe near-duplicates by exact query text.
            key = p.query.strip().lower()
            if key in seen_queries:
                continue
            seen_queries.add(key)
            pairs.append(p)
            if max_pairs is not None and len(pairs) >= max_pairs:
                return pairs
    return pairs

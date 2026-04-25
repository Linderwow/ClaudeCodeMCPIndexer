"""Phase 22: diff-aware retrieval via git-log indexing.

For every indexed root that's a git repo, walk `git log -p -m --no-color`
and emit one Chunk per `(commit_sha, file_path)` diff hunk. The chunks land
in the same Chroma + FTS + Kuzu stores as ordinary code chunks, with
`kind=ChunkKind.DOC` and `language="git_diff"` so search filters can target
or exclude them.

Why this matters
----------------
A code RAG that doesn't understand history can't answer:
  * "When was the kelly sizing logic added?"
  * "What changed in MNQAlpha when v9 shipped?"
  * "Show me the commit where regimes got the noise floor."
With git-diff chunks indexed, those become natural-language queries that
hit the same hybrid retrieval pipeline as everything else.

Storage cost
------------
~1 chunk per file-touched-per-commit. A typical RiderProjects clone with
~5000 commits x ~3 files-per-commit avg ≈ 15,000 extra chunks (~150 MB
Chroma). Run on demand with `code-rag git-log-index`; not part of the
catch-up reindex by default since it's a one-shot bulk job.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from code_rag.logging import get
from code_rag.models import Chunk, ChunkKind
from code_rag.util.hashing import chunk_id

log = get(__name__)


@dataclass
class CommitMeta:
    sha: str
    short_sha: str
    author: str
    committed_at: str  # ISO 8601
    subject: str       # first line of commit message


def _is_git_repo(path: Path) -> bool:
    return (path / ".git").exists() or _run_git(path, ["rev-parse", "--git-dir"]) is not None


def _run_git(cwd: Path, args: list[str], *, max_bytes: int | None = None) -> str | None:
    """Invoke git, return stdout (text). Returns None on failure (so the
    caller can decide whether to skip or surface)."""
    try:
        r = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        log.warning("git_log.git_not_found")
        return None
    if r.returncode != 0:
        log.debug("git_log.cmd_failed", args=args, stderr=r.stderr[:200])
        return None
    out = r.stdout
    if max_bytes is not None and len(out) > max_bytes:
        out = out[:max_bytes]
    return out


def _commit_record_chunks(
    repo_label: str,
    commit: CommitMeta,
    file_diffs: list[tuple[str, str]],
    max_chars_per_chunk: int,
) -> list[Chunk]:
    """One Chunk per (commit, file). Chunk text is the diff hunk(s) for that
    file in that commit, capped at `max_chars_per_chunk`. The chunk's symbol
    encodes the commit SHA + file so callers can locate the source commit."""
    out: list[Chunk] = []
    for path, diff_text in file_diffs:
        if not diff_text.strip():
            continue
        if len(diff_text) > max_chars_per_chunk:
            head = diff_text[: max_chars_per_chunk - 80]
            diff_text = head + f"\n... (truncated; see git show {commit.sha} -- {path})\n"
        # `path` here is the path AS IT APPEARS IN THE DIFF -- relative to the
        # repo root, posix-style courtesy of git's normalization.
        sym = f"commit:{commit.short_sha}:{path}"
        # Symbolic chunk text framed with the commit message so semantic
        # search on "when was X added" can match on intent + diff body.
        framed = (
            f"// commit {commit.sha}\n"
            f"// author: {commit.author}\n"
            f"// date:   {commit.committed_at}\n"
            f"// {commit.subject}\n"
            f"// file:   {path}\n"
            f"\n{diff_text}"
        )
        out.append(Chunk(
            id=chunk_id(repo_label, path, sym, framed, 1),
            repo=repo_label,
            path=path,
            language="git_diff",
            symbol=sym,
            kind=ChunkKind.DOC,
            start_line=1,
            end_line=1,
            text=framed,
        ))
    return out


def _parse_porcelain_log(text: str) -> list[tuple[CommitMeta, list[tuple[str, str]]]]:
    """Parse `git log -p --no-color --format=...` into commit-records.

    Format we ask for (one block per commit, separated by NUL):
        <sha>\\x1f<short>\\x1f<author>\\x1f<iso-date>\\x1f<subject>
        <newline>
        <patch body -- unified diffs for every file touched>
    """
    out: list[tuple[CommitMeta, list[tuple[str, str]]]] = []
    if not text:
        return out
    # Split on the NUL we requested between commits.
    blocks = text.split("\x00")
    for block in blocks:
        block = block.strip("\n")
        if not block:
            continue
        try:
            header, _, body = block.partition("\n")
            sha, short, author, date, subject = header.split("\x1f", 4)
        except ValueError:
            continue
        meta = CommitMeta(
            sha=sha, short_sha=short, author=author, committed_at=date, subject=subject,
        )
        files = _split_unified_diff(body)
        out.append((meta, files))
    return out


def _split_unified_diff(body: str) -> list[tuple[str, str]]:
    """Split a unified-diff body into [(path, hunks_for_that_file), ...].

    Looks for `diff --git a/<path> b/<path>` boundaries -- git's standard
    per-file separator. Falls back to whole-body single-file if no
    boundaries are found (edge case: very small commits)."""
    out: list[tuple[str, str]] = []
    if not body:
        return out
    parts: list[tuple[str, list[str]]] = []
    cur_path: str | None = None
    cur_lines: list[str] = []
    for line in body.split("\n"):
        if line.startswith("diff --git "):
            if cur_path is not None:
                parts.append((cur_path, cur_lines))
            # `diff --git a/x b/x` -- extract the b/-side path
            try:
                _, _, ab = line.partition("a/")
                cur_path = ab.split(" b/", 1)[1] if " b/" in ab else ab
            except (IndexError, ValueError):
                cur_path = "unknown"
            cur_lines = [line]
        else:
            cur_lines.append(line)
    if cur_path is not None:
        parts.append((cur_path, cur_lines))
    elif cur_lines:
        parts.append(("(untitled)", cur_lines))
    for path, lines in parts:
        out.append((path, "\n".join(lines)))
    return out


def index_git_log(
    repo_dir: Path,
    repo_label: str,
    max_commits: int = 2000,
    max_chars_per_chunk: int = 2400,
    max_diff_bytes: int = 50_000_000,
) -> list[Chunk]:
    """Walk the git log of `repo_dir` and emit Chunks for every diff hunk.

    Caller is responsible for upserting the returned chunks into vector +
    lexical + (optionally) graph stores. Embedding them is the same code
    path as ordinary doc chunks.
    """
    if not _is_git_repo(repo_dir):
        return []
    fmt = "%H\x1f%h\x1f%an <%ae>\x1f%aI\x1f%s"
    out_text = _run_git(
        repo_dir,
        [
            "log",
            "-p",
            "-m",                   # show diffs for merges too
            "--no-color",
            "--no-renames",
            f"-n{max_commits}",
            f"--format=%x00{fmt}",  # NUL-prefix each commit so the parser can split
        ],
        max_bytes=max_diff_bytes,
    )
    if out_text is None:
        return []
    records = _parse_porcelain_log(out_text)
    chunks: list[Chunk] = []
    for meta, files in records:
        chunks.extend(_commit_record_chunks(repo_label, meta, files, max_chars_per_chunk))
    log.info("git_log.indexed",
             repo=repo_label, commits=len(records), chunks=len(chunks))
    return chunks

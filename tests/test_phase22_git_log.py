"""Phase 22: git-log diff indexing produces queryable chunks for every
(commit x file) pair so retrieval can answer "when was X added" questions."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from code_rag.indexing.git_log import (
    _is_git_repo,
    _parse_porcelain_log,
    _split_unified_diff,
    index_git_log,
)


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(
        ["git", *args], cwd=str(repo),
        capture_output=True, text=True, check=True,
        encoding="utf-8", errors="replace",
    )
    return r.stdout


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "demo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Tester")
    (repo / "alpha.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "Initial: alpha helper")
    (repo / "alpha.py").write_text("def alpha():\n    return 42\n", encoding="utf-8")
    (repo / "beta.py").write_text("def beta():\n    return 'b'\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "Add beta + tweak alpha")
    return repo


# ---- pure-text parsing -----------------------------------------------------


def test_split_unified_diff_separates_by_file() -> None:
    body = """\
diff --git a/foo.py b/foo.py
index 1234..5678 100644
--- a/foo.py
+++ b/foo.py
@@ -1 +1 @@
-old
+new
diff --git a/bar.py b/bar.py
index 1234..5678 100644
--- a/bar.py
+++ b/bar.py
@@ -0,0 +1,1 @@
+brand new
"""
    parts = _split_unified_diff(body)
    paths = [p for p, _ in parts]
    assert paths == ["foo.py", "bar.py"]
    assert "+new" in dict(parts)["foo.py"]
    assert "+brand new" in dict(parts)["bar.py"]


def test_parse_porcelain_log_handles_multi_file_commit() -> None:
    text = (
        "\x00deadbeef\x1fdead\x1fAuthor <a@b>\x1f2026-04-25T00:00:00\x1ffirst\n"
        "diff --git a/a.py b/a.py\nindex 0..1 100644\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n"
        "diff --git a/b.py b/b.py\nindex 0..1 100644\n--- a/b.py\n+++ b/b.py\n@@ +1 @@\n+x\n"
    )
    parsed = _parse_porcelain_log(text)
    assert len(parsed) == 1
    meta, files = parsed[0]
    assert meta.sha == "deadbeef"
    assert meta.short_sha == "dead"
    assert meta.author == "Author <a@b>"
    assert meta.subject == "first"
    paths = [p for p, _ in files]
    assert paths == ["a.py", "b.py"]


# ---- end-to-end against a real `git init` ---------------------------------


def test_index_git_log_emits_chunks_for_every_commit_and_file(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    assert _is_git_repo(repo)
    chunks = index_git_log(repo, repo_label="demo", max_commits=10)
    # 2 commits x ~variable file count. First commit touches alpha.py;
    # second touches alpha.py + beta.py. Expect at least 3 chunks.
    assert len(chunks) >= 3, f"got {len(chunks)} chunks; expected >=3"
    paths = {c.path for c in chunks}
    assert "alpha.py" in paths
    assert "beta.py" in paths
    # Each chunk's text is the framed diff with commit + file metadata.
    sample = chunks[0]
    assert "// commit " in sample.text
    assert "// author:" in sample.text
    assert sample.language == "git_diff"
    # Symbol encodes "commit:<sha>:<path>" so retrieval results visibly tie
    # back to the commit.
    assert sample.symbol is not None and sample.symbol.startswith("commit:")
    # IDs are unique per (sha, file) -- same diff body in two commits would
    # otherwise collide.
    assert len({c.id for c in chunks}) == len(chunks)


def test_index_git_log_returns_empty_for_nongit_dir(tmp_path: Path) -> None:
    plain = tmp_path / "notarepo"
    plain.mkdir()
    assert index_git_log(plain, repo_label="demo") == []


def test_index_git_log_truncates_huge_diffs(tmp_path: Path) -> None:
    """Massive single-commit diffs must get truncated, not blow memory."""
    repo = tmp_path / "huge"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@e.com")
    _git(repo, "config", "user.name", "t")
    big = "\n".join(f"line_{i} = {i}" for i in range(2000))
    (repo / "big.py").write_text(big, encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "huge add")

    chunks = index_git_log(repo, repo_label="huge", max_chars_per_chunk=500)
    assert chunks
    # Every chunk must be <= max_chars + a small framing overhead.
    for c in chunks:
        assert len(c.text) < 1500, f"chunk len {len(c.text)} exceeds cap"
    # The truncation marker is present in at least one chunk.
    assert any("truncated" in c.text for c in chunks)


@pytest.mark.skipif(subprocess.run(["git", "--version"], capture_output=True, check=False).returncode != 0, reason="git not installed")
def test_index_git_log_handles_repo_with_renames(tmp_path: Path) -> None:
    """Renames in git history are flattened into add+delete pairs by our
    --no-renames flag, so each shows up as its own chunk. No crash."""
    repo = tmp_path / "renames"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@e.com")
    _git(repo, "config", "user.name", "t")
    (repo / "old_name.py").write_text("x = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "add old_name.py")
    (repo / "old_name.py").rename(repo / "new_name.py")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "rename to new_name.py")

    chunks = index_git_log(repo, repo_label="renames")
    paths = {c.path for c in chunks}
    assert "old_name.py" in paths
    assert "new_name.py" in paths

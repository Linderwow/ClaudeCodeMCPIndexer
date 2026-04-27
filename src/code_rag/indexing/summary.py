"""Phase 31: per-file synthetic "table of contents" chunks.

Pure-Python, no LLM. Given the per-symbol chunks emitted by the tree-sitter
chunker for one file, we produce ONE additional chunk that summarizes the
file's contents — its path, language, and the list of symbols defined in it.

Why this helps retrieval
------------------------
Pre-Phase-31, a query like "module that handles authentication" had to find
its target via per-symbol chunks. If `auth.py` defined 12 functions, none of
which had "authentication" in their bodies (they had `login`, `logout`,
`verify_token`, etc.), the file might be invisible to the dense embedder
even though its name and contents collectively scream "auth".

The synthesized chunk text reads:

    [file] services/auth.py
    [language] python
    [defines] login, logout, verify_token, refresh_session, hash_password
    [summary] python module with 12 symbols (5 functions, 4 methods, 3 classes)

That gives the embedder a single document that semantically summarizes the
file. Retrieval over this chunk catches the "find the file" intent that
per-symbol chunks miss.

Cost
----
- One extra chunk per indexed file (~few thousand on the user's corpus,
  vs 87K total — marginal blob-size impact).
- Indexer change is additive: one extra chunk emit per file. Embedding
  cost is one extra batch per file (~tens of ms).
- No new infrastructure. Stored in the same Chroma collection alongside
  the regular chunks; metadata.kind="file_summary" disambiguates.

Idempotency
-----------
The chunk's `id` is content-addressed (blake3 over path + symbol names),
so re-indexing a file with the same set of symbols is a no-op. Adding
or removing a symbol changes the id, so the stale summary is replaced
on the delete-by-path pass.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from blake3 import blake3

from code_rag.models import Chunk, ChunkKind


# How many symbol names to inline into the chunk text. More = better recall
# but bigger embedding cost; 30 is a reasonable cap for big files like
# `MNQAlpha.cs` while staying under typical embedder context limits.
_MAX_SYMBOLS_INLINE = 30


def _pluralize(word: str, n: int) -> str:
    """Tiny pluralizer for the kind histogram. Only handles the kinds we
    actually emit (function/method/class/struct/interface/enum/namespace/
    module/doc/other/file_summary). Avoids the "classs" / "structs" bugs
    a naive `+s` produces."""
    if n == 1:
        return word
    if word.endswith("s") or word.endswith("x") or word.endswith("ch"):
        return word + "es"      # class -> classes, box -> boxes
    if word.endswith("y") and not word.endswith("ay") and not word.endswith("ey"):
        return word[:-1] + "ies"  # not used today, future-proof
    return word + "s"


def synthesize_file_summary(
    repo: str,
    rel_path: str,
    language: str,
    chunks: Sequence[Chunk],
) -> Chunk | None:
    """Build one summary chunk for a file from its per-symbol chunks.

    Returns None if there are no chunks (empty file, parse failure, etc.) —
    in which case there's nothing to summarize.

    Args:
        repo: same `repo` field used by the per-symbol chunks (file's root name).
        rel_path: posix-style relative path used by the per-symbol chunks.
        language: tree-sitter language id (matches chunk.language).
        chunks: the per-symbol chunks for this file (NOT including any
                previous summary chunk).

    The returned chunk has:
      - kind = ChunkKind.FILE_SUMMARY
      - symbol = None (this isn't a symbol chunk)
      - start_line = 1, end_line = max end_line across input chunks
      - text = a small structured doc that lists path / lang / symbols
    """
    if not chunks:
        return None

    # Collect symbol names + kind histogram.
    syms: list[str] = []
    seen: set[str] = set()
    kind_counts: Counter[str] = Counter()
    end_line = 1
    for c in chunks:
        if c.kind == ChunkKind.FILE_SUMMARY:
            # Defensive: never re-summarize a summary if the caller messes up.
            continue
        kind_counts[c.kind.value] += 1
        if c.end_line > end_line:
            end_line = c.end_line
        if c.symbol and c.symbol not in seen:
            seen.add(c.symbol)
            syms.append(c.symbol)

    if not syms and not kind_counts:
        return None

    # Truncate inlined symbol list. Keep names that are most "informative":
    # original input order is fine — tree-sitter chunker tends to emit
    # functions/methods/classes in source order, which maps to file's logical
    # structure.
    inlined = syms[:_MAX_SYMBOLS_INLINE]
    extra = len(syms) - len(inlined)

    parts = [
        f"[file] {rel_path}",
        f"[language] {language}",
    ]
    if inlined:
        sym_list = ", ".join(inlined)
        if extra > 0:
            sym_list += f", ... (+{extra} more)"
        parts.append(f"[defines] {sym_list}")
    if kind_counts:
        kind_summary = ", ".join(
            f"{n} {_pluralize(k, n)}"
            for k, n in sorted(kind_counts.items(), key=lambda x: -x[1])
        )
        parts.append(
            f"[summary] {language} module with {sum(kind_counts.values())} "
            f"symbols ({kind_summary})"
        )
    text = "\n".join(parts)

    # Content-addressed id — invalidated only if the path's symbol set
    # actually changes.
    payload = f"{repo}|{rel_path}|file_summary|{','.join(syms)}".encode()
    chunk_id = blake3(payload).hexdigest()

    return Chunk(
        id=chunk_id,
        repo=repo,
        path=rel_path,
        language=language,
        symbol=None,
        kind=ChunkKind.FILE_SUMMARY,
        start_line=1,
        end_line=end_line,
        text=text,
    )

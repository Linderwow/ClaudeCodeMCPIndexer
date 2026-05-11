"""Phase 60-R: Qwen3-Embedding query-side instruct prefixing.

Qwen3-Embedding is an asymmetric retrieval model. Query and document
embeddings are produced differently *on purpose*:

* Queries  → `Instruct: {task}\\nQuery: {text}`
* Documents → raw text (no prefix)

The `{task}` string is a short English sentence telling the model what
relationship to optimize for. Qwen3 was trained with these prompts;
omitting them leaves performance on the table. Per Qwen's model card and
internal eval, a well-targeted instruct yields 1-5% retrieval lift, with
the larger gains coming from code / identifier / error-message queries
where un-prefixed embeddings drift to generic semantics.

Phase 1 (this file) ships a single DEFAULT instruct applied to every
query. Phase 2 will add typed instructs (CODE / CONFIG / DOCS / ERROR /
CONCEPT) plus a classifier that routes per query.

Two invariants enforced by the implementation:

1. The prefix is applied at QUERY time only. Document embeddings stay
   un-prefixed -- do NOT retroactively re-embed ingested chunks. If you
   ever change which prefix the query side uses, you do NOT need to
   re-ingest.
2. The instruction is written in English regardless of the query
   language (Qwen3-Embedding's training data was primarily English).
"""
from __future__ import annotations

QUERY_INSTRUCT_DEFAULT = (
    "Given a software engineering query, retrieve relevant source code, "
    "technical documentation, configuration files, or API references "
    "that answer the query"
)


def format_query(text: str, instruct: str | None = None) -> str:
    """Format a query for Qwen3-Embedding asymmetric retrieval.

    Args:
        text: the raw query string the user typed (or a rewriter variant).
        instruct: optional task-specific instruct; falls back to
            QUERY_INSTRUCT_DEFAULT when None. Phase 2 will use this
            argument to route per-query (CODE / CONFIG / DOCS / ERROR /
            CONCEPT).

    Returns:
        The string to send to the embeddings endpoint.
    """
    task = instruct or QUERY_INSTRUCT_DEFAULT
    return f"Instruct: {task}\nQuery: {text}"

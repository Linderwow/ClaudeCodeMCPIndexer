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

Phase 1 ships a single DEFAULT instruct applied to every query.
Phase 2 (this file) adds typed instructs (CODE / CONFIG / DOCS /
ERROR / CONCEPT) plus a `select_query_instruct` router that picks
the best one per query shape. The classifier booleans come from
`code_rag.retrieval.query_classify`; the router takes them as inputs
so this module stays dependency-free from regex matching.

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

QUERY_INSTRUCT_CODE = (
    "Given a code search query, retrieve relevant source code implementations, "
    "function definitions, class declarations, or API signatures "
    "that answer the query"
)
QUERY_INSTRUCT_CONFIG = (
    "Given a configuration or infrastructure query, retrieve relevant "
    "configuration files, YAML/JSON definitions, Helm charts, Terraform modules, "
    "Kubernetes manifests, or CI/CD pipeline definitions that answer the query"
)
QUERY_INSTRUCT_DOCS = (
    "Given a documentation query, retrieve relevant technical documentation, "
    "architecture guides, API references, or decision records "
    "that answer the query"
)
QUERY_INSTRUCT_ERROR = (
    "Given an error message or debugging query, retrieve relevant error "
    "explanations, fixes, similar issues, stack-trace analyses, or "
    "troubleshooting documentation that answer the query"
)
# CONCEPT intentionally closes with "the question" rather than "the
# query" — closer to Qwen3-Embedding's QA training distribution for
# how-to/explanatory prompts.
QUERY_INSTRUCT_CONCEPT = (
    "Given a conceptual programming question or how-to query, retrieve "
    "relevant explanations, tutorials, comparison guides, or architectural "
    "overviews that answer the question"
)

# Routing table for explicit content-type filters. When the caller sets
# request.content_type_filter=<kind>, we override the auto-classifier and
# pick the matching instruct directly. The map is intentionally narrow:
# only kinds the indexer actually emits as a chunk-level metadata facet.
CONTENT_TYPE_INSTRUCT: dict[str, str] = {
    "code":     QUERY_INSTRUCT_CODE,
    "yaml":     QUERY_INSTRUCT_CONFIG,
    "json":     QUERY_INSTRUCT_CONFIG,
    "markdown": QUERY_INSTRUCT_DOCS,
    "text":     QUERY_INSTRUCT_DOCS,
    "openapi":  QUERY_INSTRUCT_DOCS,
}


def format_query(text: str, instruct: str | None = None) -> str:
    """Format a query for Qwen3-Embedding asymmetric retrieval.

    Args:
        text: the raw query string the user typed (or a rewriter variant).
        instruct: optional task-specific instruct; falls back to
            QUERY_INSTRUCT_DEFAULT when None. Phase 2 uses this argument
            to route per-query (CODE / CONFIG / DOCS / ERROR / CONCEPT).

    Returns:
        The string to send to the embeddings endpoint.
    """
    task = instruct or QUERY_INSTRUCT_DEFAULT
    return f"Instruct: {task}\nQuery: {text}"


def select_query_instruct(
    *,
    content_type_filter: str | None,
    is_identifier_query: bool,
    is_error_query: bool,
    is_concept_query: bool = False,
    mode: str = "auto",
) -> str | None:
    """Pick the best query instruct for the given query shape, or None
    to fall through to QUERY_INSTRUCT_DEFAULT.

    PRECEDENCE (first match wins). The order is not arbitrary — it
    reflects eval observations from comparable code-retrieval systems:

      1. content_type_filter present → look up CONTENT_TYPE_INSTRUCT.
         User-set filter is the strongest possible signal.
      2. is_identifier_query → CODE. A named symbol like
         "NpgsqlException 23505 retry handler" is a stronger user signal
         than the pasted error pattern — route it to the retry-handler
         class, not generic troubleshooting docs.
      3. is_error_query → ERROR. Identifier already won above; if we're
         here the query is a pure error pattern.
      4. is_concept_query → CONCEPT. Both identifier and error precede
         concept so "How does FastEndpoints register dependencies" goes
         to CODE despite the "how does" scaffold.
      5. Otherwise → None (caller uses DEFAULT).

    Args:
        content_type_filter: optional filter from the request (e.g.
            "yaml", "markdown"). When present, takes precedence over
            all classifier signals.
        is_identifier_query: pre-computed boolean from query_classify.
            Pre-computing the bool keeps this module regex-free.
        is_error_query: same.
        is_concept_query: same. Defaults False so older callers don't
            have to thread it through.
        mode: escape hatch for A/B testing.
            - "auto" (default): all rules apply.
            - "content_type_only": only rule 1 applies. Use to A/B
              the classifier without touching ingest.
            - "off": always return None (prefixing disabled entirely;
              callers fall back to DEFAULT which is still applied at
              format_query — `mode=off` only disables ROUTING, not
              prefixing. To disable prefixing entirely, the caller
              should skip embed_query in favor of embed).

    Returns:
        The chosen instruct string, or None for "use DEFAULT."
    """
    if mode == "off":
        return None
    if content_type_filter:
        return CONTENT_TYPE_INSTRUCT.get(content_type_filter)
    if mode == "content_type_only":
        return None
    if is_identifier_query:
        return QUERY_INSTRUCT_CODE
    if is_error_query:
        return QUERY_INSTRUCT_ERROR
    if is_concept_query:
        return QUERY_INSTRUCT_CONCEPT
    return None

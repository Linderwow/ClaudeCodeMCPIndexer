"""Phase 60-R Phase 2: typed instructs + classifier + routing.

Three layers of tests:

1. CLASSIFIERS — `is_identifier_query` / `is_error_query` /
   `is_concept_query` regex predicates. Each gets 6-10 positive and
   4-6 negative cases, with explicit coverage of the edges the audit
   flagged (DbUp vs Datadog-in-Nextjs, NullReferenceException vs
   "timeout best practices", etc.).

2. ROUTING — `select_query_instruct` precedence: content-type wins
   over identifier wins over error wins over concept, with explicit
   mode behavior for "auto" / "content_type_only" / "off".

3. INTEGRATION — HybridSearcher actually passes the routed instruct
   through to embed_query, verified via spy.
"""
from __future__ import annotations

from collections.abc import Sequence

import pytest

from code_rag.embedders.prompts import (
    CONTENT_TYPE_INSTRUCT,
    QUERY_INSTRUCT_CODE,
    QUERY_INSTRUCT_CONCEPT,
    QUERY_INSTRUCT_CONFIG,
    QUERY_INSTRUCT_DOCS,
    QUERY_INSTRUCT_ERROR,
    select_query_instruct,
)
from code_rag.retrieval.query_classify import (
    is_concept_query,
    is_error_query,
    is_identifier_query,
)


# ---- identifier classifier --------------------------------------------------


@pytest.mark.parametrize("query", [
    "DbUp database migration",            # short query + PascalCase
    "FastEndpoints handler registration",  # short query + PascalCase
    "totalDelay computation",              # camelCase
    "getUser implementation",              # camelCase
    "ILogger setup",                       # C# interface
    "IConfiguration injection",            # C# interface
    "MOD-2561 dialer",                     # ticket ID
    "Some.Dotted.Namespace",               # dotted namespace
    "azurerm_postgresql_flexible_server",  # snake_case (Terraform)
    "Messaging:Transport queue",           # colon-separated PascalCase
])
def test_identifier_query_positive(query: str) -> None:
    """Each of these should classify as a code-identifier search."""
    assert is_identifier_query(query) is True, \
        f"expected identifier-query=True for {query!r}"


@pytest.mark.parametrize("query", [
    # Brand suffix is .net / .js / etc — not a real namespace.
    "ASP.NET Core dependency injection",
    # Single PascalCase + NL scaffolding ("how to") demotes.
    "How to configure Datadog RUM in a Next.js app",
    # Pure prose.
    "what is the difference between processes and threads",
    # Concept question.
    "explain why race conditions happen",
    # Long sentence with no identifier shape.
    "the company decided that we should consolidate our retrieval logic",
    # Plain noun phrase.
    "database migration best practices",
])
def test_identifier_query_negative(query: str) -> None:
    """These look like NL or have only brand tokens — NOT code searches."""
    assert is_identifier_query(query) is False, \
        f"expected identifier-query=False for {query!r}"


# ---- error classifier -------------------------------------------------------


@pytest.mark.parametrize("query", [
    "NullReferenceException",                       # bare *Exception
    "ArgumentNullException at startup",             # *Exception in phrase
    "SQLSTATE 23505 duplicate key",                 # Postgres SQLSTATE
    "CS8602 dereference of possibly null reference",  # C# compiler
    "TS2322 type mismatch",                         # TypeScript
    "ETIMEDOUT connecting to redis",                # node errno
    "ECONNREFUSED 127.0.0.1:8000",                  # connection refused
    "Error: cannot find module 'foo'",              # Error: prefix
    "Traceback: ZeroDivisionError",                 # Python traceback
    "ERR_CONNECTION_REFUSED in browser",            # ERR_ prefix
    "errno = 2 file not found",                     # raw errno
    "exit code 137 from container",                 # exit code
    "CORS policy blocked the origin header",        # CORS error
])
def test_error_query_positive(query: str) -> None:
    """Each should classify as an error/debug query."""
    assert is_error_query(query) is True, \
        f"expected error-query=True for {query!r}"


@pytest.mark.parametrize("query", [
    "timeout best practices",                # no error pattern
    "how to handle errors gracefully",       # the WORD error, no pattern
    "exception handling guide",              # generic noun
    "retry strategies for distributed systems",
    "what does CORS mean",                   # explanation, not a CORS error
    "best way to debug Python code",         # generic
])
def test_error_query_negative(query: str) -> None:
    """No error-pattern → not an error query."""
    assert is_error_query(query) is False, \
        f"expected error-query=False for {query!r}"


# ---- concept classifier -----------------------------------------------------


@pytest.mark.parametrize("query", [
    "how does X work",
    "what is dependency injection",
    "why use a circuit breaker",
    "explain the actor model",
    "describe the saga pattern",
    "compare REST and gRPC",
    "differences between processes and threads",
    "Redis vs Memcached",
    "when to use a microservice",
    "tradeoffs of event sourcing",
    "pros and cons of GraphQL",
    "tutorial for OpenTelemetry",
    "Kubernetes overview",
    "guide to load balancing",
])
def test_concept_query_positive(query: str) -> None:
    """Each should classify as a concept / how-to / comparison."""
    assert is_concept_query(query) is True, \
        f"expected concept-query=True for {query!r}"


@pytest.mark.parametrize("query", [
    "FastEndpoints.Connect()",         # method call, no NL scaffolding
    "ILogger.LogError signature",      # bare identifier phrase
    "azurerm_postgresql_flexible_server module",
    "NullReferenceException",          # error, not concept
    "MOD-2561",                        # ticket
])
def test_concept_query_negative(query: str) -> None:
    """Bare identifiers / errors / tickets — not concept queries."""
    assert is_concept_query(query) is False, \
        f"expected concept-query=False for {query!r}"


# ---- select_query_instruct routing precedence ------------------------------


def test_routing_content_type_wins_over_classifiers() -> None:
    """Rule 1 of precedence: an explicit content_type_filter overrides
    every classifier signal. User-set filter is the strongest signal."""
    # Even if we claim "identifier + error + concept" all True, an
    # explicit content_type_filter=yaml should route to CONFIG.
    out = select_query_instruct(
        content_type_filter="yaml",
        is_identifier_query=True,
        is_error_query=True,
        is_concept_query=True,
    )
    assert out == QUERY_INSTRUCT_CONFIG


def test_routing_identifier_beats_error() -> None:
    """Rule 2: identifier > error. A named symbol + pasted error pattern
    routes to CODE -- the symbol is the stronger signal. E.g.
    'NpgsqlException 23505 retry handler' should hit the retry-handler
    class, not generic troubleshooting."""
    out = select_query_instruct(
        content_type_filter=None,
        is_identifier_query=True,
        is_error_query=True,
        is_concept_query=False,
    )
    assert out == QUERY_INSTRUCT_CODE


def test_routing_identifier_beats_concept() -> None:
    """Rule 2 > Rule 4: identifier still wins when concept scaffolding
    is present. 'How does FastEndpoints register dependencies' → CODE."""
    out = select_query_instruct(
        content_type_filter=None,
        is_identifier_query=True,
        is_error_query=False,
        is_concept_query=True,
    )
    assert out == QUERY_INSTRUCT_CODE


def test_routing_error_beats_concept() -> None:
    """Rule 3 > Rule 4: error wins when no identifier is present.
    'how do I fix NullReferenceException' → ERROR (concept scaffolding
    isn't enough to override the error pattern)."""
    out = select_query_instruct(
        content_type_filter=None,
        is_identifier_query=False,
        is_error_query=True,
        is_concept_query=True,
    )
    assert out == QUERY_INSTRUCT_ERROR


def test_routing_concept_picked_when_alone() -> None:
    """Rule 4: with no identifier or error, concept is picked.
    'How does dependency injection work' → CONCEPT."""
    out = select_query_instruct(
        content_type_filter=None,
        is_identifier_query=False,
        is_error_query=False,
        is_concept_query=True,
    )
    assert out == QUERY_INSTRUCT_CONCEPT


def test_routing_returns_none_for_unclassified_query() -> None:
    """All flags False, no content_type_filter → fall through to None
    (caller uses DEFAULT)."""
    out = select_query_instruct(
        content_type_filter=None,
        is_identifier_query=False,
        is_error_query=False,
        is_concept_query=False,
    )
    assert out is None


# ---- mode behavior ----------------------------------------------------------


def test_routing_mode_off_always_returns_none() -> None:
    """mode='off' is the kill-switch. Every other signal is ignored
    and the caller uses DEFAULT for every query."""
    # Even with EVERYTHING set, mode=off returns None.
    out = select_query_instruct(
        content_type_filter="yaml",
        is_identifier_query=True,
        is_error_query=True,
        is_concept_query=True,
        mode="off",
    )
    assert out is None


def test_routing_mode_content_type_only_ignores_classifiers() -> None:
    """mode='content_type_only': only Rule 1 applies. Classifier flags
    are ignored entirely. Use to A/B the classifier without touching
    indexed data."""
    out_with_ct = select_query_instruct(
        content_type_filter="markdown",
        is_identifier_query=True,
        is_error_query=True,
        is_concept_query=True,
        mode="content_type_only",
    )
    assert out_with_ct == QUERY_INSTRUCT_DOCS

    # Without content_type_filter, every classifier flag goes to None.
    out_without_ct = select_query_instruct(
        content_type_filter=None,
        is_identifier_query=True,
        is_error_query=True,
        is_concept_query=True,
        mode="content_type_only",
    )
    assert out_without_ct is None


def test_routing_mode_auto_is_default_and_matches_omitted() -> None:
    """mode='auto' (default) and omitting mode give identical behavior."""
    args = dict(
        content_type_filter=None,
        is_identifier_query=True,
        is_error_query=False,
        is_concept_query=False,
    )
    assert select_query_instruct(**args) == select_query_instruct(**args, mode="auto")


# ---- CONTENT_TYPE_INSTRUCT lookup table ------------------------------------


def test_content_type_map_covers_documented_kinds() -> None:
    """The map must include the kinds the indexer emits as chunk
    metadata. If a new kind lands without an entry here, queries with
    that content_type_filter silently fall back to None / DEFAULT --
    which is a working but suboptimal prefix."""
    for kind in ("code", "yaml", "json", "markdown", "text", "openapi"):
        assert kind in CONTENT_TYPE_INSTRUCT, \
            f"CONTENT_TYPE_INSTRUCT missing key {kind!r}"


def test_content_type_map_values_are_distinct_instructs() -> None:
    """yaml/json both map to CONFIG (intentional); markdown/text/openapi
    all map to DOCS (intentional). But code → CODE must NOT alias to
    something else."""
    assert CONTENT_TYPE_INSTRUCT["code"] == QUERY_INSTRUCT_CODE
    assert CONTENT_TYPE_INSTRUCT["yaml"] == QUERY_INSTRUCT_CONFIG
    assert CONTENT_TYPE_INSTRUCT["json"] == QUERY_INSTRUCT_CONFIG
    assert CONTENT_TYPE_INSTRUCT["markdown"] == QUERY_INSTRUCT_DOCS
    assert CONTENT_TYPE_INSTRUCT["openapi"] == QUERY_INSTRUCT_DOCS


# ---- integration: HybridSearcher honors mode --------------------------------


@pytest.mark.asyncio
async def test_searcher_off_mode_sends_no_instruct(tmp_path) -> None:
    """With mode='off', every embed_query call gets instruct=None
    (the searcher does NOT route; format_query then applies DEFAULT)."""
    from code_rag.embedders.fake import FakeEmbedder
    from code_rag.rerankers.noop import NoopReranker
    from code_rag.retrieval.search import HybridSearcher, SearchParams
    from code_rag.stores.chroma_vector import ChromaVectorStore
    from code_rag.stores.sqlite_lexical import SqliteLexicalStore

    emb = FakeEmbedder(dim=32)
    captured: list[str | None] = []

    async def spy_embed_query(texts: Sequence[str], *, instruct: str | None = None):
        captured.append(instruct)
        return [[0.0] * 32 for _ in texts]

    emb.embed_query = spy_embed_query  # type: ignore[assignment]

    vec = ChromaVectorStore(tmp_path / "chroma", "test", tmp_path / "meta.json")
    lex = SqliteLexicalStore(tmp_path / "fts.db")
    meta = ChromaVectorStore.build_meta("fake", "fake-embedder-v1", 32)
    vec.open(meta)
    lex.open()
    try:
        s = HybridSearcher(
            emb, vec, lex, NoopReranker(), query_instruct_mode="off",
        )
        # "DbUp" should otherwise classify as identifier → CODE,
        # but mode=off forces None.
        await s.search("DbUp", SearchParams(k_final=2, k_vector=2, k_lexical=2, k_rerank_in=2))
    finally:
        vec.close()
        lex.close()
    assert captured, "embed_query was never called"
    assert all(i is None for i in captured), \
        f"mode='off' should pass instruct=None to embed_query; got {captured!r}"


@pytest.mark.asyncio
async def test_searcher_auto_mode_routes_identifier_to_code(tmp_path) -> None:
    """With mode='auto' and an identifier-shaped query, embed_query
    receives instruct=QUERY_INSTRUCT_CODE."""
    from code_rag.embedders.fake import FakeEmbedder
    from code_rag.rerankers.noop import NoopReranker
    from code_rag.retrieval.search import HybridSearcher, SearchParams
    from code_rag.stores.chroma_vector import ChromaVectorStore
    from code_rag.stores.sqlite_lexical import SqliteLexicalStore

    emb = FakeEmbedder(dim=32)
    captured: list[str | None] = []

    async def spy_embed_query(texts: Sequence[str], *, instruct: str | None = None):
        captured.append(instruct)
        return [[0.0] * 32 for _ in texts]

    emb.embed_query = spy_embed_query  # type: ignore[assignment]

    vec = ChromaVectorStore(tmp_path / "chroma", "test", tmp_path / "meta.json")
    lex = SqliteLexicalStore(tmp_path / "fts.db")
    meta = ChromaVectorStore.build_meta("fake", "fake-embedder-v1", 32)
    vec.open(meta)
    lex.open()
    try:
        s = HybridSearcher(
            emb, vec, lex, NoopReranker(), query_instruct_mode="auto",
        )
        # DbUp = short PascalCase, identifier-shaped → CODE.
        await s.search("DbUp", SearchParams(k_final=2, k_vector=2, k_lexical=2, k_rerank_in=2))
    finally:
        vec.close()
        lex.close()
    assert captured == [QUERY_INSTRUCT_CODE], \
        f"identifier-shaped query in auto mode should route to CODE; got {captured!r}"

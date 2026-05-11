"""Phase 60-R Phase 2: cheap regex-based query shape classifiers.

Three independent predicates that decide whether a query "looks like":

* a code identifier search (PascalCase / camelCase / snake_case / dotted
  namespace / C# interface / ticket ID),
* an error message or stack trace,
* a how-to / explanation / comparison question.

The outputs feed `embedders.prompts.select_query_instruct` which picks
the best Qwen3-Embedding instruct prefix per query shape. They are NOT
classifiers in the ML sense -- no LLM call, no token model. Just regex
over the literal query text. Cost is microseconds per query.

DESIGN NOTES

* Each classifier is independent. The router applies precedence; the
  classifiers don't know about each other. Keeping them orthogonal
  makes the test matrix small (one test per shape, plus precedence
  tests on the router side).
* The patterns are deliberately conservative on false-positives. A
  precise true-negative is worth more than a precise true-positive
  here: a wrongly-routed concept query (CODE instead of CONCEPT) just
  gets a slightly worse prefix, while an under-classified query falls
  through to DEFAULT which is still a working prefix.
* The regexes ARE pinned in tests with concrete strings on every edge
  the audit flagged (`DbUp`, `Datadog RUM in Next.js`, `NullReferenceException`,
  `SQLSTATE 23505`, `CS8602`, `vs.`, `X.Connect()`, etc.).
"""
from __future__ import annotations

import re

# ---- identifier detector ----------------------------------------------------
#
# Fires on code-shaped tokens. Each alternative captures a different
# identifier convention; together they catch the most common shapes in
# real codebases (.NET, JS/TS, Python, Terraform, Go) plus ticket IDs.
_IDENTIFIER_RE = re.compile(
    r"(?:"
    r"\b[A-Z]{2,6}-\d+\b"                          # MOD-2561, JIRA-123
    r"|[A-Z][a-zA-Z0-9]+(?::[A-Z][a-zA-Z0-9]+)+"   # Messaging:Transport
    r"|[A-Z]+[a-z]+(?:[A-Z]+[a-z0-9]+)+"           # DbUp, FastEndpoints, NCronJob
    r"|I[A-Z][a-z]+(?:[A-Z][a-z0-9]+)*"            # ILogger, IConfiguration
    r"|[a-z]+(?:[A-Z][a-z0-9]+)+"                  # totalDelay, getUser
    r"|[A-Za-z]\w+(?:\.[A-Za-z]\w+){1,}"           # Some.Dotted.Namespace
    r"|[a-z]\w*(?:_[a-z]\w*){2,}"                  # azurerm_postgresql_flexible
    r")"
)

# Domain suffixes that look like dotted namespaces but are actually
# brand/tech tokens ("ASP.NET", "Next.js", "TensorFlow.io"). When the
# only identifier-shaped match in the query is one of these, the query
# is NOT a code search.
_BRAND_SUFFIXES = frozenset({".net", ".js", ".ts", ".io", ".dev", ".ai", ".py"})

# Natural-language scaffolding words. When present, a single
# PascalCase brand token isn't enough to call this a code search --
# e.g. "How to configure Datadog RUM" should NOT route to CODE.
_NL_SCAFFOLD_RE = re.compile(
    r"\b(how|what|why|does|do|is|are|explain|mean|means)\b", re.I,
)


def is_identifier_query(query: str) -> bool:
    """True if `query` looks like a code-identifier search.

    Strategy:
      1. Find any identifier-shaped match.
      2. Apply heuristics to distinguish a real code query (short text +
         identifier, snake_case, camelCase, namespace, interface) from
         an NL query with an incidental brand (e.g. "How to use Datadog
         in Next.js").
    """
    m = _IDENTIFIER_RE.search(query)
    if not m:
        return False
    matched = m.group()
    words = query.split()
    # Pre-compute: is the matched span just a brand token? Used to demote
    # heuristic 1 and heuristic 4 (both shortcut to True when fired) so a
    # query like "ASP.NET Core dependency injection" doesn't route to CODE
    # just because "ASP.NET" matched the dotted-namespace regex.
    matched_is_brand = "." in matched and any(
        matched.lower().endswith(suffix) for suffix in _BRAND_SUFFIXES
    )
    # Heuristic 1: short query + a non-brand identifier → almost certainly
    # code. A short query whose only identifier-shaped token is a brand
    # suffix (e.g. "ASP.NET Core") drops through to the later heuristics.
    if len(words) <= 4 and not matched_is_brand:
        return True
    # Heuristic 2: snake_case is rare in NL, near-certain in code.
    if "_" in matched:
        return True
    # Heuristic 3: camelCase (lowercase-leading) almost never appears
    # in natural English; it's a code identifier.
    if matched[0].islower():
        return True
    # Heuristic 4: dotted namespace (but not a brand-suffix token).
    # See matched_is_brand above -- "ASP.NET", "Next.js" etc. are
    # brands, not code namespaces.
    if "." in matched and not matched_is_brand:
        return True
    # Heuristic 5: C#-style interface convention (I + Capitalized rest).
    if matched.startswith("I") and len(matched) > 1 and matched[1].isupper():
        return True
    # Heuristic 6: multiple PascalCase tokens AND no NL scaffolding.
    # "DbUp Foo Bar database migration" → true.
    # "ASP.NET Core dependency injection" → false (both caps are brand-
    # shaped, so they don't count toward the multi-PascalCase signal).
    caps = [
        w for w in words
        if w
        and w[0].isupper()
        and not any(w.lower().endswith(suffix) for suffix in _BRAND_SUFFIXES)
    ]
    if len(caps) >= 2 and not _NL_SCAFFOLD_RE.search(query):
        return True
    return False


# ---- error detector ---------------------------------------------------------
#
# Fires on pasted error messages, stack traces, SQLSTATE codes, compiler
# error codes, CORS errors, etc. These are queries where the user has
# pasted a literal error and wants a fix, NOT a how-to question.
_ERROR_PATTERN_RE = re.compile(
    r"(?:"
    r"ECONNREFUSED|ETIMEDOUT|ENOTFOUND|ECONNRESET"
    r"|(?:Error|Exception|Traceback)\s*[:(]"
    r"|at\s+\S+\s*\(\S+:\d+:\d+\)"               # JS stack trace
    r"|\b[A-Z]\w+(?:Error|Exception|Fault)\b"    # NullReferenceException
    r"|\b(?:ERR_|E_)\w+"                          # ERR_CONNECTION_REFUSED
    r"|errno\s*[:=]\s*\d+"
    r"|exit\s+code\s+\d+"
    r"|\bCS\d{4}\b"                               # C# compiler: CS8602
    r"|\bTS\d{4}\b"                               # TypeScript: TS2322
    r"|\b[0-9]{2}[A-Z0-9]{3}\b"                   # Postgres SQLSTATE: 23505
    r"|\bCORS\b.*\b(?:policy|origin|header)\b"
    r")", re.I,
)


def is_error_query(query: str) -> bool:
    """True if `query` looks like a pasted error message or stack trace."""
    return bool(_ERROR_PATTERN_RE.search(query))


# ---- concept detector -------------------------------------------------------
#
# Fires on how-to / explanation / comparison queries where the ideal
# answer is prose, not code. Distinct from code or error -- the router's
# precedence is identifier > error > concept so a "how does FastEndpoints"
# query goes to CODE not CONCEPT.
_CONCEPT_RE = re.compile(
    r"\b(how|what|why|explain|describe"
    r"|compare|comparison|differences?\s+between|vs\.?|versus"
    r"|when\s+to\s+use|trade-?offs?|pros\s+and\s+cons"
    r"|tutorial|overview|guide)\b", re.I,
)


def is_concept_query(query: str) -> bool:
    """True if `query` reads like a how-to / explanation / comparison."""
    return bool(_CONCEPT_RE.search(query))

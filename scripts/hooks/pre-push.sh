#!/usr/bin/env bash
# Phase 35 (B2): CI eval-gate pre-push hook.
#
# Refuses to push if eval-gate detects retrieval-quality regression vs
# the locked baseline. Catches regressions BEFORE they hit a remote.
#
# Install:
#   ln -s ../../scripts/hooks/pre-push.sh .git/hooks/pre-push
#   chmod +x .git/hooks/pre-push
#
# Skip (escape hatch — use sparingly):
#   git push --no-verify
#
# What it checks:
#   1. Tests pass (`pytest -q`).
#   2. eval-gate runs against the locked baseline. Allowed regression
#      threshold: 1.0 pp (matches the default eval-gate threshold).
#
# What it does NOT check:
#   * Cross-corpus eval — too slow for a pre-push hook.
#   * The full mined-eval fixture — same reason.
#   * Latency regressions — eval-gate does include p50/p95/p99 deltas
#     but doesn't fail on them by default; the recall metrics are the
#     hard floor.
#
# Bypass cases (legitimate):
#   * Documentation-only commits: nothing in src/ or tests/ touched.
#     The hook detects this and skips.
#   * No locked baseline yet (fresh install): hook prints a hint and
#     skips rather than blocking.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

PYTHON="${REPO_ROOT}/.venv/Scripts/python.exe"
if [ ! -x "$PYTHON" ]; then
    PYTHON="${REPO_ROOT}/.venv/bin/python"   # POSIX fallback
fi
if [ ! -x "$PYTHON" ]; then
    echo "[pre-push] ⚠  no .venv found; skipping CI eval-gate (install first)." >&2
    exit 0
fi

# --- 1. Bypass on doc-only commits ---
# Inputs to a pre-push hook are ranges of commits being pushed.
# Read stdin: each line is "<local_ref> <local_sha> <remote_ref> <remote_sha>".
range=""
while read -r local_ref local_sha remote_ref remote_sha; do
    if [ "$local_sha" = "0000000000000000000000000000000000000000" ]; then
        continue   # branch deletion
    fi
    if [ "$remote_sha" = "0000000000000000000000000000000000000000" ]; then
        # First push of this branch — diff vs the default upstream.
        base=$(git merge-base HEAD origin/main 2>/dev/null || echo "")
        if [ -n "$base" ]; then
            range="$base..$local_sha"
        else
            range="$local_sha"
        fi
    else
        range="${remote_sha}..${local_sha}"
    fi
done

if [ -n "$range" ]; then
    if ! git diff --name-only "$range" 2>/dev/null | grep -qE '^(src/|tests/|pyproject\.toml$)'; then
        echo "[pre-push] ✓  no src/ or tests/ changes in $range — skipping CI eval-gate."
        exit 0
    fi
fi

# --- 2. Tests must pass ---
echo "[pre-push] running test suite ..."
if ! "$PYTHON" -m pytest -q --tb=line 2>&1 | tail -5; then
    echo "[pre-push] ✗  tests failed — refusing push." >&2
    echo "[pre-push] (re-run with: git push --no-verify  if you genuinely need to bypass)" >&2
    exit 1
fi

# --- 3. Eval-gate (only if locked baseline exists AND retrieval files changed) ---
LOCKED="$REPO_ROOT/data/eval/baseline_locked.json"
if [ ! -f "$LOCKED" ]; then
    echo "[pre-push] ⚠  no locked baseline at $LOCKED — skipping eval-gate."
    echo "[pre-push]    Run: code-rag eval-gate --write-baseline   to lock one."
    exit 0
fi

# Only run eval if a retrieval-relevant file changed.
if [ -n "$range" ]; then
    if ! git diff --name-only "$range" 2>/dev/null | grep -qE '^src/code_rag/(retrieval|rerankers|embedders|stores|indexing|eval)/'; then
        echo "[pre-push] ✓  no retrieval changes — skipping eval-gate."
        exit 0
    fi
fi

echo "[pre-push] running eval-gate ..."
if ! "$PYTHON" -m code_rag eval-gate --max-regression-pp 1.0; then
    echo "[pre-push] ✗  eval-gate detected regression — refusing push." >&2
    echo "[pre-push] (run: code-rag eval-gate   to see details)" >&2
    echo "[pre-push] (bypass: git push --no-verify  ONLY if intentional)" >&2
    exit 1
fi

echo "[pre-push] ✓  CI gate passed."

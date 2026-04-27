#!/usr/bin/env bash
# Phase 35 (B2): install the pre-push CI hook.
#
# Idempotent: run multiple times without harm. Symlinks .git/hooks/pre-push
# at the repo's tracked hook script so users get updates via `git pull`
# rather than having stale local copies of the hook.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOK_SRC="$REPO_ROOT/scripts/hooks/pre-push.sh"
HOOK_DST="$REPO_ROOT/.git/hooks/pre-push"

if [ ! -f "$HOOK_SRC" ]; then
    echo "✗ source hook missing: $HOOK_SRC" >&2
    exit 1
fi

# .git/hooks must exist (it does on any git repo, but defensive).
mkdir -p "$REPO_ROOT/.git/hooks"

# Replace any existing hook (could be a stale symlink or an old copy).
rm -f "$HOOK_DST"

# Prefer symlink so updates propagate via `git pull`. On Windows / older git
# without symlink support, fall back to a copy.
if ln -s "../../scripts/hooks/pre-push.sh" "$HOOK_DST" 2>/dev/null; then
    echo "✓ installed pre-push hook (symlink → scripts/hooks/pre-push.sh)"
else
    cp "$HOOK_SRC" "$HOOK_DST"
    chmod +x "$HOOK_DST"
    echo "✓ installed pre-push hook (copy)"
fi

echo "  to bypass for one push:    git push --no-verify"
echo "  to uninstall:               rm $HOOK_DST"

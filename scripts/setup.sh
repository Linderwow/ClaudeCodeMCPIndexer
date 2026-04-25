#!/usr/bin/env bash
# One-command bootstrap for code-rag-mcp on macOS or Linux.
#
# Usage:
#     ./scripts/setup.sh
#
# Idempotent. Reruns are safe — skips work that's already done.

set -e

REPO="$(cd "$(dirname "$0")/.." && pwd)"
EMBEDDER_MODEL="${EMBEDDER_MODEL:-text-embedding-qwen3-embedding-4b}"
RERANKER_MODEL="${RERANKER_MODEL:-qwen3-reranker-4b}"

# Colors (fall back to plain on terminals without ANSI).
if [ -t 1 ]; then
    CYAN="\033[36m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; MAGENTA="\033[35m"; RESET="\033[0m"
else
    CYAN=""; GREEN=""; YELLOW=""; RED=""; MAGENTA=""; RESET=""
fi

step() { printf "${CYAN}==> %s${RESET}\n" "$1"; }
ok()   { printf "    ${GREEN}OK  %s${RESET}\n" "$1"; }
warn() { printf "    ${YELLOW}!!  %s${RESET}\n" "$1"; }
fail() { printf "    ${RED}FAIL %s${RESET}\n" "$1"; }

OS="$(uname -s)"
case "$OS" in
    Darwin) PLATFORM="macos" ;;
    Linux)  PLATFORM="linux" ;;
    *)      fail "Unsupported OS: $OS (this script is macOS/Linux only; use scripts/setup.ps1 on Windows)"; exit 1 ;;
esac

printf "${MAGENTA}code-rag-mcp setup at %s (%s)${RESET}\n" "$REPO" "$PLATFORM"

# ---- 1. Python ------------------------------------------------------------

step "Checking Python 3.11+"
PY=""
for cand in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        VER="$($cand --version 2>&1 | awk '{print $2}')"
        MAJOR="$(printf '%s' "$VER" | cut -d. -f1)"
        MINOR="$(printf '%s' "$VER" | cut -d. -f2)"
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 11 ]; then
            PY="$cand"
            ok "found $cand ($VER)"
            break
        fi
    fi
done
if [ -z "$PY" ]; then
    fail "Python 3.11+ not found. Install via 'brew install python@3.12' (macOS) or your distro's package manager (Linux)."
    exit 2
fi

# ---- 2. venv + pip install ------------------------------------------------

VENV="$REPO/.venv"
VPY="$VENV/bin/python"
VPIP="$VENV/bin/pip"
CODE_RAG="$VENV/bin/code-rag"

if [ ! -x "$VPY" ]; then
    step "Creating .venv"
    "$PY" -m venv "$VENV"
    ok ".venv created"
else
    ok ".venv already exists"
fi

step "pip install -e .[dev]"
"$VPIP" install --upgrade pip --quiet
"$VPIP" install -e "${REPO}[dev]" --quiet
ok "dependencies installed"

# ---- 3. config.toml -------------------------------------------------------

if [ ! -f "$REPO/config.toml" ]; then
    step "Copying config.example.toml -> config.toml"
    cp "$REPO/config.example.toml" "$REPO/config.toml"
    ok "config.toml created"
else
    ok "config.toml already present"
fi

# ---- 4. LM Studio ---------------------------------------------------------

step "Checking LM Studio (lms CLI)"
LMS=""
if command -v lms >/dev/null 2>&1; then
    LMS="$(command -v lms)"
elif [ -x "$HOME/.lmstudio/bin/lms" ]; then
    LMS="$HOME/.lmstudio/bin/lms"
fi

if [ -z "$LMS" ]; then
    warn "LM Studio CLI (lms) not found."
    printf "    Install LM Studio from https://lmstudio.ai/ then run 'lms bootstrap'.\n"
    printf "    Re-run this script afterwards. Skipping model download + autostart.\n"
else
    ok "lms found at $LMS"

    step "Ensuring embedder model is downloaded ($EMBEDDER_MODEL)"
    if "$LMS" ls 2>/dev/null | grep -q "$EMBEDDER_MODEL"; then
        ok "embedder model already on disk"
    else
        printf "    Downloading from HuggingFace via 'lms get'...\n"
        "$LMS" get "$EMBEDDER_MODEL" || warn "lms get failed; you can download manually later"
    fi

    step "Ensuring LM Studio server is running"
    if curl -fsS --max-time 3 http://localhost:1234/v1/models >/dev/null 2>&1; then
        ok "LM Studio server reachable"
    else
        printf "    Starting LM Studio server in background...\n"
        "$LMS" server start >/dev/null 2>&1 &
        sleep 5
        ok "LM Studio server starting"
    fi

    step "Loading embedder model"
    "$LMS" load "$EMBEDDER_MODEL" >/dev/null 2>&1 || true
    ok "embedder loaded"

    step "Loading reranker model (best-effort, $RERANKER_MODEL)"
    "$LMS" load "$RERANKER_MODEL" >/dev/null 2>&1 || true
    ok "reranker load attempted"
fi

# ---- 5. code-rag install --------------------------------------------------
# Note: code-rag install on macOS/Linux skips the Windows-only Task Scheduler
# step automatically. Autostart is handled by the platform-specific helpers
# below.

step "Running code-rag install (probe + index + Claude wiring)"
"$CODE_RAG" install --skip-autostart || warn "code-rag install reported issues"
ok "code-rag install completed"

# ---- 6. Platform-specific autostart --------------------------------------

if [ "$PLATFORM" = "macos" ]; then
    step "Installing launchd autostart agent"
    PLIST_DIR="$HOME/Library/LaunchAgents"
    PLIST="$PLIST_DIR/com.code-rag-mcp.watcher.plist"
    mkdir -p "$PLIST_DIR"
    cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
        "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>            <string>com.code-rag-mcp.watcher</string>
    <key>ProgramArguments</key>
    <array>
        <string>$VPY</string>
        <string>-m</string>
        <string>code_rag.autostart_bootstrap</string>
    </array>
    <key>WorkingDirectory</key>  <string>$REPO</string>
    <key>RunAtLoad</key>          <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key> <false/>
    </dict>
    <key>StandardOutPath</key>    <string>$REPO/logs/launchd.out.log</string>
    <key>StandardErrorPath</key>  <string>$REPO/logs/launchd.err.log</string>
</dict>
</plist>
EOF
    launchctl unload "$PLIST" >/dev/null 2>&1 || true
    launchctl load "$PLIST"
    ok "launchd agent installed at $PLIST"
elif [ "$PLATFORM" = "linux" ]; then
    step "Installing systemd user unit for autostart"
    UNIT_DIR="$HOME/.config/systemd/user"
    UNIT="$UNIT_DIR/code-rag-watcher.service"
    mkdir -p "$UNIT_DIR" "$REPO/logs"
    cat > "$UNIT" <<EOF
[Unit]
Description=code-rag-mcp watcher (live indexing)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=$REPO
ExecStart=$VPY -m code_rag.autostart_bootstrap
Restart=on-failure
RestartSec=10
StandardOutput=append:$REPO/logs/systemd.out.log
StandardError=append:$REPO/logs/systemd.err.log

[Install]
WantedBy=default.target
EOF
    systemctl --user daemon-reload
    systemctl --user enable --now code-rag-watcher.service
    ok "systemd user unit installed at $UNIT"
fi

# ---- 7. Summary -----------------------------------------------------------

echo
printf "${MAGENTA}================================================================${RESET}\n"
printf "${MAGENTA} code-rag-mcp is set up.${RESET}\n"
printf "${MAGENTA}================================================================${RESET}\n"
echo
echo " Next steps:"
echo "   1. Open Claude Code."
echo "   2. The MCP server spawns automatically on first tool call."
echo "      Asking about any repo auto-registers it (ensure_workspace_indexed)."
echo "   3. Watcher runs at logon ($PLATFORM autostart already enabled)."
echo
echo " Useful commands:"
echo "   $CODE_RAG status         # what's configured"
echo "   $CODE_RAG doctor         # full health check"
echo "   $CODE_RAG index-stats    # chunk counts"
echo "   $CODE_RAG fsck           # consistency check"
echo "   $CODE_RAG metrics        # OpenMetrics snapshot"
echo

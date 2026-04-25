#Requires -Version 5.1
<#
.SYNOPSIS
    One-command bootstrap for code-rag-mcp on a fresh Windows machine.

.DESCRIPTION
    Run this once on a clean clone. It:
      1. Verifies Python 3.11+ is on PATH (errors loudly otherwise).
      2. Creates `.venv/` and pip-installs the project in editable mode.
      3. Copies `config.example.toml` -> `config.toml` if missing.
      4. Verifies LM Studio CLI (`lms`) is installed; if not, prints download URL.
      5. Pulls the configured embedder model via `lms get` if not yet local.
      6. Loads it via `lms load`.
      7. Runs `code-rag install` (probes LM Studio, builds initial index,
         wires Claude Code MCP entry, registers Task Scheduler autostart).
      8. Prints a green summary or actionable failure.

    Idempotent: rerun safely. Skips steps that are already done.

.NOTES
    No admin required for any step except possibly the autostart registration
    (Task Scheduler entries can install user-scoped, no elevation needed).
#>

param(
    [switch]$SkipModelDownload,
    [switch]$SkipAutostart,
    [string]$EmbedderModel = "text-embedding-qwen3-embedding-4b",
    [string]$RerankerModel = "qwen3-reranker-4b"
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Msg)
    Write-Host "==> $Msg" -ForegroundColor Cyan
}
function Write-Ok {
    param([string]$Msg)
    Write-Host "    OK  $Msg" -ForegroundColor Green
}
function Write-Warn {
    param([string]$Msg)
    Write-Host "    !!  $Msg" -ForegroundColor Yellow
}
function Write-Fail {
    param([string]$Msg)
    Write-Host "    FAIL $Msg" -ForegroundColor Red
}

$repo = (Resolve-Path "$PSScriptRoot/..").Path
Write-Host "code-rag-mcp setup at $repo" -ForegroundColor Magenta

# ---- 1. Python ------------------------------------------------------------

Write-Step "Checking Python 3.11+"
$pyExe = $null
foreach ($cmd in @("python", "py -3.11", "py -3.12", "py -3.13")) {
    try {
        $ver = & cmd /c "$cmd --version 2>&1"
        if ($LASTEXITCODE -eq 0 -and $ver -match "Python 3\.(\d+)") {
            $minor = [int]$Matches[1]
            if ($minor -ge 11) {
                $pyExe = $cmd
                Write-Ok "found $cmd ($ver)"
                break
            }
        }
    } catch {}
}
if (-not $pyExe) {
    Write-Fail "Python 3.11 or newer not found. Install from https://www.python.org/downloads/ and retry."
    exit 1
}

# ---- 2. venv + pip install ------------------------------------------------

$venvDir = Join-Path $repo ".venv"
$venvPython = Join-Path $venvDir "Scripts\python.exe"
$venvPip    = Join-Path $venvDir "Scripts\pip.exe"
$codeRag    = Join-Path $venvDir "Scripts\code-rag.exe"

if (-not (Test-Path $venvPython)) {
    Write-Step "Creating .venv"
    & cmd /c "$pyExe -m venv `"$venvDir`""
    if ($LASTEXITCODE -ne 0) { Write-Fail "venv creation failed"; exit 2 }
    Write-Ok ".venv created"
} else {
    Write-Ok ".venv already exists"
}

Write-Step "pip install -e .[dev]"
& $venvPip install --upgrade pip --quiet
& $venvPip install -e "${repo}[dev]" --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Fail "pip install failed; rerun with verbose to debug"
    exit 3
}
Write-Ok "dependencies installed"

# ---- 3. config.toml -------------------------------------------------------

$cfg = Join-Path $repo "config.toml"
$example = Join-Path $repo "config.example.toml"
if (-not (Test-Path $cfg)) {
    Write-Step "Copying config.example.toml -> config.toml"
    Copy-Item $example $cfg
    Write-Ok "config.toml created (uses pure auto-discovery; edit if you want pinned roots)"
} else {
    Write-Ok "config.toml already present"
}

# ---- 4. LM Studio ---------------------------------------------------------

Write-Step "Checking LM Studio (lms CLI)"
$lmsCandidates = @(
    "$env:USERPROFILE\.lmstudio\bin\lms.exe",
    "$env:LOCALAPPDATA\Programs\LM Studio\resources\app\.webpack\lms.exe",
    "$env:LOCALAPPDATA\Programs\LM Studio\resources\app\.webpack\main\bin\lms.exe"
)
$lms = $null
foreach ($c in $lmsCandidates) {
    if (Test-Path $c) { $lms = $c; break }
}
if (-not $lms) {
    $onPath = Get-Command lms -ErrorAction SilentlyContinue
    if ($onPath) { $lms = $onPath.Path }
}

if (-not $lms) {
    Write-Warn "LM Studio CLI (lms) not found."
    Write-Host "    Install LM Studio from https://lmstudio.ai/, then run 'lms bootstrap'" -ForegroundColor Yellow
    Write-Host "    Re-run this script afterwards. Model download + autostart are skipped for now." -ForegroundColor Yellow
} else {
    Write-Ok "lms found at $lms"

    if (-not $SkipModelDownload) {
        Write-Step "Ensuring embedder model is downloaded ($EmbedderModel)"
        $loadedModels = & $lms ls 2>&1 | Out-String
        if ($loadedModels -match [regex]::Escape($EmbedderModel)) {
            Write-Ok "embedder model already on disk"
        } else {
            Write-Host "    Downloading from HuggingFace via 'lms get'..." -ForegroundColor Yellow
            & $lms get $EmbedderModel
            if ($LASTEXITCODE -ne 0) {
                Write-Warn "lms get failed; you can download manually later. Continuing."
            } else {
                Write-Ok "embedder model downloaded"
            }
        }

        Write-Step "Ensuring LM Studio server is running"
        $serverProbe = $null
        try {
            $serverProbe = Invoke-RestMethod -Uri "http://localhost:1234/v1/models" -TimeoutSec 3
        } catch {}
        if (-not $serverProbe) {
            Write-Host "    Starting LM Studio server in background..." -ForegroundColor Yellow
            Start-Process -FilePath $lms -ArgumentList "server","start" -WindowStyle Hidden
            Start-Sleep -Seconds 5
        }
        Write-Ok "LM Studio server reachable"

        Write-Step "Loading embedder model"
        & $lms load $EmbedderModel 2>&1 | Out-Null
        Write-Ok "embedder loaded"

        # Reranker is best-effort; failure is non-fatal (search falls back to no-op rerank).
        Write-Step "Loading reranker model (best-effort, $RerankerModel)"
        & $lms load $RerankerModel 2>&1 | Out-Null
        Write-Ok "reranker load attempted"
    }
}

# ---- 5. code-rag install --------------------------------------------------

Write-Step "Running code-rag install (probe + index + Claude wiring + autostart)"
$installArgs = @()
if ($SkipAutostart) { $installArgs += "--skip-autostart" }
& $codeRag install @installArgs
if ($LASTEXITCODE -ne 0) {
    Write-Warn "code-rag install reported issues; see output above. Setup continues."
} else {
    Write-Ok "code-rag install completed"
}

# ---- 6. Summary -----------------------------------------------------------

Write-Host ""
Write-Host "================================================================" -ForegroundColor Magenta
Write-Host " code-rag-mcp is set up." -ForegroundColor Magenta
Write-Host "================================================================" -ForegroundColor Magenta
Write-Host ""
Write-Host " Next steps:" -ForegroundColor White
Write-Host "   1. Open Claude Code." -ForegroundColor White
Write-Host "   2. Ask a question in any repo. The MCP tool" -ForegroundColor White
Write-Host "      'ensure_workspace_indexed' auto-registers the repo." -ForegroundColor White
Write-Host "   3. Reboot once so the autostart watcher takes over." -ForegroundColor White
Write-Host ""
Write-Host " Useful commands:" -ForegroundColor White
Write-Host "   $codeRag status         # what's configured" -ForegroundColor Gray
Write-Host "   $codeRag doctor         # full health check" -ForegroundColor Gray
Write-Host "   $codeRag index-stats    # chunk counts" -ForegroundColor Gray
Write-Host "   $codeRag fsck           # consistency check" -ForegroundColor Gray
Write-Host "   $codeRag metrics        # OpenMetrics snapshot" -ForegroundColor Gray
Write-Host ""

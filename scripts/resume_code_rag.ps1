# Phase 60: scheduled resume of the code-rag stack.
#
# Replaces the old "let LM Studio autostart bring it up" flow. With Phase 60
# the embedder runs as vLLM inside WSL Ubuntu (port 8000 = embed,
# port 8001 = rerank). This script:
#   1. Launches both vLLM servers via setsid+nohup so they survive the
#      schtasks shell exiting.
#   2. Waits up to 3 min for /v1/models on both ports to respond.
#   3. Restarts the bulk indexer to resume any reindex from file_hashes.db.
#   4. Logs everything to ~/Documents/code-rag-mcp/logs/resume_TIMESTAMP.log
#
# Usage (from anywhere):
#   powershell.exe -NoProfile -ExecutionPolicy Bypass -File <this-script>
#
# Wired up by `schtasks /create /sc once /st 21:00 ...` for the 9 PM resume.

$ErrorActionPreference = 'Continue'
$ts = Get-Date -Format 'yyyyMMdd_HHmmss'
$logDir = "$env:USERPROFILE\Documents\code-rag-mcp\logs"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$log = Join-Path $logDir "resume_$ts.log"

function Log($msg) {
    $line = "[$([DateTime]::Now.ToString('HH:mm:ss'))] $msg"
    $line | Out-File -FilePath $log -Append -Encoding utf8
    Write-Host $line
}

Log "=== Phase 60 resume start ==="
Log "log: $log"

# ----- 1. Launch vLLM-embed inside WSL ---------------------------------------
Log "launching vLLM-embed (port 8000)..."
$embedCmd = "setsid nohup bash `$HOME/bin/serve-qwen3-embed-8b.sh > /tmp/vllm-embed.log 2>&1 < /dev/null &"
& wsl.exe -d Ubuntu --user linderwow -e bash -c $embedCmd
Log "  embed launched"

# ----- 2. Launch vLLM-rerank inside WSL --------------------------------------
Log "launching vLLM-rerank (port 8001)..."
$rerankCmd = "setsid nohup bash `$HOME/bin/serve-rerank-bge-m3.sh > /tmp/vllm-rerank.log 2>&1 < /dev/null &"
& wsl.exe -d Ubuntu --user linderwow -e bash -c $rerankCmd
Log "  rerank launched"

# ----- 3. Poll until both endpoints respond (max 180 s each) -----------------
function WaitForPort($port, $name) {
    Log "waiting for $name on port $port..."
    for ($i = 0; $i -lt 36; $i++) {
        try {
            $r = Invoke-WebRequest -Uri "http://localhost:$port/v1/models" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
            if ($r.StatusCode -eq 200) {
                Log "  $name UP after $($i*5)s"
                return $true
            }
        } catch { }
        Start-Sleep -Seconds 5
    }
    Log "  $name TIMEOUT after 180s"
    return $false
}

$embedUp  = WaitForPort 8000 "vLLM-embed"
$rerankUp = WaitForPort 8001 "vLLM-rerank"

if (-not $embedUp) {
    Log "ERROR: vLLM-embed never came up. Bailing -- indexer would just spam errors."
    exit 1
}

# ----- 4. Snapshot VRAM before kicking the indexer ---------------------------
$vram = & wsl.exe -d Ubuntu -e bash -c "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader" 2>&1
Log "VRAM after vLLM up: $vram"

# ----- 5. Restart bulk indexer (resumes via file_hashes.db) ------------------
# Phase 60 audit fix: previous draft used Process.Start with
# RedirectStandardOutput/Error=true but never DRAINED the pipes -- Windows
# pipe buffers fill at ~64 KB each and the indexer would block on stdout
# once it logged enough HTTP requests. Fix: use Start-Process with file
# redirection so the OS handles the bytes; we never touch the pipes.
$repoRoot = "$env:USERPROFILE\Documents\code-rag-mcp"
$venvPy   = Join-Path $repoRoot ".venv\Scripts\python.exe"
$idxLog   = Join-Path $logDir "indexer_$ts.log"
$idxErr   = Join-Path $logDir "indexer_$ts.err"
Log "restarting bulk indexer (will skip already-done files via file_hashes.db)..."
$proc = Start-Process -FilePath $venvPy `
                      -ArgumentList @('-m', 'code_rag', 'index') `
                      -WorkingDirectory $repoRoot `
                      -RedirectStandardOutput $idxLog `
                      -RedirectStandardError $idxErr `
                      -WindowStyle Hidden `
                      -PassThru
Log "  indexer PID=$($proc.Id), stdout -> $idxLog, stderr -> $idxErr"

# ----- 6. Done ---------------------------------------------------------------
Log "=== Phase 60 resume complete ==="
Log "vLLM-embed: $(if($embedUp){'UP'}else{'DOWN'})"
Log "vLLM-rerank: $(if($rerankUp){'UP'}else{'DOWN (degraded; cross-encoder rerank unavailable)'})"
Log "Indexer PID: $($proc.Id)"
exit 0

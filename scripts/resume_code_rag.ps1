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
# Audit fix: after each wsl.exe call, poll for the log file's existence
# inside WSL. If wsl.exe bounces (Task Scheduler context can return without
# actually dispatching the WSL job), the bash redirect never fires and the
# log is missing. Detecting this in 10s saves us from the 180s /v1/models
# timeout downstream.
function LaunchAndConfirm($script_name, $log_path_in_wsl) {
    # Audit-2 round-3 fix: when wsl.exe is invoked from PowerShell (Task
    # Scheduler context), wsl.exe tears down the foreground bash session
    # as soon as the bash exits — and the backgrounded `setsid nohup ... &`
    # process gets killed before it can fully detach. Empirically the
    # working pattern is `& disown ; sleep 3` (or longer): disown removes
    # the job from bash's table so SIGHUP doesn't propagate, and the sleep
    # keeps the WSL session alive long enough for the launched process
    # to settle into its own session. Without these two, the launch fires
    # but the redirect target file never gets created (the launched bash
    # is killed before it opens the file).
    & wsl.exe -d Ubuntu -e bash -c "rm -f $log_path_in_wsl; setsid nohup bash `$HOME/bin/$script_name > $log_path_in_wsl 2>&1 < /dev/null & disown; sleep 3"
    for ($j = 0; $j -lt 10; $j++) {
        $exists = & wsl.exe -d Ubuntu -e bash -c "test -f $log_path_in_wsl && echo OK"
        if ($exists -match 'OK') { return $true }
        Start-Sleep -Seconds 1
    }
    return $false
}

Log "launching vLLM-embed (port 8000)..."
if (LaunchAndConfirm 'serve-qwen3-embed-8b.sh' '/tmp/vllm-embed.log') {
    Log "  embed launched (log file present in WSL)"
} else {
    Log "FATAL: wsl.exe didn't dispatch vLLM-embed launch -- log file never created. Bailing."
    exit 2
}

# Audit fix: SEQUENTIAL launch. Original draft launched embed + rerank in
# parallel — both processes computed --gpu-memory-utilization fractions
# from the SAME free-VRAM number, but their actual peak loads (CUDA graph
# capture spikes) overlap and one (usually embed, which loads later but
# requests more) crashes with "Free memory ... less than desired GPU memory
# utilization". Waiting for embed to be /v1/models-200 means rerank only
# starts loading once embed has settled into its steady-state VRAM
# footprint, which is what the per-server mem-util calculation actually
# expects.
Log "waiting for vLLM-embed to settle BEFORE launching rerank (avoid VRAM-contention crash)..."
$earlyEmbedUp = $false
for ($k = 0; $k -lt 36; $k++) {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:8000/v1/models" -TimeoutSec 3 -UseBasicParsing -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $earlyEmbedUp = $true; Log "  embed settled after $($k*5)s, launching rerank now"; break }
    } catch { }
    Start-Sleep -Seconds 5
}
if (-not $earlyEmbedUp) {
    Log "WARN: embed not up after 180s during pre-rerank wait; launching rerank anyway"
}

# ----- 2. Launch vLLM-rerank inside WSL --------------------------------------
Log "launching vLLM-rerank (port 8001)..."
if (LaunchAndConfirm 'serve-rerank-bge-m3.sh' '/tmp/vllm-rerank.log') {
    Log "  rerank launched (log file present in WSL)"
} else {
    Log "WARN: rerank launch undetectable in 10s; continuing (rerank is best-effort)"
}

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

# Phase 36: install the self-healing watchdog tasks.
#
# Three new Task Scheduler entries on top of the existing
# code-rag-watch / code-rag-dashboard / code-rag-reap:
#
#   code-rag-chroma-heal    every 10 min   probe Chroma; wipe if deadlocked
#   code-rag-chroma-defrag  weekly         wipe + reindex if data_level0 > 5 GB
#   code-rag-lms-enforce    every  1 hr    re-apply Phase 33 settings if drifted
#
# Together with the upgraded code-rag-reap (now also checks watcher
# heartbeat + unloads duplicate LM Studio aliases), these cover the
# full failure-mode set documented in the May 3 post-mortem:
#   - Chroma deadlock → chroma-heal
#   - LM Studio crash + reload with default settings → lms-enforce
#   - Watcher process wedged but alive → reap (heartbeat check)
#   - Watcher dead + missed Task Scheduler restart → reap (auto-respawn)
#   - Duplicate LM Studio model aliases → reap (auto-unload)
#   - HNSW data_level0.bin bloat → chroma-defrag
#
# All tasks idempotent; safe to install / re-install / leave running.
#
# Run from a normal PowerShell:
#   .\scripts\install-watchdog-autostart.ps1
#
# Uninstall (one-line):
#   foreach ($n in 'code-rag-chroma-heal','code-rag-chroma-defrag','code-rag-lms-enforce') {
#       Unregister-ScheduledTask -TaskName $n -Confirm:$false -EA SilentlyContinue
#   }

$ErrorActionPreference = 'Stop'

$RepoRoot    = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$VenvPythonW = Join-Path $RepoRoot '.venv\Scripts\pythonw.exe'

if (-not (Test-Path $VenvPythonW)) {
    Write-Error "pythonw.exe not found at $VenvPythonW. Set up the venv first."
    exit 1
}

function Install-RepeatingTask {
    param(
        [string]$Name,
        [string]$Argument,
        [int]$IntervalMinutes,
        [int]$ExecLimitMinutes,
        [string]$Description
    )

    if (Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $Name -Confirm:$false
        Write-Host "  removed existing $Name"
    }

    $action = New-ScheduledTaskAction `
        -Execute $VenvPythonW `
        -Argument $Argument `
        -WorkingDirectory $RepoRoot

    # AtLogOn for the bootstrap firing, repetition for ongoing cadence.
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $trigger.Repetition = (New-ScheduledTaskTrigger `
        -Once -At (Get-Date) `
        -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
        -RepetitionDuration (New-TimeSpan -Days 9999)).Repetition

    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Minutes $ExecLimitMinutes) `
        -MultipleInstances IgnoreNew

    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive

    Register-ScheduledTask `
        -TaskName $Name `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description $Description |
      Out-Null

    Write-Host ("  installed $Name (every $IntervalMinutes min)")
}

Write-Host "Phase 36 watchdog tasks:"
Install-RepeatingTask `
    -Name 'code-rag-chroma-heal' `
    -Argument '-m code_rag chroma-heal --quiet' `
    -IntervalMinutes 10 -ExecLimitMinutes 5 `
    -Description "code-rag chroma deadlock detector + auto-wipe (Phase 36-A)"

Install-RepeatingTask `
    -Name 'code-rag-lms-enforce' `
    -Argument '-m code_rag lms-enforce-settings --apply --quiet' `
    -IntervalMinutes 60 -ExecLimitMinutes 5 `
    -Description "code-rag re-apply Phase 33 LM Studio settings (Phase 36-B)"

# Defrag: weekly. Task Scheduler can't easily express "weekly via repetition";
# we use a daily trigger with the script gating itself via --check-only first.
Install-RepeatingTask `
    -Name 'code-rag-chroma-defrag' `
    -Argument '-m code_rag chroma-defrag' `
    -IntervalMinutes 1440 -ExecLimitMinutes 180 `
    -Description "code-rag chroma defrag if data_level0 > 5 GB (Phase 36-D)"

Write-Host ""
Write-Host "Done. Tasks running. Ad-hoc invocations:"
Write-Host "  code-rag chroma-heal                  # probe + heal once"
Write-Host "  code-rag chroma-heal --dry-run        # probe only"
Write-Host "  code-rag chroma-defrag --check-only   # report sizes"
Write-Host "  code-rag lms-enforce-settings --apply # reload mismatched models"

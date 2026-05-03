# Phase 37-I: register a daily eval-gate run with drift detection.
#
# What this installs:
#   Task name 1: code-rag-eval-gate
#     Trigger:   Daily at 02:30 (low-load window)
#     Action:    pythonw.exe -m code_rag eval-gate --label scheduled
#                -- writes a row to data/eval/history.jsonl
#
#   Task name 2: code-rag-eval-drift
#     Trigger:   Daily at 02:45 (15 min after eval-gate so it has fresh data)
#     Action:    pythonw.exe -m code_rag eval-drift-check
#                -- exits non-zero on regressions; Phase 37-J's dashboard
#                   alerter watches the resulting status file.
#   Window:    none (pythonw.exe has no console)
#
# Why this exists:
#   eval-gate gives a one-shot pass/fail, but quality drift over a week
#   doesn't trip the gate. Running daily and detecting median-window
#   regression means a 2pp R@1 erosion in a week is flagged automatically
#   instead of waiting for someone to notice.
#
# Run from a normal PowerShell (no admin needed):
#   .\scripts\install-eval-cron.ps1
#
# Uninstall:
#   Unregister-ScheduledTask -TaskName 'code-rag-eval-gate'  -Confirm:$false
#   Unregister-ScheduledTask -TaskName 'code-rag-eval-drift' -Confirm:$false

$ErrorActionPreference = 'Stop'

$RepoRoot    = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$VenvPythonW = Join-Path $RepoRoot '.venv\Scripts\pythonw.exe'

if (-not (Test-Path $VenvPythonW)) {
    Write-Error "pythonw.exe not found at $VenvPythonW. Set up the venv first."
    exit 1
}

function Install-Task {
    param(
        [string]$TaskName,
        [string]$Argument,
        [datetime]$RunAt,
        [string]$Description,
        [int]$TimeLimitMinutes = 30
    )

    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed existing task '$TaskName'."
    }

    $action = New-ScheduledTaskAction `
        -Execute $VenvPythonW `
        -Argument $Argument `
        -WorkingDirectory $RepoRoot

    # Daily trigger at the chosen time. StartWhenAvailable catches up if the
    # machine was off at 02:30 (laptop closed overnight).
    $trigger = New-ScheduledTaskTrigger -Daily -At $RunAt

    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -ExecutionTimeLimit (New-TimeSpan -Minutes $TimeLimitMinutes) `
        -MultipleInstances IgnoreNew

    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive

    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description $Description | Out-Null

    Write-Host ("Installed task '" + $TaskName + "' (runs daily at " + $RunAt.ToString('HH:mm') + ").")
}

# Daily eval-gate at 02:30. A 30-minute window is generous — the 40-case
# fixture takes <60s on this machine and even on a degraded GPU < 5 minutes.
Install-Task `
    -TaskName 'code-rag-eval-gate' `
    -Argument '-m code_rag eval-gate --label scheduled' `
    -RunAt (Get-Date '02:30') `
    -Description 'Phase 37-I: daily eval-gate run; appends a row to data/eval/history.jsonl.'

# Drift check 15 min later — gives eval-gate plenty of time to finish + flush.
Install-Task `
    -TaskName 'code-rag-eval-drift' `
    -Argument ('-m code_rag eval-drift-check --quiet --json-out ' + `
               '"' + (Join-Path $RepoRoot 'data\eval\drift-state.json') + '"') `
    -RunAt (Get-Date '02:45') `
    -Description 'Phase 37-I: daily drift detector; non-zero exit on regression. Watches data/eval/history.jsonl.' `
    -TimeLimitMinutes 5

Write-Host ''
Write-Host 'Manual runs:'
Write-Host ('  Start-ScheduledTask -TaskName code-rag-eval-gate')
Write-Host ('  Start-ScheduledTask -TaskName code-rag-eval-drift')
Write-Host 'Drift report:'
Write-Host ('  ' + $VenvPythonW + ' -m code_rag eval-drift-check')
Write-Host 'Uninstall:'
Write-Host '  Unregister-ScheduledTask -TaskName code-rag-eval-gate  -Confirm:$false'
Write-Host '  Unregister-ScheduledTask -TaskName code-rag-eval-drift -Confirm:$false'

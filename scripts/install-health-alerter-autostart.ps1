# Phase 37-J: register a periodic dashboard health probe + alerter.
#
# What this installs:
#   Task name: code-rag-health-alert
#   Trigger:   Every 5 minutes, starting at logon
#   Action:    pythonw.exe -m code_rag health-alert --quiet
#              -- probes /api/health, writes data/health-state.json,
#                 appends data/alerts.jsonl on transitions, fires a
#                 Windows toast on degradation if BurntToast is installed.
#   Window:    none (pythonw.exe has no console)
#
# Why this exists:
#   The dashboard /api/health endpoint is on-demand — it only computes
#   state when a client hits it. To get continuous visibility (and
#   toast notifications when something breaks), we poll on a schedule
#   and persist transitions.
#
# Toast notifications (optional):
#   Install BurntToast once with:
#       Install-Module -Name BurntToast -Scope CurrentUser
#   Without it, the alerter still writes data/health-state.json and
#   data/alerts.jsonl — toasts just no-op.
#
# Run from a normal PowerShell (no admin needed):
#   .\scripts\install-health-alerter-autostart.ps1
#
# Uninstall:
#   Unregister-ScheduledTask -TaskName 'code-rag-health-alert' -Confirm:$false

$ErrorActionPreference = 'Stop'

$TaskName    = 'code-rag-health-alert'
$RepoRoot    = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$VenvPythonW = Join-Path $RepoRoot '.venv\Scripts\pythonw.exe'

if (-not (Test-Path $VenvPythonW)) {
    Write-Error "pythonw.exe not found at $VenvPythonW. Set up the venv first."
    exit 1
}

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing task '$TaskName'."
}

$action = New-ScheduledTaskAction `
    -Execute $VenvPythonW `
    -Argument '-m code_rag health-alert --once --quiet' `
    -WorkingDirectory $RepoRoot

# Run every 5 min, starting at logon. 5 min is the right cadence: tight
# enough to catch a degraded MCP within minutes, loose enough that the
# probe overhead is negligible.
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$trigger.Repetition = (New-ScheduledTaskTrigger `
    -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 5) `
    -RepetitionDuration (New-TimeSpan -Days 9999)).Repetition

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 1) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "code-rag dashboard health alerter. Polls /api/health every 5 min, writes data/health-state.json, alerts on transitions." |
  Out-Null

Write-Host ("Installed task '" + $TaskName + "' - runs every 5 min.")
Write-Host ("Manual run:  " + $VenvPythonW + " -m code_rag health-alert --once")
Write-Host ('Optional toast support: Install-Module -Name BurntToast -Scope CurrentUser')
Write-Host ('Uninstall:   Unregister-ScheduledTask -TaskName ' + $TaskName + ' -Confirm:$false')

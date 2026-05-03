# Phase 37-L: register a daily auto-redeploy after pull.
#
# What this installs:
#   Task name: code-rag-redeploy
#   Trigger:   Daily at 03:00 (after eval-gate at 02:30 + drift at 02:45 so
#              a regression flagged today doesn't auto-deploy tomorrow's
#              push without the operator seeing the alert overnight).
#   Action:    pythonw.exe -m code_rag redeploy --pull --quiet
#              -- runs `git pull --ff-only`, then if HEAD changed, stops
#                 + restarts code-rag-watch and code-rag-dashboard so they
#                 load the new source. Idempotent: a no-change pull is a
#                 zero-cost no-op.
#   Window:    none (pythonw.exe has no console)
#
# Why this exists:
#   Without this, a `git push` from another machine doesn't deploy until
#   you remember to manually kick the watcher. Daily auto-redeploy means
#   every push deploys within 24 hours with zero manual touch.
#
# Safety notes:
#   - Uses --ff-only, so a diverged local branch fails loudly instead of
#     silently merging. Operator review required to recover.
#   - Eval-gate (Phase 37-I) runs daily at 02:30 — quality regressions
#     are detected BEFORE the next daily redeploy at 03:00. So a bad
#     deploy from yesterday is caught by drift detection at 02:45 today
#     before redeploy fires at 03:00.
#   - MCP servers are NOT touched (Claude Code owns those; killing them
#     would drop live sessions). They naturally pick up new code on next
#     respawn (typically when Claude Code restarts).
#
# Run from a normal PowerShell (no admin needed):
#   .\scripts\install-redeploy-autostart.ps1
#
# Uninstall:
#   Unregister-ScheduledTask -TaskName 'code-rag-redeploy' -Confirm:$false

$ErrorActionPreference = 'Stop'

$TaskName    = 'code-rag-redeploy'
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
    -Argument '-m code_rag redeploy --pull --quiet' `
    -WorkingDirectory $RepoRoot

# Daily at 03:00 — after eval-gate (02:30) + drift check (02:45). If yesterday's
# deploy regressed quality, drift fires its alert at 02:45 BEFORE redeploy at
# 03:00, so the operator has a chance to revert before another deploy lands.
$trigger = New-ScheduledTaskTrigger -Daily -At (Get-Date '03:00')

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "code-rag daily auto-redeploy. git pull + restart watcher/dashboard if HEAD changed." |
  Out-Null

Write-Host ("Installed task '" + $TaskName + "' (runs daily at 03:00).")
Write-Host 'Manual run:'
Write-Host ('  ' + $VenvPythonW + ' -m code_rag redeploy --pull')
Write-Host 'Dry-run (no kills):'
Write-Host ('  ' + $VenvPythonW + ' -m code_rag redeploy --pull --dry-run')
Write-Host 'Uninstall:'
Write-Host ('  Unregister-ScheduledTask -TaskName ' + $TaskName + ' -Confirm:$false')

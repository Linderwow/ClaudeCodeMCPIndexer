# Phase 32: register a periodic process-hygiene reaper.
#
# What this installs:
#   Task name: code-rag-reap
#   Trigger:   Every 10 minutes, starting at logon
#   Action:    pythonw.exe -m code_rag reap --kill --quiet
#              -- finds and kills orphaned MCP / dashboard / watcher
#                 processes (those whose legitimate parent is dead).
#   Window:    none (pythonw.exe has no console)
#   Duration:  runs forever (every 10 min)
#
# Why this exists:
#   When Claude Code crashes or restarts ungracefully, the `code-rag mcp`
#   subprocess it spawned can outlive its parent. Without cleanup, these
#   orphans accumulate over days/weeks and can each consume 600+ MB of RAM.
#   `code-rag reap` walks the live process tree, identifies orphans, and
#   terminates them. Self-bounding (only kills processes whose ancestor is
#   gone — never touches active sessions).
#
# Run from a normal PowerShell (no admin needed):
#   .\scripts\install-reaper-autostart.ps1
#
# Uninstall:
#   Unregister-ScheduledTask -TaskName 'code-rag-reap' -Confirm:$false

$ErrorActionPreference = 'Stop'

$TaskName    = 'code-rag-reap'
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
    -Argument '-m code_rag reap --kill --quiet' `
    -WorkingDirectory $RepoRoot

# Run every 10 min, indefinitely, starting at logon (so first run is soon
# after boot rather than waiting for the first 10-min interval).
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$trigger.Repetition = (New-ScheduledTaskTrigger `
    -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes 10) `
    -RepetitionDuration (New-TimeSpan -Days 9999)).Repetition

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 2) `
    -MultipleInstances IgnoreNew

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "code-rag periodic orphan reaper. Kills MCP/dashboard/watcher subprocesses whose ancestor is gone." |
  Out-Null

Write-Host ("Installed task '" + $TaskName + "' - runs every 10 min.")
Write-Host ("Manual run:   Start-ScheduledTask -TaskName " + $TaskName)
Write-Host ("Dry-run any time: " + $VenvPythonW + " -m code_rag reap")
Write-Host ('Uninstall:    Unregister-ScheduledTask -TaskName ' + $TaskName + ' -Confirm:$false')

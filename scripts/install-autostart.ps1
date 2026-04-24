# Register code-rag's full boot chain as a windowless Windows scheduled task.
#
# What this installs:
#   Task name: code-rag-watch
#   Trigger:   At log on (current user)
#   Action:    pythonw.exe -m code_rag.autostart_bootstrap
#              -- starts LM Studio, loads the embedder, then runs the watcher
#   Window:    none (pythonw.exe has no console)
#   Restart:   5 attempts, 1 min apart, on any failure
#   Duration:  runs indefinitely
#
# Why Task Scheduler and not the Startup folder?
#   - pythonw.exe via Task Scheduler produces zero window flashes.
#   - Task Scheduler auto-restarts on crash; the Startup folder won't.
#   - UAC is handled cleanly: task runs with your Interactive principal.
#
# Run from an ELEVATED PowerShell (admin) once:
#   .\scripts\install-autostart.ps1
#
# Uninstall:
#   .\scripts\uninstall-autostart.ps1

$ErrorActionPreference = 'Stop'

$TaskName = 'code-rag-watch'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$VenvPythonW = Join-Path $RepoRoot '.venv\Scripts\pythonw.exe'

if (-not (Test-Path $VenvPythonW)) {
    Write-Error "pythonw.exe not found at $VenvPythonW. Run 'python -m venv .venv; pip install -e .' first."
    exit 1
}

# Remove existing task (idempotent reinstall).
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing task '$TaskName'."
}

$action   = New-ScheduledTaskAction `
    -Execute $VenvPythonW `
    -Argument '-m code_rag.autostart_bootstrap' `
    -WorkingDirectory $RepoRoot

$trigger  = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# Settings: keep it alive, auto-restart on failure, don't stop when on battery.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 5 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero)   # run indefinitely

# Run with the user's Interactive privileges (no full admin required).
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "code-rag boot chain. Starts LM Studio, loads the embedder, then runs the live watcher." |
  Out-Null

Write-Host "Installed task '$TaskName'. It will start at next logon."
Write-Host "To start it now:   Start-ScheduledTask -TaskName $TaskName"
Write-Host "To see status:     Get-ScheduledTaskInfo -TaskName $TaskName"
Write-Host "Log file:          $RepoRoot\logs\autostart.log"
Write-Host "To uninstall:      .\scripts\uninstall-autostart.ps1"

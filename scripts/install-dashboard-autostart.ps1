# Register the code-rag dashboard as a windowless Windows scheduled task.
#
# What this installs:
#   Task name: code-rag-dashboard
#   Trigger:   At log on (current user)
#   Action:    pythonw.exe -m code_rag dashboard --no-browser --port 7321
#   Window:    none (pythonw.exe has no console)
#   Restart:   5 attempts, 1 min apart, on any failure
#   Duration:  runs indefinitely
#
# After install, the dashboard is reachable at http://127.0.0.1:7321/ from
# every login until you explicitly stop it. Companion to install-autostart.ps1
# (which manages the indexer watcher) -- the two tasks are independent;
# stopping or uninstalling one doesn't affect the other.
#
# Run from an ELEVATED PowerShell once:
#   .\scripts\install-dashboard-autostart.ps1
#
# Uninstall:
#   .\scripts\uninstall-dashboard-autostart.ps1

$ErrorActionPreference = 'Stop'

$TaskName = 'code-rag-dashboard'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$VenvPythonW = Join-Path $RepoRoot '.venv\Scripts\pythonw.exe'

if (-not (Test-Path $VenvPythonW)) {
    Write-Error "pythonw.exe not found at $VenvPythonW. Run 'python -m venv .venv; pip install -e .' first."
    exit 1
}

# Remove any prior registration so reinstalls are idempotent.
if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed existing task '$TaskName'."
}

$action   = New-ScheduledTaskAction `
    -Execute $VenvPythonW `
    -Argument '-m code_rag dashboard --no-browser --port 7321' `
    -WorkingDirectory $RepoRoot

$trigger  = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

# Settings: keep alive; auto-restart on crash; don't stop on battery.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 5 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit ([TimeSpan]::Zero)   # run indefinitely

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "code-rag dashboard. Serves the local control UI at http://127.0.0.1:7321/." |
  Out-Null

Write-Host "Installed task '$TaskName'. It will start at next logon."
Write-Host "To start it now:   Start-ScheduledTask -TaskName $TaskName"
Write-Host "URL:               http://127.0.0.1:7321/"
Write-Host "To uninstall:      .\scripts\uninstall-dashboard-autostart.ps1"

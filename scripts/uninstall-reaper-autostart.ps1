# Uninstall Phase 32 reaper. Reverses install-reaper-autostart.ps1.

$ErrorActionPreference = 'Stop'
$TaskName = 'code-rag-reap'

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed scheduled task '$TaskName'."
} else {
    Write-Host "Task '$TaskName' was not registered."
}

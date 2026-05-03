# Uninstall Phase 37-J health alerter.
# Reverses install-health-alerter-autostart.ps1.

$ErrorActionPreference = 'Stop'
$TaskName = 'code-rag-health-alert'

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed scheduled task '$TaskName'."
} else {
    Write-Host "Task '$TaskName' was not registered."
}

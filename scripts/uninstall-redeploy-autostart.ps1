# Uninstall Phase 37-L daily redeploy.
# Reverses install-redeploy-autostart.ps1.

$ErrorActionPreference = 'Stop'
$TaskName = 'code-rag-redeploy'

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed scheduled task '$TaskName'."
} else {
    Write-Host "Task '$TaskName' was not registered."
}

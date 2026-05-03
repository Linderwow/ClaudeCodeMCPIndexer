# Uninstall the Phase 36 self-healing watchdog suite.
# Reverses install-watchdog-autostart.ps1 (chroma-heal + lms-enforce + chroma-defrag).

$ErrorActionPreference = 'Stop'
$Tasks = @('code-rag-chroma-heal', 'code-rag-lms-enforce', 'code-rag-chroma-defrag')

foreach ($n in $Tasks) {
    if (Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $n -Confirm:$false
        Write-Host "Removed scheduled task '$n'."
    } else {
        Write-Host "Task '$n' was not registered."
    }
}

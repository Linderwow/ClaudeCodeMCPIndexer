# Uninstall Phase 37-I eval cron.
# Reverses install-eval-cron.ps1 (eval-gate + eval-drift).

$ErrorActionPreference = 'Stop'
$Tasks = @('code-rag-eval-gate', 'code-rag-eval-drift')

foreach ($n in $Tasks) {
    if (Get-ScheduledTask -TaskName $n -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $n -Confirm:$false
        Write-Host "Removed scheduled task '$n'."
    } else {
        Write-Host "Task '$n' was not registered."
    }
}

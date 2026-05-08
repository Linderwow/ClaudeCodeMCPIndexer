# Quick dashboard restart — kills duplicates, relaunches via Task Scheduler.
$dashes = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*code_rag dashboard*' }
foreach ($d in $dashes) {
    try { Stop-Process -Id $d.ProcessId -Force; Write-Host "killed $($d.ProcessId)" } catch { }
}
Start-Sleep -Seconds 4
Start-ScheduledTask -TaskName 'code-rag-dashboard'
Start-Sleep -Seconds 6
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*code_rag dashboard*' } |
    Select-Object ProcessId, CreationDate | Format-Table -AutoSize

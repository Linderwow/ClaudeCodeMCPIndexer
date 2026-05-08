# Hide every user-space scheduled task that isn't already hidden.
# Skips Microsoft system tasks, ASUS / Adobe / OneDrive / NVIDIA / Zoom etc.
# (those have to manage their own UX).

$tasks = Get-ScheduledTask | Where-Object {
    (-not $_.Settings.Hidden) -and
    ($_.State -ne 'Disabled') -and
    ($_.TaskPath -notlike '\Microsoft\*') -and
    ($_.TaskPath -notlike '\ASUS\*') -and
    ($_.TaskName -notmatch '^(Adobe|OneDrive|NVIDIA|nWizard|Zoom|PowerENGAGE|SamsungMagician|ETW|ThePremise_watch)')
}

$ok = 0
$fail = @()
foreach ($t in $tasks) {
    try {
        $ns = $t.Settings
        $ns.Hidden = $true
        Set-ScheduledTask -TaskName $t.TaskName -TaskPath $t.TaskPath -Settings $ns | Out-Null
        Write-Host "OK   $($t.TaskName)"
        $ok++
    } catch {
        Write-Host "FAIL $($t.TaskName)"
        $fail += $t.TaskName
    }
}

Write-Host ""
Write-Host "summary: $ok fixed, $($fail.Count) failed"
if ($fail.Count -gt 0) { Write-Host ("  failures: " + ($fail -join ', ')) }

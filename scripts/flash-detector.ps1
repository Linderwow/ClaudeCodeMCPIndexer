# Flash detector — logs every NEW visible top-level window appearing on the
# desktop, with timestamp + window title + owning process. Run this in a
# separate PowerShell while you're gaming. Next time something pops focus
# from your foreground app, you'll see EXACTLY what created the window.
#
# Polls every 250 ms for new top-level windows via Win32 EnumWindows. Compares
# against the previous tick's set; logs any window that appeared this tick
# AND is currently visible (IsWindowVisible). Filters out our own polling
# overhead and the desktop / taskbar / explorer chrome.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File flash-detector.ps1
#
# Stop with Ctrl-C. Log written to %TEMP%\flash-detector.log.

$LogPath = Join-Path $env:TEMP 'flash-detector.log'
"flash-detector started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss.fff')" | Out-File $LogPath -Encoding utf8
"polling every 250 ms; logs to $LogPath" | Tee-Object -FilePath $LogPath -Append

Add-Type @'
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

public static class WinEnum {
    public delegate bool EnumWindowsProc(IntPtr hWnd, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern bool EnumWindows(EnumWindowsProc lpEnumFunc, IntPtr lParam);

    [DllImport("user32.dll")]
    public static extern bool IsWindowVisible(IntPtr hWnd);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int GetWindowTextLength(IntPtr hWnd);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int GetWindowText(IntPtr hWnd, StringBuilder lpString, int nMaxCount);

    [DllImport("user32.dll", SetLastError=true)]
    public static extern uint GetWindowThreadProcessId(IntPtr hWnd, out uint lpdwProcessId);

    public static List<Tuple<IntPtr, string, uint>> Snapshot() {
        var list = new List<Tuple<IntPtr, string, uint>>();
        EnumWindows((hWnd, lParam) => {
            if (!IsWindowVisible(hWnd)) return true;
            int len = GetWindowTextLength(hWnd);
            string title = "";
            if (len > 0) {
                var sb = new StringBuilder(len + 1);
                GetWindowText(hWnd, sb, sb.Capacity);
                title = sb.ToString();
            }
            uint procId;
            GetWindowThreadProcessId(hWnd, out procId);
            list.Add(Tuple.Create(hWnd, title, procId));
            return true;
        }, IntPtr.Zero);
        return list;
    }
}
'@

function Get-ProcInfo {
    param([uint32]$Pid)
    try {
        $p = Get-Process -Id $Pid -ErrorAction Stop
        $cim = Get-CimInstance Win32_Process -Filter "ProcessId=$Pid" -ErrorAction SilentlyContinue
        $cmd = if ($cim) { $cim.CommandLine } else { '' }
        if ($cmd) {
            $cmd = $cmd.Substring(0, [Math]::Min(180, $cmd.Length))
        }
        "$($p.ProcessName) [$Pid]  cmd=$cmd"
    } catch {
        "<gone> [$Pid]"
    }
}

# Initial snapshot — establish "already open" windows so we don't log them.
$prev = @{}
foreach ($w in [WinEnum]::Snapshot()) {
    $prev[$w.Item1] = $true
}
"baseline: $($prev.Count) visible windows already open" | Tee-Object -FilePath $LogPath -Append

# Poll loop.
while ($true) {
    Start-Sleep -Milliseconds 250
    $cur = [WinEnum]::Snapshot()
    foreach ($w in $cur) {
        if (-not $prev.ContainsKey($w.Item1)) {
            $title = if ($w.Item2) { $w.Item2 } else { '<no title>' }
            $info = Get-ProcInfo -Pid $w.Item3
            $ts = Get-Date -Format 'HH:mm:ss.fff'
            $line = "[$ts] FLASH  hwnd=0x$([Convert]::ToString($w.Item1.ToInt64(), 16))  title='$title'  src: $info"
            Write-Host $line -ForegroundColor Yellow
            $line | Out-File $LogPath -Append -Encoding utf8
        }
    }
    # Refresh baseline so a window that closes + reopens is logged.
    $prev = @{}
    foreach ($w in $cur) { $prev[$w.Item1] = $true }
}

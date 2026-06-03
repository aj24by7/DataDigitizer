$ErrorActionPreference = "Stop"

$sourceDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$targetDir = Join-Path $env:LOCALAPPDATA "DataDigitizer\2.11"
$desktopDir = [Environment]::GetFolderPath("Desktop")
$programsDir = [Environment]::GetFolderPath("Programs")
$startMenuDir = Join-Path $programsDir "Data Digitizer"

$apps = @(
    @{ File = "digitizer.exe"; Shortcut = "Digitizer.lnk"; Description = "Data Digitizer" },
    @{ File = "accuracytester.exe"; Shortcut = "AccuracyTester.lnk"; Description = "Accuracy Tester" },
    @{ File = "DataDigitizer-2.11.exe"; Shortcut = "DataDigitizer CLI.lnk"; Description = "Data Digitizer CLI" }
)

New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
New-Item -ItemType Directory -Force -Path $startMenuDir | Out-Null

foreach ($app in $apps) {
    $source = Join-Path $sourceDir $app.File
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Missing $($app.File). Keep this installer in the same folder as the exe files."
    }
    Copy-Item -LiteralPath $source -Destination (Join-Path $targetDir $app.File) -Force
}

$shell = New-Object -ComObject WScript.Shell

function New-Shortcut {
    param(
        [string]$ShortcutPath,
        [string]$TargetPath,
        [string]$Description
    )

    $shortcut = $shell.CreateShortcut($ShortcutPath)
    $shortcut.TargetPath = $TargetPath
    $shortcut.WorkingDirectory = Split-Path -Parent $TargetPath
    $shortcut.IconLocation = $TargetPath
    $shortcut.Description = $Description
    $shortcut.Save()
}

foreach ($app in $apps) {
    $target = Join-Path $targetDir $app.File
    New-Shortcut -ShortcutPath (Join-Path $desktopDir $app.Shortcut) -TargetPath $target -Description $app.Description
    New-Shortcut -ShortcutPath (Join-Path $startMenuDir $app.Shortcut) -TargetPath $target -Description $app.Description
}

Write-Host "Data Digitizer 2.11 installed successfully."
Write-Host ""
Write-Host "Desktop shortcuts created:"
Write-Host "  Digitizer"
Write-Host "  AccuracyTester"
Write-Host "  DataDigitizer CLI"
Write-Host ""
Write-Host "Installed files:"
Write-Host "  $targetDir"
Write-Host ""
Write-Host "To pin Digitizer to the taskbar:"
Write-Host "  1. Double-click the Digitizer desktop shortcut."
Write-Host "  2. Right-click the Digitizer icon on the taskbar."
Write-Host "  3. Click Pin to taskbar."

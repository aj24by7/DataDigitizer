param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (-not $SkipInstall) {
    py -m pip install -r requirements.txt
}

# Single one-click executable: Digitizer.exe
# The Tesseract OCR runtime in vendor\tesseract is bundled into the exe, so OCR
# works on machines without Tesseract installed.
py -m PyInstaller --clean --noconfirm digitizer.spec

Write-Host ""
Write-Host "Built: $ScriptDir\dist\Digitizer.exe"

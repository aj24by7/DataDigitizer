param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (-not $SkipInstall) {
    py -m pip install -r requirements.txt
}

py -m PyInstaller --clean --noconfirm accuracytester.spec

Write-Host ""
Write-Host "Built: $ScriptDir\dist\AccuracyTester.exe"

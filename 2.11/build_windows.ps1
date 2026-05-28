param(
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

if (-not $SkipInstall) {
    py -m pip install -r requirements-2.11.txt
}

py -m PyInstaller --clean --noconfirm DataDigitizer-2.11.spec
py -m PyInstaller --clean --noconfirm digitizer.spec
py -m PyInstaller --clean --noconfirm accuracytester.spec

Write-Host ""
Write-Host "Built: $ScriptDir\dist\DataDigitizer-2.11.exe"
Write-Host "Built: $ScriptDir\dist\digitizer.exe"
Write-Host "Built: $ScriptDir\dist\accuracytester.exe"
Write-Host "Note: OCR requires the Tesseract engine. If vendor\tesseract contains a full Tesseract runtime, it is bundled into the one-file exe."

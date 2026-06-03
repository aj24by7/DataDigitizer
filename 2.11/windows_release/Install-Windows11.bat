@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0DataDigitizer-2.11-Windows11-Installer.exe" (
    start "" "%~dp0DataDigitizer-2.11-Windows11-Installer.exe"
    exit /b 0
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-Windows11.ps1"
echo.
pause

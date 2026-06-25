@echo off
REM ===========================================================================
REM  Data Digitizer - one-click Windows build  (code  ->  Digitizer.exe)
REM
REM  Double-click this file, or run it in a terminal:   build.cmd
REM  It installs the Python dependencies and runs PyInstaller via digitizer.spec,
REM  bundling the Tesseract OCR runtime from vendor\tesseract into the .exe.
REM
REM  Output:  dist\Digitizer.exe
REM  Skip the dependency install with:   build.cmd -SkipInstall
REM
REM  Requires Python 3 on PATH (the "py" launcher). Get it from https://python.org
REM ===========================================================================
setlocal
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_windows.ps1" %*
echo.
echo Done. If it succeeded, your app is at:  %~dp0dist\Digitizer.exe
echo.
pause

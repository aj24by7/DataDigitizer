# Accuracy Tester

A standalone desktop tool that compares a digitized CSV against a reference
("original") CSV and reports accuracy metrics. Independent of the Digitizer app.

## Run from source

```powershell
py -m pip install -r requirements.txt
py accuracytester_desktop.py
```

Load two CSV files (original vs digitized). Drag-and-drop works when `tkinterdnd2`
is installed; otherwise click the panels to browse for files.

## Build the Windows executable

```powershell
.\build_windows.ps1
```

Output: `dist\AccuracyTester.exe` — a windowed, one-file app (double-click to open).

## Folder contents

```text
AccuracyTesterPro.py     # The tool (Tkinter + matplotlib + pandas)
accuracytester_desktop.py# Entry point
accuracytester.spec      # PyInstaller spec (AccuracyTester.exe)
build_windows.ps1        # Build script
requirements.txt         # Python dependencies
version.json             # App name + version
assets/                  # Application icon
```

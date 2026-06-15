# Digitizer 2.12

Converts graph images into numeric data points and exports CSV/Excel files plus an
overlay image. This folder is fully self-contained — run it from source or build the
single `Digitizer.exe`.

## Run from source

GUI:

```powershell
py digitizer_2_11.py
```

CLI (function-call syntax):

```powershell
py digitizer.py 'digitizer_cli(pic_dir="C:\path\to\plot.png", output_dir="C:\path\to\out")'
```

CLI with manual color / ticks / axis values:

```powershell
py digitizer.py 'digitizer_cli(pic_dir="C:\path\to\plot.png", color=(255,0,0), tick_setting=([10,200],[500,200],[10,200],[10,20]), axis_values=(0,10,0,100), output_dir="C:\path\to\out")'
```

CLI (flag syntax):

```powershell
py digitizer_2_11.py cli --pic-dir path\to\plot.png --color 255,0,0 --ticks "[10,200],[500,200],[10,200],[10,20]" --axis-values 0,10,0,100
```

CLI outputs `<image>_digitized_points.csv` and `<image>_digitized_overlay.png`.

## Build the Windows executable

```powershell
py -m pip install -r requirements.txt
.\build_windows.ps1
```

Output: `dist\Digitizer.exe` — a windowed, one-file app (double-click to open). The
Tesseract OCR runtime under `vendor\tesseract` is bundled into the exe, so OCR
features (axis-value detection, text/number/legend masking) work without a separate
Tesseract install.

## Folder contents

```text
digitizer.py        # CLI / function-call launcher
digitizer_2_11.py   # GUI + CLI dispatch and runtime path setup
digitizer_desktop.py# GUI-only entry used by the PyInstaller build
digitizer_cli.py    # Terminal CLI
digitizer_api.py    # Programmatic API (digitize_image)
function.py         # Thin re-export of digitize_image
UI.py               # Main window and GUI workflows
ImageTray.py        # Canvas rendering and interaction
PointPlacer.py      # Color-based point extraction + interpolation
AxisReader.py       # OCR axis detection
Calibration.py      # Calibration strategies
Masking.py          # OCR / manual masking helpers
ErrorLogger.py      # Error logging utilities
digitizer.spec      # PyInstaller spec (Digitizer.exe)
build_windows.ps1   # Build script
requirements.txt    # Python dependencies
version.json        # App name + version
assets/             # Application icon
vendor/tesseract/   # Bundled Tesseract OCR runtime
```

Runtime logs are written to `%LOCALAPPDATA%\DataDigitizer\2.12\logs` (viewable in-app
via `Advanced -> Error Log`).

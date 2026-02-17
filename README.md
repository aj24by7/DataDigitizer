# Data Digitizer 2.5

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![GUI](https://img.shields.io/badge/GUI-PyQt6-41CD52?logo=qt&logoColor=white)
![Export](https://img.shields.io/badge/Export-CSV%20%7C%20XLSX-1F6FEB)
![OCR](https://img.shields.io/badge/OCR-Optional-F39C12)

Extract numeric data from graph images with a GUI workflow that supports color-based tracing, masking, calibration, and export to CSV/Excel.

Tags: `graph-digitizer` `spectra` `data-extraction` `ocr` `pyqt6` `calibration` `csv` `xlsx`

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Key Features](#key-features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Quick Start (2 Minutes)](#quick-start-2-minutes)
6. [Full Tutorial](#full-tutorial)
7. [Advanced Features](#advanced-features)
8. [Export Format](#export-format)
9. [Troubleshooting](#troubleshooting)
10. [Project Structure](#project-structure)
11. [Contributing](#contributing)
12. [License](#license)

## What This Project Does

Data Digitizer converts plotted curves in an image into numeric `(x, y)` points.

Typical use cases:

- Raman/IR/UV-vis spectra from papers
- Line charts from reports or slides
- Legacy plots where raw data is unavailable

## Key Features

- Import image from file or clipboard
- Auto color suggestion plus manual color picking
- Point extraction by RGB tolerance
- Optional interpolation for denser sampling
- Optional chroma filter to suppress gray/low-color noise
- OCR-assisted axis value detection (optional dependency)
- Three calibration modes:
  - Manual calibration (click 4 reference points)
  - Coordinate-mediated calibration (uses OCR axis detections)
  - Line-mediated calibration (uses black border detection)
- Masking workflow:
  - Mask words (OCR)
  - Mask numbers (OCR)
  - Mask legend (OCR cluster)
  - Manual mask draw/remove
- Export:
  - CSV: raw or Y-normalized
  - Excel (.xlsx): raw or Y-normalized
- Error logging UI and persistent logs

## Requirements

- Python `3.9+`
- OS: Windows/macOS/Linux

Core runtime:

- `PyQt6`

Recommended:

- `numpy` (faster point extraction/calibration path)

Optional for advanced features:

- `pillow` + `pytesseract` (OCR axis detection and OCR masking)
- Tesseract OCR engine installed and available on `PATH` (or placed in `vendor/tesseract/tesseract.exe`)
- `openpyxl` (Excel export)

## Installation

### 1) Clone and enter repository

```bash
git clone https://github.com/aj24by7/DataDigitizer.git
cd DataDigitizer
```

### 2) Create and activate virtual environment

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

Minimal:

```bash
pip install PyQt6
```

Recommended (includes advanced features):

```bash
pip install PyQt6 numpy pillow pytesseract openpyxl
```

### 4) Optional OCR verification

```bash
tesseract --version
```

If this fails, install Tesseract OCR and ensure it is on your system `PATH`.

## Quick Start (2 Minutes)

```bash
cd 2.5
python 2.5.py
```

When prompted:

```text
Run program? [y/n]: y
```

Then in the GUI:

1. `Import -> Import Image`
2. `Tools -> Pick Color`, click your curve color
3. `Tools -> Place Points -> Run`
4. Fill axis min/max fields
5. Run calibration
6. `Export -> Export CSV -> Raw values`

## Full Tutorial

### Step 1: Load an image

- Use `Import -> Import Image` for PNG/JPG/JPEG/BMP/TIF/TIFF.
- Or use `Import -> Paste Image` to pull from clipboard.

### Step 2: Select the target curve color

- Use `Tools -> Pick Color`.
- Click on the graph line.
- A color swatch confirms selected RGB.

Tip:

- Auto color selection runs at image load. If points are wrong, manually repick.

### Step 3: Remove unwanted regions (recommended)

Use `Advanced -> Masking`:

- `Mask Words` for labels/titles
- `Mask Numbers` for tick values/text
- `Mask Legend` for legend block
- `Manual Mask Mode` for custom redaction

Manual mask controls:

- Drag to create a mask rectangle
- `Shift+Click` inside a mask to remove it
- `Esc` to exit manual mask mode

### Step 4: Detect axis scale (optional but useful)

Run: `Advanced -> Axis Scale Detection -> Run Axis Detection`

Outcome:

- Detected values are copied to X/Y min-max fields
- OCR overlays are shown on the image

If OCR is partial:

- Fill missing axis values manually in the min-max fields

### Step 5: Calibrate pixel-to-data mapping

Use `Tools -> Calibration` and choose one mode:

1. Manual Calibration
- Click points in this exact order:
  - X min
  - X max
  - Y min
  - Y max
- Best when OCR is unavailable or axis text is noisy

2. Coordinate-Mediated Calibration
- Uses OCR-detected axis points
- Requires axis detection first

3. Line-Mediated Calibration
- Detects a black graph border automatically
- Best for boxed plots with clear dark frame

### Step 6: Place points

Run: `Tools -> Place Points -> Run`

Optional refinements:

- `Tools -> Interpolation` to densify points
- `Advanced -> Chroma Filter` to suppress low-color background noise
- `Tools -> Place Points -> Limit to Calibration Window` to keep points inside calibrated area

### Step 7: Export

Use `Export` menu:

- CSV:
  - `Raw values`
  - `Y normalized (0-1)`
- Excel:
  - `Raw values`
  - `Y normalized (0-1)` (requires `openpyxl`)

## Advanced Features

### OCR-Assisted Workflow

If `pytesseract` + Tesseract are available:

- Axis values can be auto-detected from the chart
- Text/number/legend masks become one-click operations

Without OCR:

- App still works with manual axis entry + manual calibration + manual masks

### Affine Mapping (for non-perfect captures)

During export, the app can use an affine mapper from calibration points. This is more robust than pure rectangular scaling when the image has mild perspective/rotation effects.

### Error Logging and Diagnostics

- In-app log viewer: `Advanced -> Error Log`
- Persistent logs:
  - `2.5/logs/error_log.txt`
  - `2.5/logs/error_log.csv`

Use these logs when reporting issues.

## Export Format

Raw export columns:

- `x`
- `y`
- `x_px` (pixel x)
- `y_px` (pixel y)

Normalized export adds:

- `y_norm` in `[0, 1]`

## Troubleshooting

`Axis detection requires pytesseract + pillow installed.`

- Install `pillow` and `pytesseract`
- Install Tesseract engine and verify `tesseract --version`

`Export requires X/Y min/max values.`

- Fill min-max boxes manually or run axis detection

`Export requires calibration. Run calibration first.`

- Run manual, coordinate-mediated, or line-mediated calibration

`Line-mediated calibration failed to find a black border.`

- Use manual calibration or coordinate-mediated calibration
- Ensure border is visible and dark

`No points available to export.`

- Pick color again
- Disable/adjust masks
- Adjust workflow: run place points after calibration

Accuracy tips:

- Use high-resolution images
- Prefer high-contrast curve colors
- Mask legend/text before extracting points
- Recheck axis min/max and calibration order

## Project Structure

```text
2.5/
  2.5.py            # App entry point (PyQt6)
  UI.py             # Main window and workflows
  ImageTray.py      # Canvas rendering and interaction
  PointPlacer.py    # Color-based point extraction + interpolation
  AxisReader.py     # OCR axis detection
  Calibration.py    # Calibration strategies
  Masking.py        # OCR/manual masking helpers
  ErrorLogger.py    # Error logging utilities
  logs/             # Runtime log output
  main.py           # Legacy tkinter implementation
```

## Contributing

Issues and PRs are welcome.

Suggested contribution style:

- Include before/after screenshots for UI changes
- Include reproducible sample image for bug reports
- Include dependency notes for OCR/export related changes

## License

No `LICENSE` file is currently present in this repository. Add one if you plan to distribute or accept external contributions under defined terms.


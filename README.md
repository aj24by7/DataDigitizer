# Data Digitizer

Two Windows desktop tools:

- **Digitizer** — converts graph images into numeric data points and exports CSV/Excel
  (color tracing, masking, manual & coordinate calibration, OCR axis detection).
- **AccuracyTester** — compares a digitized CSV against a reference CSV and reports
  accuracy metrics.

## Download (Windows, one click)

Grab the ready-to-run executables from the latest release — no install, no Python,
just double-click:

**https://github.com/aj24by7/DataDigitizer/releases/latest**

- `Digitizer.exe` — OCR is bundled, so no separate Tesseract install is needed.
- `AccuracyTester.exe`

If Windows shows **"Windows protected your PC"**, click **More info → Run anyway**
(the executables are not code-signed with a paid certificate).

## Build from source

Each tool is self-contained in its own folder:

```powershell
cd Digitizer        # or: cd AccuracyTester
py -m pip install -r requirements.txt
.\build_windows.ps1
```

See [`Digitizer/README.md`](Digitizer/README.md) and
[`AccuracyTester/README.md`](AccuracyTester/README.md) for details.

To bundle the Tesseract OCR runtime into `Digitizer.exe`, place a Tesseract install
under `Digitizer/vendor/tesseract/` before building. That folder is gitignored (it is
large) and not committed; the released `Digitizer.exe` already has it baked in.

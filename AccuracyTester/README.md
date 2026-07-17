# Accuracy Tester

Compares a digitized CSV against a reference ("original") CSV and reports accuracy
metrics. Independent of the Digitizer app — you don't need it to get your data out.

Two ways to use it: the **desktop app** (two files at a time, with diagnostic plots) and
the **batch CLI** (a whole folder, scored by file name).

## Run from source

```powershell
py -m pip install -r requirements.txt
py accuracytester_desktop.py
```

Load two CSV files (original vs digitized). Drag-and-drop works when `tkinterdnd2`
is installed; otherwise click the panels to browse for files.

## Batch CLI — score a whole folder

```powershell
py accuracy_batch_cli.py --digitized-dir "path\to\digitized_csvs" --ground-truth-dir "path\to\truth"
```

Writes, next to the digitized CSVs:

- **`accuracy_report\accuracy_report.csv`** — one row per image: RMSE, correlation, MSE,
  R2, MAE, bias, MAPE/sMAPE/WAPE, `nrmse_range_pct`, point count, plus a `status`.
- **`accuracy_report\accuracy_summary.json`** — totals and the overall failure rate.

Pairs are matched by file-name stem after stripping a `_digitized_points` / `_digitized` /
`_points` / `_ground_truth` / `_gt` suffix. Exact matches take priority, and a tie between
several candidates is reported as `ambiguous_match` — never guessed. Every unmatched or
ignored file is reported rather than silently skipped.

| Status | Means |
| --- | --- |
| `ok` | Pair scored; metrics present. |
| `no_method_match` | A reference with no digitized file to compare against. |
| `no_ground_truth_match` | A digitized file whose name matches no reference. |
| `ambiguous_match` | Several files matched one reference equally well. |
| `invalid_data` | Can't be scored as y=f(x) — e.g. a vertical line. |
| `insufficient_overlap` | The curves share too little X range to compare. |
| `parse_error` | A CSV could not be read at all. |

## Reading the numbers honestly

- **Quote `nrmse_range_pct`, not raw RMSE.** RMSE carries the units of the Y axis, so it is
  meaningless across figures with different scales. An RMSE of 63.9 on an axis running to
  30,000 is *better* than 0.03 on an axis running 0–1.
- **Correlation and R2 are undefined against a flat reference.** Both divide by the variance
  of the reference, which is zero for a horizontal line, so the CSV prints a degenerate
  value like `-2.6e+25`. Read that as **N/A**, not as an accuracy.
- **A vertical line is not a function of x**, so `invalid_data` is the correct answer rather
  than a number. This is by design.

## Tests

The tests check the scoring maths against scipy and scikit-learn as independent
references, so they need a few packages the app itself does not:

```powershell
py -m pip install -r requirements-dev.txt
py -m pytest test_accuracy_core.py test_batch_scoring.py -q
```

## Build the Windows executable

```powershell
.\build_windows.ps1
```

Output: `dist\AccuracyTester.exe` — a windowed, one-file app (double-click to open).

## Folder contents

```text
AccuracyTesterPro.py     # The desktop tool (Tkinter + matplotlib + pandas)
accuracytester_desktop.py# GUI entry point
accuracy_core.py         # The scoring maths (alignment, metrics)
batch_scoring.py         # File matching + pair scoring for batch runs
accuracy_batch_cli.py    # Batch CLI entry point
test_accuracy_core.py    # Tests for the scoring maths
test_batch_scoring.py    # Tests for file matching + pair scoring
requirements-dev.txt     # Test-only deps (pytest, scipy, scikit-learn)
accuracytester.spec      # PyInstaller spec (AccuracyTester.exe)
build_windows.ps1        # Build script
requirements.txt         # Python dependencies
version.json             # App name + version
assets/                  # Application icon
```

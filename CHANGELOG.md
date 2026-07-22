# Changelog

All notable changes to this project are documented here. Versions follow the release tags
on GitHub; download the apps from the [Releases page](https://github.com/aj24by7/DataDigitizer/releases/latest).

## Unreleased

### Changed

- **Exported points are no longer limited to the calibration box by default.** The box is
  built from the tick positions OCR found, so an outermost tick read even slightly short
  makes the box smaller than the real plot area -- and limiting to it silently clips real
  data off the ends of the curve. On `example_3_corundum_raman.png` it kept **81 of 930
  points**, discarding 91% of the spectrum with no error.

  The GUI already defaulted to off, but the CLI, the batch CLI and the `digitize_image()`
  API each defaulted to **on**, so the same image produced different data depending on how
  you ran it. All four now agree. `--limit-to-calibration` still turns it on deliberately,
  and `--no-limit-to-calibration` is kept as an explicit no-op so existing scripts and
  documented commands keep working unchanged. `test_defaults.py` locks all four entry
  points together.

### Fixed

- The exe is blocked silently by **Smart App Control** on Windows 11 (on by default for new
  installs) because it is unsigned -- double-clicking does nothing at all, with no dialog.
  The docs previously described only the SmartScreen *"Run anyway"* flow, which never
  appears under Smart App Control. Documented how to identify it and pointed at
  `Digitizer.pyw` as the working alternative. The real fix is code-signing.
- The "where to put Tesseract" instructions had their path corrupted to
  `Digitizer<VT>endor<TAB>esseract` by unescaped backslashes, making the only guidance for a
  required manual step unreadable.
- The build claimed it would fail without `vendor/tesseract`; it silently produced an exe
  with no OCR instead. It now warns loudly.
- The version was hardcoded in six files, so 2.14 reported itself as 2.13 everywhere
  user-visible, including the `%LOCALAPPDATA%\DataDigitizer\<version>\logs` path the README
  documents. `version.json` is now read at runtime and is the single source of truth.
- The documented test command could not run from the declared dependencies (`pytest`,
  `scipy`, `scikit-learn` were undeclared). Added `requirements-dev.txt`.

## 2.14

### Fixed

- **Calibration could silently skew every exported value.** Axis detection excludes each
  OCR text box when snapping a tick label to its axis line. An oversized or misplaced text
  box could cover the axis itself, so the snap sailed past it and stopped on the *opposite*
  spine. The two X ticks then spanned a diagonal instead of the axis, and the affine
  transform built from them skewed — producing plausible-looking output that was wrong.
  The overlay still looked perfect, because overlays draw pixel positions rather than
  mapped values.

  Calibration now detects the plot's axis lines directly and repairs a tick pair that is
  tilted further than any real scan could be (>6°), forcing it back onto a common row or
  column. The repair is deliberately narrow: a pair that already agrees is untouched, a
  genuinely rotated axis is preserved, and anything ambiguous is left exactly as it was.
  On the reference set this changed one figure from unusable (R2 −30.9) to R2 0.950, and
  left every other image byte-identical.

### Added

- **Batch mode.** `digitizer.py` now accepts a folder and digitizes every image in it,
  writing `digitized_csvs\`, `digitized_overlays\`, and a `batch_report.csv` / `.json`.
  One bad image cannot ruin the run: failures are isolated and reported per file,
  non-image files are reported as skipped, case-insensitive duplicate names are refused
  rather than silently overwritten, and partial output is cleaned up on failure.
- **Batch accuracy scoring.** `AccuracyTester/accuracy_batch_cli.py` scores a folder of
  digitized CSVs against reference CSVs by file name, emitting per-image metrics and an
  overall failure rate. Ambiguous name matches are reported, never guessed.
- **Tests.** 8 tests covering the calibration repair, plus 49 covering the scoring maths
  and batch file matching.
- **`Digitizer.pyw`** — launches the GUI from source with no console window, reflecting
  the current source without a rebuild.
- Documentation for batch mode, the accuracy statuses, and how to read the metrics
  honestly (see below).

### Changed

- Automatic colour picking now prefers **saturated** colours over merely frequent ones,
  with a full-resolution fallback for hairline curves. This stops it choosing grey
  gridlines over a thin coloured trace.
- OCR axis reading uses a pair-consensus ("tick ladder") check to reject misread tick
  labels, and only overrides the raw extremes on a supermajority — otherwise it falls back
  rather than guessing. Scientific notation and comma-separated labels are now parsed.
- Low OCR confidence (<55%) is now a non-blocking warning instead of a silent result.
- A misleading error that surfaced the *success* string as a failure reason was removed.
- Download links in the docs now point at `releases/latest` instead of a pinned tag.

### Known limitations

- OCR still drops minus signs on negative tick labels and misreads superscript /
  power-of-ten axes. The consensus check repairs a minority of misreads among good ticks,
  but cannot invent a lost minus sign. Use `--axis` to override when this bites.
- **`--axis` values describe the ticks the detector locked onto, not the range of your
  data.** Run `--verbose 1` first and read the `tick -> values` line.
- Correlation and R2 are mathematically undefined against a flat reference, and a vertical
  line is not a function of x. Both are reported as N/A / `invalid_data` by design.

## 2.13

- Log-scale axes, guided manual calibration, build tooling.

## 2.12

- Restructured into separate `Digitizer` and `AccuracyTester` tools.

## 2.11 and earlier

- See the [release history](https://github.com/aj24by7/DataDigitizer/releases).

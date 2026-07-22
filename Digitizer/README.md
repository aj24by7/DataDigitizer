# Data Digitizer 2.14 — User Guide

## What is this?

**Data Digitizer** turns a *picture of a graph* into *numbers you can use*. You give it an image of a chart (a screenshot, a scan, a figure from a paper), and it reads the curve off the picture and hands you back a **spreadsheet-friendly table of data points** (a CSV file) plus an **overlay image** that shows you exactly which points it detected, drawn on top of your original graph. That way you can check its work at a glance.

There are **two ways to use it**:

- **The click-and-go app** — a normal Windows program. Double-click and follow the on-screen steps. No coding, nothing to install. **This is what most people want.**
- **The command line** — a one-line typed command for people who like terminals or want to process images quickly. This needs Python and the source files (see [Using the command line](#using-the-command-line)).

If you just want your data out, head to the [Quick start](#quick-start).

> **New to all this?** Don't worry about the technical words yet. Just follow [Quick start](#quick-start) and then [Using the app](#using-the-app-step-by-step) — every button is explained as you go. You can ignore the entire command-line half of this guide.

---

## Quick start

1. Go to the **[Releases page](https://github.com/aj24by7/DataDigitizer/releases/latest)**. Scroll down until you see a heading called **Assets**, then click the file named **`Digitizer.exe`** to download it. (See [Download & install](#download--install-no-python-needed) for exactly what that page looks like.)
2. **Double-click `Digitizer.exe`** to open it. (*Double-click* means quickly press the left mouse button twice in a row on the file's icon.) On the very first run, Windows shows a blue warning box — click the small **More info** text, then the **Run anyway** button. This is normal and safe; the full explanation is in [Download & install](#the-first-run-windows-warning-expected).
3. In the app, work top to bottom: **Import** your graph image → set the **curve color** → run **Axis Detection** → run **Calibration** → **Place Points** → **Export** to CSV.

That's the whole flow. **Don't worry if those words mean nothing yet** — each one is explained, with exactly what to click, in [Using the app](#using-the-app-step-by-step) below.

---

## What is new in 2.14

- **A calibration bug that could silently skew your numbers is fixed.** In some figures the
  axis detection could lock onto the wrong axis line, which quietly distorted every exported
  value while the overlay still looked perfect. See the [changelog](https://github.com/aj24by7/DataDigitizer/blob/main/CHANGELOG.md) for the
  detail. If you have results from an earlier version that looked right but scored badly,
  they are worth re-running.
- **Batch mode**: point it at a *folder* and it digitizes every image, with a per-image
  report — see [Batch mode](#batch-mode--a-whole-folder-at-once).
- **Batch accuracy scoring** for a whole folder against reference data.
- Better automatic colour picking for thin and black curves, and more robust axis OCR.

---

## Download & install (no Python needed)

You do **not** need Python, and you do **not** need to install anything. The OCR engine (the part that reads the axis numbers) is already built into the app.

### Step 1 — Open the Releases page

Go to <https://github.com/aj24by7/DataDigitizer/releases/latest>.

This is a GitHub *release page*. It shows a title and a description at the top. **Scroll down** past the description until you see a grey heading called **Assets** with a small triangle next to it. If the list under it looks collapsed, click the word **Assets** to open it.

### Step 2 — Download the program

In the **Assets** list, click the file named **`Digitizer.exe`**. (Ignore the green **Source code (zip)** and **Source code (tar.gz)** links — you do not need those.)

Your browser will **download** the file rather than open it. Edge or Chrome may warn that the file is "not commonly downloaded" and offer **Keep** or **Discard** — choose **Keep** (then **Keep anyway** if it asks again). If you accidentally chose Discard, nothing is harmed — just click `Digitizer.exe` again and choose Keep.

> *Optional:* if you also want the accuracy-checking tool, download **`AccuracyTester.exe`** the same way — see the [last section](#accuracy-tester-optional).

### Step 3 — Find the downloaded file

The file lands in your **Downloads** folder. To get there: open **File Explorer** (the yellow folder icon on the taskbar, or press the **Windows key + E**), then click **Downloads** in the left-hand list.

You can leave it there, or drag it somewhere easy to find like your **Desktop**. There is nothing to "install" — the `.exe` *is* the program.

### Step 4 — Launch it

**Double-click `Digitizer.exe`** (quickly press the left mouse button twice in a row on its icon) to open the app.

### The first-run Windows warning (expected)

The first time you run it, Windows SmartScreen may show a blue box that says **"Windows protected your PC."** This is **not** a virus warning. It appears because the file is not "code-signed" (a paid certificate the author hasn't bought), and Windows is cautious about any program it hasn't seen before.

To run it:

1. The box at first shows only a **Don't run** button — **do not click that.** Instead, click the small **More info** text in the middle of the box.
2. A **Run anyway** button now appears. Click it.

*(If you accidentally clicked **Don't run**, nothing bad happened — just double-click `Digitizer.exe` again and try the steps above.)*

You'll only need to do this once.

### Nothing happens at all when you double-click? (Smart App Control)

If double-clicking `Digitizer.exe` produces **no window, no error, no warning box** -- just
nothing -- you are almost certainly running into **Smart App Control**, which is **on by
default on new Windows 11 installations**. It is stricter than SmartScreen: where
SmartScreen warns and offers *Run anyway*, Smart App Control blocks unsigned programs
**silently**. There is no dialog and no *Run anyway* button, so the app simply appears to
do nothing.

**How to confirm it:** open **Windows Security -> App & browser control -> Smart App
Control**. If it says **On**, that is what is stopping the app. You can also check
**Event Viewer -> Applications and Services Logs -> Microsoft -> Windows -> CodeIntegrity
-> Operational** for a *"Smart App Control Block"* entry naming `Digitizer.exe`.

**What you can do:**

- **Run it from source instead (recommended, no downsides).** This is not blocked, because
  Python itself is signed. See [Run from source](#run-from-source-for-developers), then
  **double-click `Digitizer.pyw`** -- it opens the same GUI with no console window. This is
  the best option if you already have Python.
- **Ask for a signed build.** The only real fix is for the app to be code-signed with a
  certificate. Until then, Smart App Control will keep blocking it on machines where it is
  enabled.
- **Turning Smart App Control off** does allow the app to run, but think carefully first:
  > **This is a one-way door.** Microsoft does not let you turn Smart App Control back on
  > once it is off -- re-enabling it requires **resetting or reinstalling Windows**. It
  > also lowers protection for *every* program on the machine, not just this one. Prefer
  > running from source above.

This is not specific to version 2.14 -- it affects every unsigned release equally.

---

## Using the app (step by step)

When the app opens you'll see your future graph in the big main area, a floating **Cursor Zoom** panel in the top-right corner, and a **status bar** along the bottom. A few things to know up front:

- **The status bar is the thin strip along the very bottom edge of the window.** It shows a short message after every action — **watch it.** It confirms success or tells you why something didn't work. At the start it reads *"Load an image to begin."*
- **Cursor Zoom** is just a magnifying glass — it shows an enlarged view of whatever is under your mouse so you can click precisely. It's a helper only; it has nothing to do with exporting.
- **Import, Tools, Advanced, and Export are buttons along the top of the window.** Clicking one opens a small menu of choices. Throughout this guide, "open the **Tools** button → **Pick Color**" means: click **Tools** at the top, then click **Pick Color** in the menu that drops down.

Follow these steps in order.

### 1. Load your graph image

Click the **Import** button (top-left), then choose:

- **Import Image** — to pick a file from your computer (PNG, JPG, JPEG, BMP, TIF/TIFF), or
- **Paste Image** — to use a screenshot you've already copied to the clipboard.

Your chart appears in the main area. The app automatically guesses your curve's color and shows it as the first colored square in the **Selected color** row at the bottom of the window.

### 2. Confirm or set the curve color

Look at the auto-picked color square in the **Selected color** row (the strip of colored squares near the bottom).

- If it already matches your data line, **keep it** — you're done with this step.
- To set it yourself: open the **Tools** button, click **Pick Color** (your cursor becomes a crosshair), then click directly on the curve in the image. The color square updates, and Pick Color turns itself off.

> **Tracing more than one curve?** Click the small green **`+`** square in the Selected color row to add another color slot, select it, then use Pick Color again for that line. (Most people only have one curve and can skip this.)

### 3. Read the axis numbers automatically

Open the **Advanced** button → **Axis Scale Detection** submenu → click **Run Axis Detection**.

The app uses OCR (text recognition) to read the numbers printed along your axes and automatically fills in the four boxes in the **Min-Max Coordinates** panel: **X min, X max, Y min, Y max**. The same submenu also displays the detected values for reference.

### 4. Check and fix the axis numbers

Look at the four **Min-Max Coordinates** boxes. OCR is good but not perfect, so **always sanity-check them.** If any value is missing or misread, just **type the correct number into the box** yourself.

> These four values are required. Export will not run until all four boxes contain valid numbers.

### 5. Calibrate the plot box

This tells the app exactly where your plotted area sits in the image. Open **Tools → Calibration**, then choose one:

- **Coordinate-Mediated Calibration** *(recommended — easiest)* — run **Axis Detection** first (step 3), because this option uses those detected axis positions. It automatically snaps a **dashed green box** onto your plot area. **If you ran step 3, use this and you can skip manual calibration entirely.**
- **Manual Calibration** *(by hand)* — you click four ticks **directly on your chart image**. A **large green step-by-step banner** appears above the image and walks you through each click in order: **X MIN** (smallest X), **X MAX** (largest X), **Y MIN** (smallest Y), **Y MAX** (largest Y), telling you exactly which point to click at each step. You are clicking *locations on the picture*, not typing numbers. The points don't have to be in textbook positions — the box is built correctly from wherever you click, as long as **X MIN is left of X MAX** and **Y MIN is below Y MAX**.

Either way, you'll end up with a **dashed green box** (a rectangle drawn with a dashed green outline) marking the plotted region.

> If you pick Coordinate-Mediated Calibration without running Axis Detection first, the status bar will say *"Run Axis Scale Detection first."*

### 6. Adjust the box if needed (optional)

If the dashed green box edges don't line up exactly with your axes, hover over an edge until the cursor becomes a **resize arrow**, then **drag** it into place. The app remembers your adjusted box for export.

### 7. Extract the data points

Open **Tools → Place Points → Run**.

The app scans the image for your chosen color and overlays the detected points on the curve. The **status bar tells you how many points were placed.**

> **Stray text or a legend getting picked up as data?** Use the optional **Masking** tools first. Under **Advanced → Masking** you can blank out words, numbers, or a legend, or hand-draw mask rectangles (Manual Mask Mode), so they aren't mistaken for your curve. Do this *before* Place Points. There's also a **Clear Masks** option to undo them.

### 8. Choose which colors to export (only for multiple curves)

If you traced more than one curve: open the **Export** button → **Colors to Export** submenu, and tick the color slots you want (or choose **Active Color Only** / **All Configured Colors**). **With a single curve this is already handled for you** — skip it.

### 9. Export your data

Open the **Export** button and choose:

- **Export CSV → Raw values** — a plain text spreadsheet file that opens in Excel, Google Sheets, Python, etc.
- **Export Excel → Raw values** — a real Excel `.xlsx` workbook.

A standard Windows **Save As** dialog appears. **Note which folder is shown at the top** (Downloads or Desktop are easy choices so you can find it again), keep the suggested name **`digitized_points.csv`**, and click **Save**.

**That's it — you now have a spreadsheet of your graph's data.** To open it, go to the folder you just saved into and **double-click the CSV file**; it opens in Excel by default, where you'll see your X and Y columns.

### Where your files go

Exports go **wherever you choose in the Save dialog** — they are not forced into a fixed folder, so make a note of the location you pick. The app suggests the filename **`digitized_points.csv`** (or **`digitized_points.xlsx`** for Excel) and forces the correct extension.

Each row in the file is one detected point, with these columns:

| Column | Meaning |
| --- | --- |
| `color_slot` | which color slot the point belongs to (1 for a single curve) |
| `color_r`, `color_g`, `color_b` | the curve's color (0–255 each) |
| `x`, `y` | the real-world data values |
| `x_px`, `y_px` | the original pixel position in the image |

The CSV is saved as UTF-8, with one header row followed by one row per point.

> **Excel export needs a helper library** (`openpyxl`). In the prebuilt app it's already included. If you're running from source and it's missing, the status bar will say so and nothing is saved — use CSV instead, or install it (see [Run from source](#run-from-source-for-developers)).

### Two things that block export (and how the app tells you)

Export will **quietly refuse** and show a status message unless **both** of these are true:

1. **The four Min-Max Coordinate boxes contain valid numbers** — otherwise you'll see *"Export requires X/Y min/max values."*
2. **A calibration has been run** (the dashed green box exists) — otherwise you'll see *"Export requires calibration."*

If either message appears, glance at the status bar, go back, and finish that step.

### Log-scale axes (optional)

Most plots are linear, so leave this alone for a normal graph — but if your graph uses a **logarithmic** X or Y axis, turn it on with the small **log** toggle buttons that sit just to the **right of the Min-Max Coordinates boxes** (one on the X-min/X-max row, one on the Y-min/Y-max row):

- Click the **log** button beside the X boxes and/or the one beside the Y boxes. It turns **green** when it's on.
- Calibrate and enter the axis min/max **as the real numbers printed on the axis** (e.g. `1` and `1000`), exactly as you would for a linear plot.
- On export, that axis is interpolated in log space, so the values come out correct for a log scale. Leave the buttons off (grey) for normal linear axes.

> A log axis needs **positive** min/max values (you can't take the log of zero or a negative number). If a value is zero or negative, export will tell you instead of producing wrong numbers.

### Tips for the best results

- A **clean, distinctly-colored curve** works best. Point extraction matches your selected color within a tolerance and takes the vertical midpoint of matching pixels in each column.
- **Loading a new image resets everything** — points, color slots, calibration, and the coordinate boxes. Finish one image before loading the next.
- If **Axis Detection** ever says Tesseract OCR isn't available, just **type the four axis values in by hand** (step 4) — everything else still works.
- Stuck on an error? **Advanced → Error Log** opens a window of recent errors.

---

# For developers

> **Using the click-and-go app? You can stop here.** Everything below is the optional command-line and build-from-source material for people comfortable with a terminal. If you just want your data, you're already done — see [Troubleshooting](#troubleshooting) if anything went wrong.

## Using the command line

> **Important:** The command line runs from the **source files with Python installed** — see [Run from source](#run-from-source-for-developers). The prebuilt **`Digitizer.exe` is the click-and-go app only**: it always opens the graphical window and ignores any command-line options. If you want to type commands, you need the source code and Python, not the `.exe`.

### Open a terminal in the Digitizer folder

1. Open the folder that contains the Digitizer source files in **File Explorer**.
2. **Right-click an empty area** inside the folder.
3. Choose **Open in Terminal** (or **Open PowerShell window here**).

A blue/black window opens, already pointed at the right folder. Type your commands there.

> The examples below use the placeholder image name **`graph.png`**. Substitute the real name of your own image file. The image must be either in the folder the terminal is pointed at, or in your Downloads folder, or you must give its full path in quotes.

### The simplest command

Give it an image. That's it:

```powershell
py digitizer.py graph.png
```

With nothing else, the app automatically:

1. **detects the curve color**,
2. **detects the axes and tick positions** with OCR (so it reads X min / X max / Y min / Y max straight off the image), and
3. **writes a CSV of data points plus an overlay PNG** to your **Downloads** folder.

A bare filename like `graph.png` is looked up first in the current folder, then in your Downloads folder — so the one-liner works as long as the image is in either place. By default it stays quiet: it prints just a success line and the output folder. Add **`--verbose 1`** to see the full detail — the color, pixel coordinates, the OCR readings, the point count, and the **OCR confidence score** (see [Seeing more detail](#seeing-more-detail---verbose)).

> Two other spellings do exactly the same thing: `py digitizer_2_11.py cli graph.png` and `py digitizer_2_11.py digitize graph.png`.

### Options

Add any of these to the command for more control:

| Option (aliases) | What it does | Default |
| --- | --- | --- |
| `pic_path` *(positional)* | Path to the image. A bare filename is searched in the current folder, then Downloads. Either this **or** `--pic-dir` is required. | none (required) |
| `--pic-dir` | Another way to give the image path. If both are given, `--pic-dir` wins. | none |
| `--color R,G,B` | Curve color, e.g. `255,0,0` (each part 0–255). Also accepts `[r,g,b]`. **Must match your curve's actual color**, or the run finds no points. | auto-detected |
| `--axis-values` *(`--axis`)* | Axis numbers as `xmin,xmax,ymin,ymax`. X min must differ from X max, and Y min from Y max. | read by OCR |
| `--ticks` *(`--tick-setting`, `--tick-coordinates`)* | Four tick pixel points in `x_min,x_max,y_min,y_max` order, e.g. `[10,200],[500,200],[10,200],[10,20]`. | found by OCR |
| `--log-x` | Read the X axis in base-10 **log** space (X min/max must be positive). Mirrors the GUI's **log** toggle. | off (linear) |
| `--log-y` | Read the Y axis in base-10 **log** space (Y min/max must be positive). Mirrors the GUI's **log** toggle. | off (linear) |
| `--output-dir` *(`--out`, `-o`)* | Folder to save into. Created if it doesn't exist. | your Downloads folder |
| `--verbose N` *(`-v`)* | How much to print. `1` (or a bare `-v` / `--verbose`) shows the color, pixel coords, tick→OCR values, point count, and OCR confidence, and writes a `<image>_log.txt`. `0` prints only success + the output folder. | `0` (quiet) |
| `--json` | Print the result details as JSON instead of plain text. | off |
| `-h` / `--help` | Show the usage help and exit. | — |
| `--normalize-y` *(optional extra)* | Adds an extra `y_norm` column (Y rescaled to 0–1 over the axis range). Leave it off for normal use. | off |
| `--limit-to-calibration` *(optional extra)* | Export only points inside the calibration box. **This is the CLI default.** | on |
| `--no-limit-to-calibration` *(optional extra)* | Also export points that fall outside the calibration box (matches the GUI default). | not set |

### The "fill-in-the-blank" template and function-call style

If you'd like a copy-and-edit reference, run:

```powershell
py digitizer.py template
```

This **prints a fill-in-the-blank line** (it does **not** digitize anything). It shows a minimal example, a full example listing every option, and a short legend explaining each value.

You can then run that **function-call form** — one quoted line. In PowerShell, wrap the whole `digitizer_cli(...)` call in **single quotes** so the inner double quotes survive.

**Minimal example** (just the image; everything else auto-detected). This is the form to copy first:

```powershell
py digitizer.py 'digitizer_cli(pic_dir="graph.png")'
```

**Full example** — the `color`, `axis_values`, and `tick_setting` values below are **placeholders you must edit to match your own chart.** A leftover `color=(255,0,0)` (red) on a non-red curve finds no points and exits with *"produced no points."* Either delete the options you don't need (so those values auto-detect) or replace them with your real values:

```powershell
py digitizer.py 'digitizer_cli(pic_dir="graph.png", color=(255,0,0), axis_values=(0,10,0,100), tick_setting=([10,200],[500,200],[10,200],[10,20]), log_x=False, log_y=False, output_dir="C:/Users/You/Downloads/out", verbose=1, json=False, normalize_y=False, limit_to_calibration=True)'
```

The values you can put inside `digitizer_cli(...)`:

| Name (aliases) | What it is | Default |
| --- | --- | --- |
| `pic_dir` *(`pic_path`, `plot_location`, `image`, `image_path`)* | **Required.** The image path; a bare filename is looked up in the current folder, then Downloads. | none (required) |
| `color` *(`rgb`, `color_rgb`)* | Curve color as `(R,G,B)` or `"255,0,0"`. Blank = auto-detect. Must match your curve's real color. | auto-detected |
| `tick_setting` *(`ticks`, `tick_coordinates`, `tick_coordinate`)* | Four tick pixel points in `x_min,x_max,y_min,y_max` order. Blank = OCR. | OCR-detected |
| `axis_values` *(`axis`, `bounds`)* | Axis numbers as `(xmin,xmax,ymin,ymax)`. Blank = OCR. | OCR-detected |
| `log_x` *(`logx`, `x_log`)* | `True` reads the X axis in base-10 log space (X min/max must be positive). | False |
| `log_y` *(`logy`, `y_log`)* | `True` reads the Y axis in base-10 log space (Y min/max must be positive). | False |
| `output_dir` *(`out_dir`, `out`)* | Folder to save into. Blank = Downloads. | Downloads |
| `verbose` *(`v`)* | `1` prints full detail and writes a `<image>_log.txt`; `0` stays quiet. | `0` |
| `json` *(`as_json`, `print_json`)* | `True` prints full details as JSON. **Only accepted in this function-call form** (the flag version is `--json`). | False |
| `normalize_y` *(`normalize`)* — *optional extra* | `True` adds the `y_norm` (0–1) column. Leave off for normal use. | False |
| `limit_to_calibration` *(`limit`)* — *optional extra* | `True` keeps only points inside the calibration window. | True |

Arguments can be **positional** in the order `pic_dir, color, tick_setting, axis_values, output_dir`, or by **name** using any alias above. Empty positions are allowed and fall back to defaults, and the words `none` / `null` / blank are treated as "auto".

### Worked examples

Auto-detect everything (color and axes), save to Downloads — **the safest example to copy as-is**:

```powershell
py digitizer.py graph.png
```

Set the axis values yourself but still auto-detect the color:

```powershell
py digitizer.py graph.png --axis 0,10,0,100
```

Force a specific curve color — **change `130,5,255` to your curve's real R,G,B**; a color that doesn't match the curve produces no points:

```powershell
py digitizer.py graph.png --color 130,5,255 --axis 0,10,0,100
```

Use a full path, choose an output folder, and read the Y axis on a **log scale**:

```powershell
py digitizer.py "C:\path\to\graph.png" --out "C:\path\to\out" --log-y
```

See everything the run worked out — color, pixel coordinates, the tick→OCR values, point count, and the **OCR confidence score** — and drop a `log.txt` alongside the output:

```powershell
py digitizer.py graph.png --verbose 1
```

> **Tip:** There's also an interactive wizard that asks for each field one at a time. Start it with `py digitizer_2_11.py interactive` (or `wizard`).

### Seeing more detail (`--verbose`)

By default the command line is **quiet** — it just confirms success and tells you the output folder. When you want to see (or keep a record of) exactly what the tool did, raise the verbosity:

```powershell
py digitizer.py graph.png --verbose 1
```

At level `1` it prints, and the run also writes a `<image>_log.txt`, the following:

- **color (r,g,b)** — the curve color it used.
- **pixel coords** — where the four axis ticks sit in the image, in pixels.
- **tick → values** — the four axis numbers (X min, X max, Y min, Y max) the tool used, and whether it read them automatically or you typed them in.
- **OCR confidence** — how sure the built-in number-reader (Tesseract) was about the axis numbers it read off your image, as a percentage. A low number is a hint to double-check the numbers, or to type them in yourself with `--axis`. Shows **n/a** only when you supply *both* the axis values **and** the tick pixel coordinates yourself (so no reading is done); if you pass `--axis` but let the tick positions auto-detect, the reader still runs and a percentage is shown.
- **num points** — how many data points were extracted.
- **elapsed (s)** — how long the run took, in seconds.

Level `0` (the default) stays quiet; you can also write just `--verbose` or `-v` as shorthand for level `1`.

### Output files

Every run writes two files (named after your image), plus a third at `--verbose 1`:

- **`<imageName>_digitized_points.csv`** — the table of digitized points.
- **`<imageName>_digitized_overlay.png`** — a copy of your image with the calibration box, the OCR axis-detection rectangles, and the detected points drawn on top (in the inverse of the curve color, so they stand out).
- **`<imageName>_log.txt`** — *(only with `--verbose 1`)* a plain-text record of the run: time, color, pixel coords, tick→OCR values, point count, OCR confidence, and the output paths.

The CSV columns:

| Column | Meaning |
| --- | --- |
| `color_slot` | 1-based color slot number (1 for a single-color run). |
| `color_r` | Red component (0–255) of the curve color. |
| `color_g` | Green component (0–255) of the curve color. |
| `color_b` | Blue component (0–255) of the curve color. |
| `x` | Digitized X value in data coordinates (rounded to 6 decimals). |
| `y` | Digitized Y value in data coordinates (rounded to 6 decimals). |
| `x_px` | Raw X pixel position in the image. |
| `y_px` | Raw Y pixel position in the image. |
| `y_norm` | *Only when `--normalize-y` is on.* Y rescaled to 0–1 over the axis range; appended as the last column (rounded to 6 decimals). |

---

## Batch mode — a whole folder at once

Point `digitizer.py` at a **folder** instead of a single image and it digitizes every image
inside, one after another:

```powershell
py digitizer.py "C:\path\to\my_images" --output-dir "C:\path\to\results"
```

Every option from the [table above](#options) still applies and is used for **all** images
in the folder — so `--color 0,0,0` means "every curve in this folder is black". The two
exceptions are `--pic-dir` (the folder is the positional argument here) and `--json` (batch
already writes machine-readable `batch_report.json`); both are rejected by the batch parser.

### What batch mode gives you

```text
results\
    digitized_csvs\       one <image>_digitized_points.csv per image
    digitized_overlays\   one <image>_digitized_overlay.png per image
    batch_report.csv      per-image: ok / failed / skipped, point count, reason
    batch_report.json     the same, plus totals and the overall failure rate
```

Open a few overlays: if the coloured dots sit on the curve, that image digitized well.

> **An overlay can look perfect while the numbers are wrong.** Overlays draw *pixel*
> positions, so they show which pixels were detected — not whether those pixels were
> converted to the right values. A broken axis calibration produces a beautiful overlay and
> useless data. Check the CSV, not just the picture.

### How it handles trouble

Batch mode is built so **one bad image cannot ruin the run**:

- **Per-file isolation** — an image that fails is recorded as `failed` with the reason, and
  the run continues to the next one.
- **Non-image files** are reported as `skipped`, never silently ignored.
- **Duplicate names** (case-insensitively, e.g. `Plot.png` and `plot.PNG`) are refused
  rather than one silently overwriting the other.
- **Partial output is cleaned up** if a file fails midway, so you never keep half a CSV.
- Every input file appears in `batch_report.csv` exactly once. The report is the record of
  what happened; nothing is dropped without a line explaining it.

### Scoring a batch against known-good data

If you have reference ("ground truth") CSVs, the Accuracy Tester scores a whole folder by
matching file names:

```powershell
cd ..\AccuracyTester
py accuracy_batch_cli.py --digitized-dir "C:\path\to\results\digitized_csvs" --ground-truth-dir "C:\path\to\truth"
```

It writes `accuracy_report\accuracy_report.csv` (one row per image: RMSE, correlation, MSE,
R2, and more, plus a `status`) and `accuracy_summary.json` (the overall failure rate).

Matching is deliberately strict: a `_digitized_points` / `_digitized` / `_points` / `_gt`
suffix is stripped before comparing stems, exact matches win, and if several files match one
reference equally well it reports `ambiguous_match` rather than guessing.

| Status | Means |
| --- | --- |
| `ok` | Pair scored; metrics present. |
| `no_method_match` | A reference file with no digitized file to compare against — that image failed to digitize. |
| `no_ground_truth_match` | A digitized file whose name matches no reference. |
| `ambiguous_match` | Several files matched one reference equally well; reported, never guessed. |
| `invalid_data` | Can't be scored as y=f(x) — e.g. a vertical line, where one x has many y values. |
| `insufficient_overlap` | The two curves share too little X range to compare. |
| `parse_error` | A CSV could not be read (corrupt, empty, malformed). |

### Reading the accuracy numbers honestly

Two traps worth knowing, because the raw CSV will happily print a number in both cases:

- **Quote `nrmse_range_pct`, not raw RMSE.** RMSE carries the units of the Y axis. An RMSE
  of 63.9 on an axis running to 30,000 is *better* than 0.03 on an axis running 0–1. The
  `nrmse_range_pct` column expresses the error as a percentage of that figure's own Y range,
  which is comparable across figures.
- **Correlation and R2 are undefined against a flat reference.** Both divide by the variance
  of the reference data, which is zero for a horizontal line. The CSV prints a degenerate
  value (e.g. `-2.6e+25`); the honest reading is **N/A**, not "terrible accuracy".

---

## Run from source (for developers)

You'll need a **recent Python 3** installed (from <https://python.org> — tick *"Add python.exe to PATH"* during setup).

> **OCR needs the Tesseract program, not just the Python package.** `requirements.txt`
> installs `pytesseract`, which is only a *wrapper* — it cannot read anything on its own.
> For axis detection to work from source, install **Tesseract 5.5 for Windows** (the UB
> Mannheim build) and let the installer add it to your PATH, or set `TESSERACT_CMD` to its
> full path. Without it, `py digitizer.py graph.png` stops with
> *"Tesseract OCR executable not found"*. Passing `--axis` is **not** enough on its own —
> tick *positions* are still read by OCR, so you would need `--ticks` as well. The prebuilt
> `Digitizer.exe` from Releases has Tesseract bundled and needs none of this.

1. Get the source code — download it from the [repo](https://github.com/aj24by7/DataDigitizer) or clone it:

   ```powershell
   git clone https://github.com/aj24by7/DataDigitizer.git
   ```

2. Open a terminal in the `Digitizer` folder and install the dependencies:

   ```powershell
   py -m pip install -r requirements.txt
   ```

3. Launch the graphical app:

   ```powershell
   py digitizer_2_11.py
   ```

   Or just **double-click `Digitizer.pyw`** in the `Digitizer` folder. That opens the same
   window with **no console/command prompt behind it**, so it behaves like a normal desktop
   app -- and because it loads the source directly, it always reflects the current code with
   no rebuild. This is also the way to run the app if
   [Smart App Control](#nothing-happens-at-all-when-you-double-click-smart-app-control) is
   blocking the `.exe`.

The command-line interface ([above](#using-the-command-line)) also runs from this same source setup.

The dependencies are: `PyInstaller` (for building the exe), `PyQt6` (the window), `numpy`, `pillow` (image handling), `pytesseract` (OCR), and `openpyxl` (Excel export).

---

## Build pipeline — turning the code into an app

The project is plain Python (a PyQt6 GUI plus a small CLI). There are three "shapes" you can run it as, and the conversion is the same idea each time — **[PyInstaller](https://pyinstaller.org)** bundles the Python interpreter, this code, and the dependencies into one self-contained program described by a `.spec` file:

| Target | What it is | How it's made |
| --- | --- | --- |
| **CLI** | `python3 digitizer.py …` | No build needed — it runs straight from the source files with Python (see [Run from source](#run-from-source-for-developers)). |
| **Windows `.exe`** | a one-file `Digitizer.exe` you double-click | PyInstaller reads `digitizer.spec`, bundles the GUI entry `digitizer_desktop.py` **and** the Tesseract OCR runtime in `vendor\tesseract`, and writes `dist\Digitizer.exe`. |
| **macOS `.app`** | a `Digitizer.app` bundle | Same tool, the macOS specs/scripts in the separate **[digitizer_mac](https://github.com/RayanA07/digitizer_mac)** repo (`bash build_macos.sh`) produce `Digitizer.app`. |

### Before you build from a fresh clone

The bundled OCR runtime under `Digitizer\vendor\tesseract` is **not committed to this
repository** — it is a third-party binary distribution of ~83 MB with its own licence, so it
is excluded by `.gitignore`. A fresh clone will not have it. The build still succeeds, but it
prints a loud warning and the resulting exe has **no OCR** -- axis detection will not work for
anyone you hand it to, even though it may appear to work on your own machine if you have
Tesseract installed separately.

To restore it, install **Tesseract 5.5 for Windows** (the UB Mannheim build) and copy the
installed folder — `tesseract.exe`, its DLLs, and `tessdata` — to
`Digitizer\vendor\tesseract`. This repo was built against `tesseract v5.5.0.20241111`
with leptonica 1.85.0.

If you only want to **run from source** rather than build the exe, you do not need this:
install Tesseract normally and it will be found on your PATH. And if OCR is unavailable
entirely, everything still works — you just type the four axis values in by hand.

### Starter command (Windows)

One-click: double-click **`build.cmd`** in this folder. (It just runs the PowerShell build with the execution policy unblocked, then keeps the window open so you can read the result.)

Or from a terminal in this folder:

```powershell
py -m pip install -r requirements.txt
.\build_windows.ps1
```

Either way the result is **`dist\Digitizer.exe`** — a windowed, one-file app you can double-click. The **Tesseract OCR runtime** under `vendor\tesseract` is **bundled into the exe**, so OCR features (axis-value detection and the text/number/legend masking tools) work without anyone needing a separate Tesseract install.

> **macOS starter:** in the `digitizer_mac` repo it's `bash build_macos.sh` (in each app folder), which vendors Tesseract via Homebrew and produces the `.app`. That repo also builds on a GitHub Actions macOS runner automatically.

---

## Troubleshooting

**"Windows protected your PC" when I open the .exe.**
Expected on the first run — the file just isn't code-signed. Click the small **More info** text, then **Run anyway**. (Full explanation in [Download & install](#the-first-run-windows-warning-expected).)

**I can't find `Digitizer.exe` after downloading it.**
It's in your **Downloads** folder. Open **File Explorer** (the yellow folder icon on the taskbar, or **Windows key + E**) and click **Downloads** on the left.

**"`py` is not recognized" in the terminal.**
Python isn't installed (or wasn't added to your PATH). Install it from <https://python.org> and make sure to tick **"Add python.exe to PATH"** during setup. Then open a fresh terminal and try again. (Reminder: the command line needs Python; the `.exe` does not.)

**It picked the wrong curve color.**
- *In the app:* use **Tools → Pick Color** and click your curve (step 2).
- *On the command line:* pass it yourself, e.g. `--color 130,5,255` (flags) or `color=(130,5,255)` (function-call form) — using your curve's real R,G,B.

**The command line says "produced no points" / exits without a CSV.**
The color it's matching doesn't match your curve. If you passed `--color` (or `color=`), the values are wrong — fix them, or drop the option entirely to let it auto-detect. The placeholder `255,0,0` (red) in the template will fail on any non-red plot.

**The axis numbers came out wrong.**
OCR can misread numbers. Always check them.
- *In the app:* type the correct value into the **Min-Max Coordinates** box.
- *On the command line:* give them with `--axis xmin,xmax,ymin,ymax` (flags) or `axis_values=(xmin,xmax,ymin,ymax)` (function-call form). Run with `--verbose 1` to see the **OCR confidence score** — a low number is a strong hint that you should type the axis values in yourself with `--axis`.

**"Image file not found" on the command line.**
Put the image in your **Downloads** folder or in the folder the terminal is pointed at (a bare filename is searched in both), or pass the **full path in quotes**, e.g. `py digitizer.py "C:\Users\You\Pictures\graph.png"`.

**Export does nothing in the app.**
Check the status bar (the thin strip at the bottom). You need **all four Min-Max Coordinate boxes filled** *and* **a calibration run** (the dashed green box). Finish whichever is missing, then export again.

**Where did my files go?**
- *In the app:* wherever you chose in the **Save As** dialog.
- *On the command line:* your **Downloads** folder by default, or the folder you set with `--out` / `output_dir=`.

---

## Folder contents

```text
digitizer.py        # Entry point: dispatches single-image vs batch (folder) runs
digitizer_2_11.py   # GUI + CLI dispatch and runtime path setup
digitizer_desktop.py# GUI-only entry used by the PyInstaller build
Digitizer.pyw       # Runs the GUI from source with no console window (no rebuild needed)
digitizer_cli.py    # Terminal CLI (single image)
digitizer_batch_cli.py # Terminal CLI (whole folder) + batch_report
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
build.cmd           # One-click build (double-click this)
build_windows.ps1   # Build script
requirements.txt    # Python dependencies
requirements-dev.txt# Test-only deps (pytest)
version.json        # App name + version (read at runtime by app_version.py)
app_version.py      # Single source of truth for the version string
test_calibration.py # Tests for the axis-pair repair in calibration
assets/             # Application icon
vendor/tesseract/   # Bundled Tesseract OCR runtime
```

---

## Accuracy Tester (optional)

This is a **separate, optional** tool. It's completely independent of the main Digitizer — you don't need it to get your data out.

**What it does:** it measures how faithfully a digitized curve reproduces a reference curve. You load **two CSV files** — an *original* reference and a *digitized* one — and it lines them up on a common X grid and reports accuracy numbers (MAE, RMSE, R-squared, bias, MAPE/sMAPE/WAPE, and more) alongside diagnostic plots: the curve overlay, residuals, absolute error, and zoomable outliers. It cleans messy data (drops non-numeric rows, collapses duplicate X values), can filter by color slot, and can optionally optimize a constant X-shift before comparing. Results can be exported to a comparison CSV.

**Run the app (no Python needed):**
Double-click **`AccuracyTester.exe`** (download it from [Releases](https://github.com/aj24by7/DataDigitizer/releases/latest), or, if you built from source, find it in the `AccuracyTester` folder). It's a windowed, one-file app — no install or console needed.

**Run from source:**

```powershell
py accuracytester_desktop.py
```

---

*Runtime logs are written to `%LOCALAPPDATA%\DataDigitizer\2.14\logs` (you can also view them in-app via **Advanced → Error Log**).*

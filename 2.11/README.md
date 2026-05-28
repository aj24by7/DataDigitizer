# Data Digitizer 2.11

2.11 adds terminal/API wrappers and Windows one-file build assets around the 2.10 digitizer runtime. The digitizer modules and algorithms are preserved; new behavior lives in wrapper files.

## Run GUI

```powershell
py digitizer_2_11.py
```

Double-clicking the built executable also launches the GUI.

## Run CLI

```powershell
py digitizer_2_11.py cli --pic-dir path\to\plot.png --color 255,0,0 --ticks "[10,200],[500,200],[10,200],[10,20]" --axis-values 0,10,0,100
```

Inputs:

- `--pic-dir`: path to the image file.
- `--color`: RGB as `r,g,b`; omit or pass `null` to use auto color selection.
- `--ticks`: four pixel points in `x_min,x_max,y_min,y_max` order; omit or pass `null` to use OCR axis detection plus coordinate-mediated calibration.
- `--axis-values`: `xmin,xmax,ymin,ymax`; omit or pass `null` to use OCR axis detection.
- `--output-dir`: output folder; defaults to the image folder.

Outputs:

- `<image>_digitized_points.csv`
- `<image>_digitized_overlay.png`

## Python API

```python
from function import digitize_image

result = digitize_image(
    pic_dir="plot.png",
    color_rgb=None,
    tick_points=None,
    axis_values=None,
    output_dir=None,
)
print(result.csv_path)
```

## Windows One-File Build

```powershell
.\build_windows.ps1
```

The output is:

```text
dist\DataDigitizer-2.11.exe
```

The executable is built as a single PyInstaller one-file console app so it can be called from a terminal and also launched by double-clicking. OCR still requires the Tesseract engine. If a full Tesseract runtime is placed under `vendor\tesseract` before building, the spec bundles it into the one-file executable.

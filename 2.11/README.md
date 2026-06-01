# Data Digitizer 2.11

2.11 adds terminal/API wrappers and Windows one-file build assets around the 2.10 digitizer runtime. The digitizer modules and algorithms are preserved; new behavior lives in wrapper files.

## Run GUI

```powershell
py digitizer_2_11.py
```

Double-clicking the built executable also launches the GUI.

## Run CLI

Function-call command from the `2.11` folder:

```powershell
py 2.11.py "digitizer_cli(pic_dir='C:/path/to/plot.png', output_dir='')"
```

Full manual call:

```powershell
py 2.11.py "digitizer_cli(pic_dir='C:/path/to/plot.png', color=(255,0,0), tick_setting=([10,200],[500,200],[10,200],[10,20]), axis_values=(0,10,0,100), output_dir='C:/path/to/output')"
```

Blank optional slots use auto/default behavior:

```powershell
py 2.11.py "digitizer_cli(pic_dir='C:/path/to/plot.png', , , , output_dir='')"
```

One-line flag command still works:

```powershell
py digitizer_2_11.py cli --pic-dir path\to\plot.png --color 255,0,0 --ticks "[10,200],[500,200],[10,200],[10,20]" --axis-values 0,10,0,100
```

Inputs:

- `--pic-dir`: path to the image file.
- `--color`: RGB as `r,g,b`; omit or pass `null` to use auto color selection.
- `--ticks`: four pixel points in `x_min,x_max,y_min,y_max` order; omit or pass `null` to use OCR axis detection plus coordinate-mediated calibration.
- `--axis-values`: `xmin,xmax,ymin,ymax`; omit or pass `null` to use OCR axis detection.
- `--output-dir`: output folder; defaults to the image folder.

Function-call inputs:

- `pic_dir`: required path to the image file.
- `color`: RGB tuple such as `(255,0,0)`; omit, leave blank, or pass `null` to use auto color selection.
- `tick_setting`: four pixel points in `x_min,x_max,y_min,y_max` order; omit, leave blank, or pass `null` to use OCR axis detection plus coordinate-mediated calibration.
- `axis_values`: `(xmin,xmax,ymin,ymax)`; omit, leave blank, or pass `null` to use OCR axis detection.
- `output_dir`: output folder; blank defaults to the image folder.

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
dist\digitizer.exe
dist\accuracytester.exe
```

`digitizer.exe` and `accuracytester.exe` are windowed desktop apps with no terminal window. OCR requires the Tesseract engine at `C:\Program Files\Tesseract-OCR\tesseract.exe` or a bundled runtime under `vendor\tesseract`.

## Sending To Another Windows 11 User

The Desktop `digitizer.exe` build can be made fully self-contained by bundling the Tesseract runtime under `vendor\tesseract` before building. The resulting executable is large, so Outlook may block or reject a direct `.exe` attachment. If that happens, send `DataDigitizer-Windows11-Apps.zip` through OneDrive/SharePoint or another file link instead of attaching the raw executable.

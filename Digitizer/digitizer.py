from __future__ import annotations

import sys
from pathlib import Path

from digitizer_2_11 import configure_runtime_paths


USAGE = r"""Data Digitizer 2.13 - command line

Digitize one image. With no options it auto-detects the curve color and the axes
(via OCR) and saves the CSV + overlay to your Downloads folder:

  py digitizer.py plot2.png

A bare filename is searched for in the current folder and in Downloads, so the line
above works as long as plot2.png is in either place.

Add any of these for more control:

  --color  R,G,B        curve color (default: auto-detected)
  --axis   xmin,xmax,ymin,ymax   axis values (default: read by OCR)
  --ticks  [x,y],[x,y],[x,y],[x,y]   tick pixel points: x_min,x_max,y_min,y_max
  --log-x               read the X axis on a base-10 log scale (X min/max > 0)
  --log-y               read the Y axis on a base-10 log scale (Y min/max > 0)
  --out    FOLDER       where to save output (default: Downloads)
  --verbose N (or -v)   N=1 shows full detail (color, pixel coords, tick->OCR,
                        points, OCR confidence) and writes a <image>_log.txt;
                        N=0 (default) prints only success + the output folder
  --json                print the result details as JSON
  --normalize-y         (optional) add a 0-1 normalized Y column to the CSV

Examples:

  py digitizer.py plot2.png --verbose 1
  py digitizer.py plot2.png --color 255,0,0 --axis 0,10,0,100
  py digitizer.py "C:\Users\User\Desktop\graph.png" --out "C:\Users\User\Desktop\out"

Function-call / template style (one quoted line) also works. This is the form to
copy-paste and edit when you want to set several things at once:

  py digitizer.py 'digitizer_cli(pic_dir="plot2.png", color=(255,0,0), axis_values=(0,10,0,100))'

Print a fill-in-the-blank template with every option:

  py digitizer.py template

Digitize a whole FOLDER of images at once (batch mode). Give a folder instead of
a file and every image in it is digitized; outputs land in two subfolders inside
that folder (digitized_csvs/ and digitized_overlays/) plus a batch_report:

  py digitizer.py C:\\charts
  py digitizer.py batch C:\\charts --output-dir D:\\results

To open the graphical app instead:

  py digitizer_2_11.py
"""


def _first_positional(args: list[str]) -> str | None:
    """First argument that is not an option flag or its value (used to sniff a folder)."""
    skip_next = False
    for token in args:
        if skip_next:
            skip_next = False
            continue
        if token.startswith("-"):
            # options that take a separate value; skip the value so it is not mistaken
            # for the positional path
            if "=" not in token and token in {
                "--color", "--ticks", "--tick-setting", "--tick-coordinates",
                "--axis-values", "--axis", "--output-dir", "--out", "-o",
                "--pic-dir", "-v", "--verbose",
            }:
                skip_next = True
            continue
        return token
    return None


if __name__ == "__main__":
    configure_runtime_paths()
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help", "help"}:
        print(USAGE)
        raise SystemExit(0)
    if args[0] in {"template", "--template"}:
        from digitizer_cli import print_template

        print_template()
        raise SystemExit(0)

    # Batch mode: an explicit `batch` keyword, or a positional argument that is a
    # folder. A single image file falls through to the normal single-image CLI.
    if args[0] in {"batch", "--batch"}:
        from digitizer_batch_cli import main as batch_main

        raise SystemExit(batch_main(args[1:]))
    first = _first_positional(args)
    if first is not None and Path(first).expanduser().is_dir():
        from digitizer_batch_cli import main as batch_main

        raise SystemExit(batch_main(args))

    from digitizer_cli import main as cli_main

    raise SystemExit(cli_main(args))

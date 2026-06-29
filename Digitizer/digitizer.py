from __future__ import annotations

import sys

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

To open the graphical app instead:

  py digitizer_2_11.py
"""


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

    from digitizer_cli import main as cli_main

    raise SystemExit(cli_main(args))

"""Windowless launcher that runs the Digitizer GUI straight from THIS folder's
source code. Double-clicking a .pyw file runs it with pythonw.exe, so there is
no console/cmd window -- and because it loads the source live, it always reflects
your latest edits (no rebuild needed). Requires Python 3 with PyQt6 installed.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from digitizer_2_11 import configure_runtime_paths, launch_digitizer_gui

configure_runtime_paths()
raise SystemExit(launch_digitizer_gui())

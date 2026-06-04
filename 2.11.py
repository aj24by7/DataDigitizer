from __future__ import annotations

import runpy
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent / "2.11"
ENTRYPOINT = APP_DIR / "2.11.py"

sys.path.insert(0, str(APP_DIR))
sys.argv[0] = str(ENTRYPOINT)

runpy.run_path(str(ENTRYPOINT), run_name="__main__")

"""Single source of truth for the app name and version.

version.json is already bundled into the exe by digitizer.spec, so read it at runtime
instead of repeating the number in banners and the log path. Those were hardcoded in six
places and silently drifted: 2.14 shipped with every string still reading 2.13, including
the %LOCALAPPDATA%\\DataDigitizer\\<version>\\logs directory. test_app_version.py fails if
version.json and the fallback below ever disagree again.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Tuple

# Used only if version.json is unreadable (a corrupt or partial install). Kept in step with
# version.json by test_app_version.py.
_FALLBACK_NAME = "DataDigitizer"
_FALLBACK_VERSION = "2.14"


def _version_json_path() -> Optional[Path]:
    roots = []
    bundle = getattr(sys, "_MEIPASS", None)  # set when running from the PyInstaller exe
    if bundle:
        roots.append(Path(bundle))
    roots.append(Path(__file__).resolve().parent)
    for root in roots:
        candidate = root / "version.json"
        if candidate.is_file():
            return candidate
    return None


def _load() -> Tuple[str, str]:
    path = _version_json_path()
    if path is None:
        return _FALLBACK_NAME, _FALLBACK_VERSION
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return _FALLBACK_NAME, _FALLBACK_VERSION
    if not isinstance(data, dict):
        return _FALLBACK_NAME, _FALLBACK_VERSION
    name = str(data.get("name") or _FALLBACK_NAME)
    version = str(data.get("version") or _FALLBACK_VERSION)
    return name, version


APP_NAME, APP_VERSION = _load()

# "Data Digitizer 2.14" - the banner every entry point prints.
APP_TITLE = f"Data Digitizer {APP_VERSION}"

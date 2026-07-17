"""Guards against the version drifting out of sync again.

2.14 shipped with version.json bumped but every user-visible string still reading 2.13 --
the CLI banner, the GUI help, the run-log header, and the %LOCALAPPDATA% log directory --
because the number was copy-pasted into six files. These tests fail if any of that recurs.
"""

import json
import re
from pathlib import Path

import app_version

HERE = Path(__file__).resolve().parent
SOURCES = sorted(p for p in HERE.glob("*.py") if p.name not in {"app_version.py", Path(__file__).name})


def test_version_json_is_the_source_of_truth():
    data = json.loads((HERE / "version.json").read_text(encoding="utf-8"))
    assert app_version.APP_VERSION == data["version"]
    assert app_version.APP_NAME == data["name"]


def test_fallback_matches_version_json():
    """The fallback only fires on a broken install, but a stale one would lie about the version."""
    data = json.loads((HERE / "version.json").read_text(encoding="utf-8"))
    assert app_version._FALLBACK_VERSION == data["version"]
    assert app_version._FALLBACK_NAME == data["name"]


def test_no_hardcoded_version_strings_in_sources():
    """Nothing may spell a version number itself; it must come from app_version."""
    pattern = re.compile(r"Data Digitizer\s+\d+\.\d+")
    offenders = []
    for path in SOURCES:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.name}:{number}: {line.strip()}")
    assert not offenders, "hardcoded version strings found:\n" + "\n".join(offenders)


def test_app_title_matches_version():
    assert app_version.APP_TITLE == f"Data Digitizer {app_version.APP_VERSION}"

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
import traceback
from typing import Optional

LOG_DIR = Path(__file__).resolve().parent / "logs"
TXT_PATH = LOG_DIR / "error_log.txt"
CSV_PATH = LOG_DIR / "error_log.csv"


def log_exception(context: str, exc: BaseException) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"{type(exc).__name__}: {exc}"
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    with TXT_PATH.open("a", encoding="utf-8") as handle:
        handle.write("=" * 80 + "\n")
        handle.write(f"Time: {timestamp}\n")
        handle.write(f"Context: {context}\n")
        handle.write(f"Message: {message}\n")
        handle.write("Traceback:\n")
        handle.write(tb)
        handle.write("\n")

    new_file = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(["timestamp", "context", "message", "traceback"])
        writer.writerow([timestamp, context, message, tb])


def read_recent_entries(limit: int = 200) -> str:
    if not TXT_PATH.exists():
        return "No error log entries yet."
    lines = TXT_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    if limit <= 0:
        return "\n".join(lines)
    return "\n".join(lines[-limit:])


def get_csv_path() -> Path:
    return CSV_PATH

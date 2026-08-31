from __future__ import annotations

from app_version import APP_NAME, APP_TITLE, APP_VERSION

import os
import shutil
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    configure_runtime_paths()
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return launch_digitizer_gui()

    command = args[0].lower()
    if command in {"gui", "--gui"}:
        return launch_digitizer_gui()
    if command in {"batch", "--batch"}:
        from digitizer_batch_cli import main as batch_main

        return batch_main(args[1:])
    if command in {"cli", "digitize"}:
        if _folder_positional(args[1:]) is not None:
            from digitizer_batch_cli import main as batch_main

            return batch_main(args[1:])
        from digitizer_cli import main as cli_main

        return cli_main(args[1:])
    if command in {"interactive", "wizard"}:
        from digitizer_cli import interactive_main

        return interactive_main()
    if command in {"template", "--template"}:
        from digitizer_cli import print_template

        print_template()
        return 0
    if command in {"--help", "-h", "help"}:
        print_help()
        return 0
    if _looks_like_cli_invocation(args):
        # A folder means batch mode, exactly as in digitizer.py - this entry point
        # used to reject folders with "pic_dir must point to an image file".
        if _folder_positional(args) is not None:
            from digitizer_batch_cli import main as batch_main

            return batch_main(args)
        from digitizer_cli import main as cli_main

        return cli_main(args)

    print_help()
    return 2


def configure_runtime_paths() -> None:
    """Set writable runtime paths without changing the existing modules."""

    _suppress_subprocess_console()

    app_dir = _user_app_data_dir()
    log_dir = app_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    try:
        import ErrorLogger

        ErrorLogger.LOG_DIR = log_dir
        ErrorLogger.TXT_PATH = log_dir / "error_log.txt"
        ErrorLogger.CSV_PATH = log_dir / "error_log.csv"
    except Exception:
        pass

    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    tessdata = bundle_root / "vendor" / "tesseract" / "tessdata"
    if tessdata.exists() and "TESSDATA_PREFIX" not in os.environ:
        os.environ["TESSDATA_PREFIX"] = str(tessdata)

    tesseract_cmd = _resolve_tesseract_cmd(bundle_root)
    if tesseract_cmd is not None:
        os.environ.setdefault("TESSERACT_CMD", str(tesseract_cmd))
        try:
            import pytesseract

            pytesseract.pytesseract.tesseract_cmd = str(tesseract_cmd)
        except Exception:
            pass


def _suppress_subprocess_console() -> None:
    """Prevent bundled console tools (Tesseract, invoked by pytesseract) from
    briefly flashing a console window when run from the windowed executable."""
    if sys.platform != "win32":
        return
    import subprocess

    if getattr(subprocess, "_digitizer_no_window_patched", False):
        return
    create_no_window = 0x08000000  # CREATE_NO_WINDOW
    original_init = subprocess.Popen.__init__

    def _init(self, *args, **kwargs):
        kwargs["creationflags"] = kwargs.get("creationflags", 0) | create_no_window
        original_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = _init
    subprocess._digitizer_no_window_patched = True


def launch_digitizer_gui() -> int:
    from PyQt6 import QtWidgets

    import ErrorLogger
    from UI import DigitizerWindow

    # PyQt6 calls qFatal() on any exception that escapes a slot, aborting the app and
    # losing the user's digitizing session. Log it and keep running instead; the
    # individual handlers still report their own errors to the status bar.
    def _log_and_continue(exc_type, exc, tb) -> None:
        try:
            ErrorLogger.log_exception("unhandled exception in Qt event", exc)
        except Exception:
            pass
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = _log_and_continue

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(["Digitizer"])
    window = DigitizerWindow()
    window.show()
    return int(app.exec())


def print_help() -> None:
    print(
        "\n".join(
            [
                APP_TITLE,
                "",
                "Open the graphical app:",
                "  double-click Digitizer.exe   (or from source: py digitizer_2_11.py)",
                "",
                "Digitize one image from the command line (saves CSV + overlay to Downloads):",
                "  py digitizer.py plot2.png",
                "",
                "Add detail as needed:",
                "  py digitizer.py plot2.png --color 255,0,0 --axis 0,10,0,100 --out C:/out --log-y",
                "",
                "See everything (color, pixel coords, tick->OCR, points, OCR confidence) + write a log:",
                "  py digitizer.py plot2.png --verbose 1",
                "",
                "Function-call / template style (one quoted line, copy and edit it):",
                "  py digitizer.py 'digitizer_cli(pic_dir=\"plot2.png\", color=(255,0,0), axis_values=(0,10,0,100))'",
                "",
                "Print a fill-in-the-blank template with every option:",
                "  py digitizer.py template",
                "",
                "Full CLI options:",
                "  py digitizer.py --help",
            ]
        )
    )


def _looks_like_cli_invocation(args: list[str]) -> bool:
    joined = " ".join(args).strip()
    if joined.startswith("digitizer_cli("):
        return True
    cli_flags = {
        "--pic-dir",
        "--color",
        "--ticks",
        "--tick-setting",
        "--tick-coordinates",
        "--axis-values",
        "--axis",
        "--log-x",
        "--log-y",
        "--output-dir",
        "--out",
        "-o",
        "--normalize-y",
        "--limit-to-calibration",
        "--no-limit-to-calibration",
        "--verbose",
        "-v",
        "--json",
    }
    if any(arg.split("=", 1)[0] in cli_flags for arg in args):
        return True
    if args:
        first = Path(args[0]).expanduser()
        if first.exists():
            return True
        if len(first.parts) == 1 and (Path.home() / "Downloads" / first.name).is_file():
            return True
    return False


def _folder_positional(args: list[str]) -> str | None:
    """The first positional argument when it is an existing directory, else None."""
    from digitizer import _first_positional

    first = _first_positional(list(args))
    if first is not None and Path(first).expanduser().is_dir():
        return first
    return None


def _user_app_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / APP_NAME / APP_VERSION
    return Path.home() / f".{APP_NAME.lower()}" / APP_VERSION


def _resolve_tesseract_cmd(bundle_root: Path) -> Path | None:
    # Same search order as AxisReader/Masking: env override, vendored copies,
    # standard installs, then PATH - so a PATH-only Tesseract install is found
    # here too instead of relying on the OCR modules to re-resolve later.
    env_cmd = os.environ.get("TESSERACT_CMD", "").strip().strip('"')
    candidates = [
        Path(env_cmd) if env_cmd else None,
        bundle_root / "vendor" / "tesseract" / "tesseract.exe",
        Path(__file__).resolve().parent / "vendor" / "tesseract" / "tesseract.exe",
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    on_path = shutil.which("tesseract")
    return Path(on_path) if on_path else None


if __name__ == "__main__":
    raise SystemExit(main())

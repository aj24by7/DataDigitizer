from __future__ import annotations

import os
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
    if command in {"cli", "digitize"}:
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

    from UI import DigitizerWindow

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
                "Data Digitizer 2.12",
                "",
                "Open the graphical app:",
                "  double-click Digitizer.exe   (or from source: py digitizer_2_11.py)",
                "",
                "Digitize one image from the command line (saves CSV + overlay to Downloads):",
                "  py digitizer.py plot2.png",
                "",
                "Add detail as needed:",
                "  py digitizer.py plot2.png --color 255,0,0 --axis 0,10,0,100 --out C:/out --normalize-y",
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
        "--output-dir",
        "--out",
        "-o",
        "--normalize-y",
        "--limit-to-calibration",
        "--no-limit-to-calibration",
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


def _user_app_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if base:
        return Path(base) / "DataDigitizer" / "2.12"
    return Path.home() / ".datadigitizer" / "2.12"


def _resolve_tesseract_cmd(bundle_root: Path) -> Path | None:
    candidates = [
        bundle_root / "vendor" / "tesseract" / "tesseract.exe",
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


if __name__ == "__main__":
    raise SystemExit(main())

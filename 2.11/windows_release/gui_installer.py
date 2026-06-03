from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path
from tkinter import BOTH, BOTTOM, DISABLED, LEFT, NORMAL, RIGHT, X, Button, Frame, Label, StringVar, Tk, messagebox
from tkinter import ttk


APP_VERSION = "2.11"
RELEASE_BASE_URL = "https://github.com/aj24by7/DataDigitizer/releases/download/v2.11"
INSTALL_DIR = Path(os.environ["LOCALAPPDATA"]) / "DataDigitizer" / APP_VERSION
START_MENU_DIR = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Data Digitizer"
DESKTOP_DIR = Path.home() / "Desktop"

APPS = [
    {"file": "digitizer.exe", "shortcut": "Digitizer.lnk", "description": "Data Digitizer"},
    {"file": "accuracytester.exe", "shortcut": "AccuracyTester.lnk", "description": "Accuracy Tester"},
    {"file": "DataDigitizer-2.11.exe", "shortcut": "DataDigitizer CLI.lnk", "description": "Data Digitizer CLI"},
]


def resource_root() -> Path:
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def is_real_exe(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024 * 1024:
        return False
    with path.open("rb") as handle:
        return handle.read(2) == b"MZ"


def candidate_dirs() -> list[Path]:
    here = Path(sys.argv[0]).resolve().parent
    source_dir = Path(__file__).resolve().parent
    root = resource_root()
    dirs = [
        root / "payload",
        here,
        here / "dist",
        here.parent / "dist",
        here.parent,
        here.parent.parent / "2.11" / "dist",
        source_dir,
        source_dir / "dist",
        source_dir.parent / "dist",
    ]
    unique: list[Path] = []
    for directory in dirs:
        try:
            resolved = directory.resolve()
        except OSError:
            continue
        if resolved not in unique:
            unique.append(resolved)
    return unique


def find_app_exe(file_name: str) -> Path | None:
    for directory in candidate_dirs():
        candidate = directory / file_name
        if is_real_exe(candidate):
            return candidate
    return None


def copy_with_progress(source: Path, destination: Path, progress_callback) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = source.stat().st_size
    copied = 0
    with source.open("rb") as src, destination.open("wb") as dst:
        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            copied += len(chunk)
            progress_callback(copied, total)


def download_with_progress(file_name: str, destination: Path, progress_callback) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = f"{RELEASE_BASE_URL}/{file_name}"
    with urllib.request.urlopen(url) as response, destination.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            progress_callback(downloaded, total or downloaded)
    if not is_real_exe(destination):
        raise RuntimeError(f"{file_name} did not download as a valid Windows executable.")
    return destination


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def create_shortcut(shortcut_path: Path, target_path: Path, description: str) -> None:
    shortcut_path.parent.mkdir(parents=True, exist_ok=True)
    script = "\n".join(
        [
            "$shell = New-Object -ComObject WScript.Shell",
            f"$shortcut = $shell.CreateShortcut({ps_quote(str(shortcut_path))})",
            f"$shortcut.TargetPath = {ps_quote(str(target_path))}",
            f"$shortcut.WorkingDirectory = {ps_quote(str(target_path.parent))}",
            f"$shortcut.IconLocation = {ps_quote(str(target_path))}",
            f"$shortcut.Description = {ps_quote(description)}",
            "$shortcut.Save()",
        ]
    )
    subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-WindowStyle",
            "Hidden",
            "-Command",
            script,
        ],
        check=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


class InstallerApp:
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("Data Digitizer 2.11 Installer")
        self.root.geometry("520x230")
        self.root.resizable(False, False)
        self.status = StringVar(value="Ready to install Data Digitizer 2.11.")
        self.detail = StringVar(value="This installs Digitizer, AccuracyTester, and the CLI.")

        outer = Frame(self.root, padx=18, pady=16)
        outer.pack(fill=BOTH, expand=True)
        Label(outer, text="Data Digitizer 2.11", font=("Segoe UI", 16, "bold")).pack(anchor="w")
        Label(outer, textvariable=self.status, font=("Segoe UI", 10)).pack(anchor="w", pady=(12, 0))
        Label(outer, textvariable=self.detail, font=("Segoe UI", 9), fg="#555555").pack(anchor="w", pady=(4, 10))

        self.progress = ttk.Progressbar(outer, orient="horizontal", mode="determinate", maximum=100)
        self.progress.pack(fill=X, pady=(4, 16))

        button_row = Frame(outer)
        button_row.pack(side=BOTTOM, fill=X)
        self.install_button = Button(button_row, text="Install", width=14, command=self.start_install)
        self.install_button.pack(side=LEFT)
        self.open_button = Button(button_row, text="Open Digitizer", width=16, command=self.open_digitizer, state=DISABLED)
        self.open_button.pack(side=LEFT, padx=(8, 0))
        self.close_button = Button(button_row, text="Close", width=12, command=self.root.destroy)
        self.close_button.pack(side=RIGHT)

    def set_progress(self, value: float, status: str | None = None, detail: str | None = None) -> None:
        def update() -> None:
            self.progress["value"] = max(0, min(100, value))
            if status is not None:
                self.status.set(status)
            if detail is not None:
                self.detail.set(detail)

        self.root.after(0, update)

    def start_install(self) -> None:
        self.install_button.configure(state=DISABLED)
        self.open_button.configure(state=DISABLED)
        threading.Thread(target=self.install, daemon=True).start()

    def install(self) -> None:
        try:
            INSTALL_DIR.mkdir(parents=True, exist_ok=True)
            START_MENU_DIR.mkdir(parents=True, exist_ok=True)
            temp_download = Path(tempfile.gettempdir()) / "DataDigitizer-2.11-Installer"
            total_apps = len(APPS)

            for index, app in enumerate(APPS):
                file_name = app["file"]
                source = find_app_exe(file_name)
                base = index * 70 / total_apps
                span = 70 / total_apps
                if source is None:
                    self.set_progress(base, f"Downloading {file_name}...", "Using the official GitHub release.")
                    source = download_with_progress(
                        file_name,
                        temp_download / file_name,
                        lambda done, total, b=base, s=span: self.set_progress(b + (done / total) * s if total else b),
                    )
                else:
                    self.set_progress(base, f"Found {file_name}.", str(source))

                destination = INSTALL_DIR / file_name
                self.set_progress(base, f"Installing {file_name}...", str(destination))
                copy_with_progress(
                    source,
                    destination,
                    lambda done, total, b=base, s=span: self.set_progress(b + (done / total) * s if total else b),
                )

            for index, app in enumerate(APPS):
                target = INSTALL_DIR / app["file"]
                self.set_progress(72 + index * 8, f"Creating shortcut: {app['description']}...", str(target))
                create_shortcut(DESKTOP_DIR / app["shortcut"], target, app["description"])
                create_shortcut(START_MENU_DIR / app["shortcut"], target, app["description"])

            self.set_progress(100, "Data Digitizer 2.11 installed successfully.", f"Installed to {INSTALL_DIR}")
            self.root.after(0, lambda: self.open_button.configure(state=NORMAL))
            self.root.after(0, lambda: messagebox.showinfo("Install complete", "Desktop shortcuts were created for Digitizer, AccuracyTester, and DataDigitizer CLI."))
        except Exception as exc:
            self.set_progress(0, "Install failed.", str(exc))
            self.root.after(0, lambda: self.install_button.configure(state=NORMAL))
            self.root.after(0, lambda: messagebox.showerror("Install failed", str(exc)))

    def open_digitizer(self) -> None:
        digitizer = INSTALL_DIR / "digitizer.exe"
        if not digitizer.exists():
            messagebox.showerror("Missing app", f"Could not find {digitizer}")
            return
        subprocess.Popen([str(digitizer)], cwd=str(INSTALL_DIR), creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

    def run(self) -> None:
        self.root.mainloop()


def main() -> int:
    InstallerApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

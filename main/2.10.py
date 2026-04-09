from __future__ import annotations

import sys

from PyQt6 import QtWidgets

from UI import DigitizerWindow
from ErrorLogger import log_exception


def _prompt_action() -> str:
    banner = [
        "========================================",
        " Data Digitizer 2.10",
        "----------------------------------------",
        " GitHub: https://github.com/aj24by7/DataDigitizer",
        "========================================",
        " 1) Launch Data Digitizer (PyQt)",
        " 2) Launch Accuracy Tester Pro",
        " 3) Launch Click Test Software",
        " 4) Exit",
    ]
    print("\n".join(banner))
    while True:
        choice = input("Select option [1-4] (y=1, n=4): ").strip().lower()
        if choice in {"1", "2", "3", "4"}:
            return choice
        if choice in {"y", "yes"}:
            return "1"
        if choice in {"n", "no", "q", "quit", "exit"}:
            return "4"
        print("Please enter 1, 2, 3, or 4.")


def _launch_digitizer() -> None:
    def _excepthook(exc_type, exc_value, exc_tb):
        try:
            log_exception("Unhandled exception", exc_value)
        finally:
            sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook
    app = QtWidgets.QApplication([])
    window = DigitizerWindow()
    window.show()
    app.exec()


def _launch_accuracy_tester_pro() -> None:
    try:
        from AccuracyTesterPro import main as accuracy_tester_pro_main
    except ModuleNotFoundError as exc:
        print(f"Cannot launch Accuracy Tester Pro: missing dependency or module ({exc.name}).")
        print("Install tester dependencies: pip install pandas numpy matplotlib")
        print("Optional drag-and-drop support: pip install tkinterdnd2")
        return

    accuracy_tester_pro_main()

def _launch_click_test_software() -> None:
    try:
        from ClickTestSoftware import main as click_test_main
    except ModuleNotFoundError as exc:
        print(f"Cannot launch Click Test Software: missing dependency or module ({exc.name}).")
        print("Install GUI dependencies: pip install PyQt6")
        return

    click_test_main()


def main() -> None:
    while True:
        try:
            action = _prompt_action()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. See you next time.")
            return

        if action == "4":
            print("Exiting. See you next time.")
            return
        if action == "1":
            _launch_digitizer()
        elif action == "2":
            _launch_accuracy_tester_pro()
        elif action == "3":
            _launch_click_test_software()

        print()
        print("Returned to launcher.")
        print()


if __name__ == "__main__":
    main()

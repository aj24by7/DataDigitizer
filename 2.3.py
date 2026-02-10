#!/usr/bin/env python3

from PyQt6 import QtWidgets

from UI import DigitizerWindow


def _prompt_launch() -> bool:
    banner = [
        "========================================",
        " Data Digitizer 2.3",
        "----------------------------------------",
        " GitHub: https://github.com/aj24by7/DataDigitizer",
        "========================================",
    ]
    print("\n".join(banner))
    while True:
        choice = input("Run program? [y/n]: ").strip().lower()
        if choice in {"y", "yes"}:
            return True
        if choice in {"n", "no"}:
            return False
        print("Please enter y or n.")


def main() -> None:
    if not _prompt_launch():
        print("Exiting. See you next time.")
        return
    app = QtWidgets.QApplication([])
    window = DigitizerWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

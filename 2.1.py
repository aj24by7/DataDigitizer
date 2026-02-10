#!/usr/bin/env python3

from PyQt6 import QtWidgets

from UI import DigitizerWindow


def main() -> None:
    app = QtWidgets.QApplication([])
    window = DigitizerWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()

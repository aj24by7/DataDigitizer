from __future__ import annotations

from typing import Optional, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from ImageTray import ImageTray
from PointPlacer import PlacedPoint, find_points_by_color, interpolate_points


class DigitizerWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Data Digitizer")
        self.resize(1100, 750)

        self._image: Optional[QtGui.QImage] = None
        self._selected_color: Optional[Tuple[int, int, int]] = None
        self._pick_color_mode = False
        self._base_points: list[PlacedPoint] = []
        self._points: list[PlacedPoint] = []
        self._interpolate_enabled = False
        self._chroma_filter_enabled = False
        self._chroma_min = 20

        self._build_ui()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        central.setAutoFillBackground(True)
        palette = central.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(255, 255, 255))
        central.setPalette(palette)
        self.setCentralWidget(central)

        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.setSpacing(6)

        self.import_button = QtWidgets.QToolButton()
        self.import_button.setText("Import")
        self.import_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.import_menu = QtWidgets.QMenu(self)
        self.act_import_image = self.import_menu.addAction("Import Image")
        self.act_paste_image = self.import_menu.addAction("Paste Image")
        self.import_button.setMenu(self.import_menu)
        top_bar.addWidget(self.import_button)

        self.tools_button = QtWidgets.QToolButton()
        self.tools_button.setText("Tools")
        self.tools_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.tools_menu = QtWidgets.QMenu(self)
        self.act_place_points = self.tools_menu.addAction("Place Points")
        self.act_interpolate = self.tools_menu.addAction("Interpolation")
        self.act_interpolate.setCheckable(True)
        self.act_pick_color = self.tools_menu.addAction("Pick Color")
        self.act_pick_color.setCheckable(True)
        self.tools_button.setMenu(self.tools_menu)
        top_bar.addWidget(self.tools_button)

        self.filters_button = QtWidgets.QToolButton()
        self.filters_button.setText("Advanced")
        self.filters_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.filters_menu = QtWidgets.QMenu(self)
        self.act_chroma_filter = self.filters_menu.addAction("Chroma Filter")
        self.act_chroma_filter.setCheckable(True)
        self.filters_button.setMenu(self.filters_menu)
        top_bar.addWidget(self.filters_button)

        top_bar.addStretch(1)
        layout.addLayout(top_bar)

        self.image_tray = ImageTray()
        layout.addWidget(self.image_tray, 1)

        self.status_label = QtWidgets.QLabel("Load an image to begin.")
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.status_label)

        self.color_row = QtWidgets.QHBoxLayout()
        self.color_row.setSpacing(6)
        self.selected_color_label = QtWidgets.QLabel("Selected color")
        self.selected_color_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self.color_swatch = QtWidgets.QLabel()
        swatch_size = self.selected_color_label.sizeHint().height()
        self.color_swatch.setFixedSize(swatch_size, swatch_size)
        self.color_swatch.setStyleSheet("border: 1px solid black; background: white;")
        self.color_row.addWidget(self.selected_color_label)
        self.color_row.addWidget(self.color_swatch)
        self.color_row.addStretch(1)
        layout.addLayout(self.color_row)

        self.act_import_image.triggered.connect(self.open_image_dialog)
        self.act_paste_image.triggered.connect(self.paste_image)
        self.act_place_points.triggered.connect(self.place_points)
        self.act_interpolate.toggled.connect(self.toggle_interpolation)
        self.act_pick_color.toggled.connect(self.toggle_pick_color)
        self.act_chroma_filter.toggled.connect(self.toggle_chroma_filter)
        self.image_tray.image_clicked.connect(self.on_image_clicked)

    def open_image_dialog(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if not path:
            return
        image = QtGui.QImage(path)
        if image.isNull():
            self.status_label.setText("Failed to load image.")
            return
        self.set_image(image)

    def paste_image(self) -> None:
        clipboard = QtWidgets.QApplication.clipboard()
        image = clipboard.image()
        if image.isNull():
            pixmap = clipboard.pixmap()
            if not pixmap.isNull():
                image = pixmap.toImage()
        if image.isNull():
            self.status_label.setText("Clipboard has no image.")
            return
        self.set_image(image)

    def set_image(self, image: QtGui.QImage) -> None:
        self._image = image
        self._base_points = []
        self._points = []
        self.image_tray.set_image(image)
        self.image_tray.set_points([])
        self._update_color_swatch(QtGui.QColor(255, 255, 255))
        self.status_label.setText("Image loaded. Pick a color to start.")

    def toggle_pick_color(self, checked: bool) -> None:
        self._pick_color_mode = checked
        if checked:
            self.image_tray.setCursor(QtCore.Qt.CursorShape.CrossCursor)
            self.status_label.setText("Pick color: click on the image.")
        else:
            self.image_tray.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            if self._selected_color:
                self.status_label.setText(f"Selected color: {self._selected_color}")
            else:
                self.status_label.setText("Pick a color from the image.")

    def on_image_clicked(self, x: int, y: int) -> None:
        if self._image is None:
            return
        if self._pick_color_mode:
            color = self._image.pixelColor(x, y)
            self._selected_color = (color.red(), color.green(), color.blue())
            self._update_color_swatch(color)
            self.act_pick_color.setChecked(False)
            self.status_label.setText(f"Selected color: {self._selected_color}")

    def place_points(self) -> None:
        if self._image is None:
            self.status_label.setText("Load an image first.")
            return
        if self._selected_color is None:
            self.status_label.setText("Pick a color first.")
            return
        self._compute_points()
        suffix = " (interpolated)" if self._interpolate_enabled else ""
        self.status_label.setText(f"Placed {len(self._points)} points{suffix}.")

    def toggle_interpolation(self, checked: bool) -> None:
        self._interpolate_enabled = checked
        if self._image is None or self._selected_color is None:
            state = "on" if checked else "off"
            self.status_label.setText(f"Interpolation {state}. Load an image and pick a color.")
            return
        self._compute_points()
        state = "on" if checked else "off"
        self.status_label.setText(f"Interpolation {state}. Points: {len(self._points)}.")

    def toggle_chroma_filter(self, checked: bool) -> None:
        self._chroma_filter_enabled = checked
        if self._image is None or self._selected_color is None:
            state = "on" if checked else "off"
            self.status_label.setText(f"Chroma filter {state}. Load an image and pick a color.")
            return
        self._compute_points()
        state = "on" if checked else "off"
        self.status_label.setText(f"Chroma filter {state}. Points: {len(self._points)}.")

    def _apply_interpolation(self, points: list[PlacedPoint]) -> list[PlacedPoint]:
        if self._interpolate_enabled:
            return interpolate_points(points, segment_len=5.0, points_per_segment=3)
        return list(points)

    def _compute_points(self) -> None:
        min_chroma = self._chroma_min if self._chroma_filter_enabled else None
        self._base_points = find_points_by_color(
            self._image,
            self._selected_color,
            min_chroma=min_chroma,
        )
        self._points = self._apply_interpolation(self._base_points)
        self.image_tray.set_points([(pt.x, pt.y) for pt in self._points])

    def _update_color_swatch(self, color: QtGui.QColor) -> None:
        self.color_swatch.setStyleSheet(
            f"border: 1px solid black; background: rgb({color.red()}, {color.green()}, {color.blue()});"
        )

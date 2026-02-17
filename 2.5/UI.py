from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from AxisReader import AxisDetectionError, AxisScaleResult, detect_axis_scale
from Calibration import CalibrationResult, coordinate_mediated_calibration, line_mediated_calibration
from ErrorLogger import get_csv_path, log_exception, read_recent_entries
from Masking import detect_legend_mask, detect_number_masks, detect_word_masks
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
        self._palette_summary: list[Tuple[Tuple[int, int, int], float]] = []
        self._axis_result: Optional[AxisScaleResult] = None
        self._calibration_result: Optional[CalibrationResult] = None
        self._calibration_mode: Optional[str] = None
        self._manual_stage = 0
        self._limit_points_to_calib = False
        self._mask_rects: dict[str, list[Tuple[float, float, float, float]]] = {
            "words": [],
            "numbers": [],
            "legend": [],
            "manual": [],
        }
        self._mask_mode = False
        self._manual_points: dict[str, Optional[Tuple[int, int]]] = {
            "x_min": None,
            "x_max": None,
            "y_min": None,
            "y_max": None,
        }

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
        self.place_menu = QtWidgets.QMenu("Place Points", self)
        self.act_place_points_run = self.place_menu.addAction("Run")
        self.act_place_points_limit = self.place_menu.addAction("Limit to Calibration Window")
        self.act_place_points_limit.setCheckable(True)
        self.act_place_points_limit.setChecked(False)
        self.tools_menu.addMenu(self.place_menu)
        self.calibration_menu = QtWidgets.QMenu("Calibration", self)
        self.act_calib_manual = self.calibration_menu.addAction("Manual Calibration")
        self.act_calib_coord = self.calibration_menu.addAction("Coordinate-Mediated Calibration")
        self.act_calib_line = self.calibration_menu.addAction("Line-Mediated Calibration")
        self.tools_menu.addMenu(self.calibration_menu)
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
        self.axis_menu = QtWidgets.QMenu("Axis Scale Detection", self)
        self.act_axis_run = self.axis_menu.addAction("Run Axis Detection")
        self.axis_menu.addSeparator()
        self.act_axis_x_min = self.axis_menu.addAction("X min: --")
        self.act_axis_x_max = self.axis_menu.addAction("X max: --")
        self.act_axis_y_min = self.axis_menu.addAction("Y min: --")
        self.act_axis_y_max = self.axis_menu.addAction("Y max: --")
        for action in (self.act_axis_x_min, self.act_axis_x_max, self.act_axis_y_min, self.act_axis_y_max):
            action.setEnabled(False)
        self.filters_menu.addMenu(self.axis_menu)
        self.mask_menu = QtWidgets.QMenu("Masking", self)
        self.act_mask_words = self.mask_menu.addAction("Mask Words")
        self.act_mask_numbers = self.mask_menu.addAction("Mask Numbers")
        self.act_mask_legend = self.mask_menu.addAction("Mask Legend")
        self.mask_menu.addSeparator()
        self.act_mask_manual = self.mask_menu.addAction("Manual Mask Mode")
        self.act_mask_manual.setCheckable(True)
        self.act_mask_clear = self.mask_menu.addAction("Clear Masks")
        self.filters_menu.addMenu(self.mask_menu)
        self.filters_menu.addSeparator()
        self.act_error_log = self.filters_menu.addAction("Error Log")
        self.filters_button.setMenu(self.filters_menu)
        top_bar.addWidget(self.filters_button)

        self.export_button = QtWidgets.QToolButton()
        self.export_button.setText("Export")
        self.export_button.setPopupMode(QtWidgets.QToolButton.ToolButtonPopupMode.InstantPopup)
        self.export_menu = QtWidgets.QMenu(self)
        self.export_csv_menu = QtWidgets.QMenu("Export CSV", self)
        self.act_export_csv_raw = self.export_csv_menu.addAction("Raw values")
        self.act_export_csv_norm = self.export_csv_menu.addAction("Y normalized (0-1)")
        self.export_menu.addMenu(self.export_csv_menu)
        self.export_excel_menu = QtWidgets.QMenu("Export Excel", self)
        self.act_export_excel_raw = self.export_excel_menu.addAction("Raw values")
        self.act_export_excel_norm = self.export_excel_menu.addAction("Y normalized (0-1)")
        self.export_menu.addMenu(self.export_excel_menu)
        self.export_button.setMenu(self.export_menu)
        top_bar.addWidget(self.export_button)

        top_bar.addStretch(1)
        layout.addLayout(top_bar)

        self.image_tray = ImageTray()
        layout.addWidget(self.image_tray, 1)

        self.coord_group = QtWidgets.QGroupBox("Min-Max Coordinates")
        coord_layout = QtWidgets.QGridLayout(self.coord_group)
        coord_layout.setContentsMargins(8, 6, 8, 6)
        coord_layout.setHorizontalSpacing(10)
        coord_layout.addWidget(QtWidgets.QLabel("X min"), 0, 0)
        coord_layout.addWidget(QtWidgets.QLabel("X max"), 0, 2)
        coord_layout.addWidget(QtWidgets.QLabel("Y min"), 1, 0)
        coord_layout.addWidget(QtWidgets.QLabel("Y max"), 1, 2)
        self.coord_x_min = QtWidgets.QLineEdit()
        self.coord_x_max = QtWidgets.QLineEdit()
        self.coord_y_min = QtWidgets.QLineEdit()
        self.coord_y_max = QtWidgets.QLineEdit()
        for field in (self.coord_x_min, self.coord_x_max, self.coord_y_min, self.coord_y_max):
            field.setMaximumWidth(120)
        coord_layout.addWidget(self.coord_x_min, 0, 1)
        coord_layout.addWidget(self.coord_x_max, 0, 3)
        coord_layout.addWidget(self.coord_y_min, 1, 1)
        coord_layout.addWidget(self.coord_y_max, 1, 3)
        coord_layout.setColumnStretch(4, 1)
        layout.addWidget(self.coord_group)

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
        self.act_place_points_run.triggered.connect(self.place_points)
        self.act_place_points_limit.toggled.connect(self.toggle_limit_points)
        self.act_axis_run.triggered.connect(self.run_axis_detection)
        self.act_calib_manual.triggered.connect(self.start_manual_calibration)
        self.act_calib_coord.triggered.connect(self.run_coordinate_calibration)
        self.act_calib_line.triggered.connect(self.run_line_calibration)
        self.act_mask_words.triggered.connect(self.run_mask_words)
        self.act_mask_numbers.triggered.connect(self.run_mask_numbers)
        self.act_mask_legend.triggered.connect(self.run_mask_legend)
        self.act_mask_manual.toggled.connect(self.toggle_manual_mask)
        self.act_mask_clear.triggered.connect(self.clear_masks)
        self.act_error_log.triggered.connect(self.show_error_log)
        self.act_export_csv_raw.triggered.connect(lambda: self.export_points("csv", normalize_y=False))
        self.act_export_csv_norm.triggered.connect(lambda: self.export_points("csv", normalize_y=True))
        self.act_export_excel_raw.triggered.connect(lambda: self.export_points("excel", normalize_y=False))
        self.act_export_excel_norm.triggered.connect(lambda: self.export_points("excel", normalize_y=True))
        self.act_interpolate.toggled.connect(self.toggle_interpolation)
        self.act_pick_color.toggled.connect(self.toggle_pick_color)
        self.act_chroma_filter.toggled.connect(self.toggle_chroma_filter)
        self.image_tray.image_clicked.connect(self.on_image_clicked)
        self.image_tray.mask_rect_created.connect(self.on_mask_rect_created)
        self.image_tray.mask_remove_requested.connect(self.on_mask_remove_requested)

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
        self.image_tray.set_axis_overlays([], [])
        self.image_tray.set_calibration_overlays([], None)
        self.image_tray.set_mask_overlays([])
        self._axis_result = None
        self._update_axis_menu(None)
        self._reset_calibration_state()
        self._update_coord_fields(None)
        for key in self._mask_rects:
            self._mask_rects[key] = []
        self._mask_mode = False
        self.act_mask_manual.setChecked(False)
        self.image_tray.set_mask_draw_enabled(False)
        auto_color = self._auto_select_color(image)
        if auto_color:
            self._selected_color = auto_color
            self._update_color_swatch(QtGui.QColor(*auto_color))
            self.status_label.setText(f"Image loaded. Auto-selected color: {auto_color}")
        else:
            self._selected_color = None
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
        if self._calibration_mode == "manual":
            self._handle_manual_calibration_click(x, y)
            return
        if self._pick_color_mode:
            color = self._image.pixelColor(x, y)
            self._selected_color = (color.red(), color.green(), color.blue())
            self._update_color_swatch(color)
            self.act_pick_color.setChecked(False)
            self.status_label.setText(f"Selected color: {self._selected_color}")

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key.Key_Escape and self._mask_mode:
            self.act_mask_manual.setChecked(False)
            event.accept()
            return
        super().keyPressEvent(event)

    def place_points(self) -> None:
        if self._image is None:
            self.status_label.setText("Load an image first.")
            return
        if self._selected_color is None:
            self.status_label.setText("Pick a color first.")
            return
        self._compute_points()
        suffix = " (interpolated)" if self._interpolate_enabled else ""
        limit_suffix = " (limited to calibration)" if self._limit_points_to_calib else ""
        self.status_label.setText(f"Placed {len(self._points)} points{suffix}{limit_suffix}.")

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

    def toggle_limit_points(self, checked: bool) -> None:
        self._limit_points_to_calib = checked
        if self._image is None or self._selected_color is None:
            state = "on" if checked else "off"
            self.status_label.setText(f"Limit to calibration {state}.")
            return
        self._compute_points()
        state = "on" if checked else "off"
        self.status_label.setText(f"Limit to calibration {state}. Points: {len(self._points)}.")

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
            exclude_rects=self._all_mask_rects(),
        )
        points = self._base_points
        if self._limit_points_to_calib:
            box = self._get_calibration_box()
            if box is not None:
                left, right, top, bottom = box
                points = [pt for pt in points if left <= pt.x <= right and top <= pt.y <= bottom]
        self._points = self._apply_interpolation(points)
        self.image_tray.set_points([(pt.x, pt.y) for pt in self._points])

    def run_mask_words(self) -> None:
        if self._image is None:
            self.status_label.setText("Load an image first.")
            return
        try:
            rects = detect_word_masks(self._image)
        except Exception as exc:
            log_exception("Mask words", exc)
            self.status_label.setText("Mask words failed. See error log.")
            return
        self._set_mask_category("words", rects)
        self.status_label.setText("Masked words.")

    def run_mask_numbers(self) -> None:
        if self._image is None:
            self.status_label.setText("Load an image first.")
            return
        try:
            rects = detect_number_masks(self._image)
        except Exception as exc:
            log_exception("Mask numbers", exc)
            self.status_label.setText("Mask numbers failed. See error log.")
            return
        self._set_mask_category("numbers", rects)
        self.status_label.setText("Masked numbers.")

    def run_mask_legend(self) -> None:
        if self._image is None:
            self.status_label.setText("Load an image first.")
            return
        try:
            rects = detect_legend_mask(self._image)
        except Exception as exc:
            log_exception("Mask legend", exc)
            self.status_label.setText("Mask legend failed. See error log.")
            return
        if not rects:
            self.status_label.setText("Legend mask not found.")
            self._set_mask_category("legend", [])
            return
        self._set_mask_category("legend", rects)
        self.status_label.setText("Masked legend.")

    def toggle_manual_mask(self, checked: bool) -> None:
        self._mask_mode = checked
        self.image_tray.set_mask_draw_enabled(checked)
        if checked:
            if self._pick_color_mode:
                self.act_pick_color.setChecked(False)
                self._pick_color_mode = False
                self.image_tray.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            self.status_label.setText("Manual mask mode: drag to add, Shift+click to remove, Esc to exit.")
        else:
            self.status_label.setText("Manual mask mode off.")

    def clear_masks(self) -> None:
        for key in self._mask_rects:
            self._mask_rects[key] = []
        self.image_tray.set_mask_overlays([])
        self.status_label.setText("Masks cleared.")

    def on_mask_rect_created(self, x: float, y: float, w: float, h: float) -> None:
        rect = (x, y, w, h)
        self._mask_rects["manual"].append(rect)
        self.image_tray.set_mask_overlays(self._all_mask_rects())
        self.status_label.setText("Mask added.")

    def on_mask_remove_requested(self, x: int, y: int) -> None:
        for key, rects in self._mask_rects.items():
            for idx, (left, top, w, h) in enumerate(rects):
                if left <= x <= left + w and top <= y <= top + h:
                    rects.pop(idx)
                    self.image_tray.set_mask_overlays(self._all_mask_rects())
                    self.status_label.setText("Mask removed.")
                    return

    def show_error_log(self) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Error Log")
        dialog.resize(700, 400)
        layout = QtWidgets.QVBoxLayout(dialog)
        text = QtWidgets.QTextEdit()
        text.setReadOnly(True)
        text.setText(read_recent_entries(200))
        layout.addWidget(text)
        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        open_csv = QtWidgets.QPushButton("Open CSV")
        open_csv.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(get_csv_path()))))
        button_row.addWidget(open_csv)
        close_btn = QtWidgets.QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        button_row.addWidget(close_btn)
        layout.addLayout(button_row)
        dialog.exec()

    def export_points(self, kind: str, normalize_y: bool) -> None:
        if self._image is None:
            self.status_label.setText("Load an image first.")
            return
        if not self._points:
            if self._selected_color is None:
                self.status_label.setText("Pick a color and place points first.")
                return
            self._compute_points()
        if not self._points:
            self.status_label.setText("No points available to export.")
            return

        axis_values = self._get_axis_values()
        if axis_values is None:
            self.status_label.setText("Export requires X/Y min/max values.")
            return
        calibration_box = self._get_calibration_box()
        mapper = self._build_affine_mapper(axis_values)
        if calibration_box is None and mapper is None:
            self.status_label.setText("Export requires calibration. Run calibration first.")
            return

        x_min_val, x_max_val, y_min_val, y_max_val = axis_values
        if calibration_box is not None:
            left, right, top, bottom = calibration_box
            if right <= left or bottom <= top:
                self.status_label.setText("Calibration box is invalid. Re-run calibration.")
                return

        rows = []
        for pt in self._points:
            x_px = pt.x
            y_px = pt.y
            if mapper is not None:
                x_val, y_val = self._map_pixel_affine(mapper, x_px, y_px)
            else:
                x_val = x_min_val + (x_px - left) * (x_max_val - x_min_val) / (right - left)
                y_val = y_min_val + (bottom - y_px) * (y_max_val - y_min_val) / (bottom - top)
            row = [x_val, y_val, x_px, y_px]
            if normalize_y:
                denom = y_max_val - y_min_val
                y_norm = (y_val - y_min_val) / denom if denom != 0 else 0.0
                row.append(y_norm)
            rows.append(row)

        if not rows:
            self.status_label.setText("No points available to export.")
            return

        headers = ["x", "y", "x_px", "y_px"]
        if normalize_y:
            headers.append("y_norm")

        saved = False
        if kind == "csv":
            saved = self._export_csv(headers, rows)
        else:
            saved = self._export_excel(headers, rows)
        if not saved:
            return
        suffix = " (normalized)" if normalize_y else ""
        msg = f"Exported {len(rows)} points{suffix}."
        self.status_label.setText(msg)

    def _export_csv(self, headers: list[str], rows: list[list[float]]) -> bool:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export CSV",
            "digitized_points.csv",
            "CSV Files (*.csv)",
        )
        if not path:
            return False
        file_path = Path(path)
        if file_path.suffix.lower() != ".csv":
            file_path = file_path.with_suffix(".csv")
        with file_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            for row in rows:
                writer.writerow([self._fmt_cell(value) for value in row])
        return True

    def _export_excel(self, headers: list[str], rows: list[list[float]]) -> bool:
        try:
            import openpyxl
        except ImportError:
            self.status_label.setText("Excel export requires openpyxl. Install with pip.")
            return False
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Excel",
            "digitized_points.xlsx",
            "Excel Files (*.xlsx)",
        )
        if not path:
            return False
        file_path = Path(path)
        if file_path.suffix.lower() != ".xlsx":
            file_path = file_path.with_suffix(".xlsx")
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "Points"
        sheet.append(headers)
        for row in rows:
            sheet.append([self._fmt_cell(value) for value in row])
        workbook.save(file_path)
        return True

    def _fmt_cell(self, value: float) -> float:
        if isinstance(value, float):
            return round(value, 6)
        return value

    def _get_axis_values(self) -> Optional[Tuple[float, float, float, float]]:
        try:
            x_min_val = float(self.coord_x_min.text())
            x_max_val = float(self.coord_x_max.text())
            y_min_val = float(self.coord_y_min.text())
            y_max_val = float(self.coord_y_max.text())
        except ValueError:
            return None
        return x_min_val, x_max_val, y_min_val, y_max_val

    def _get_calibration_box(self) -> Optional[Tuple[int, int, int, int]]:
        box = None
        if self._calibration_result and self._calibration_result.box:
            box = self._calibration_result.box
        elif all(self._manual_points.values()):
            x_min = self._manual_points["x_min"]
            x_max = self._manual_points["x_max"]
            y_min = self._manual_points["y_min"]
            y_max = self._manual_points["y_max"]
            if x_min and x_max and y_min and y_max:
                left = y_min[0]
                right = x_max[0]
                bottom = x_min[1]
                top = y_max[1]
                box = [(left, top), (right, top), (right, bottom), (left, bottom)]
        if not box:
            return None
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))

    def _get_axis_points(
        self,
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]]:
        if self._calibration_result:
            x_min = self._calibration_result.x_min_point
            x_max = self._calibration_result.x_max_point
            y_min = self._calibration_result.y_min_point
            y_max = self._calibration_result.y_max_point
            if x_min and x_max and y_min and y_max:
                return x_min, x_max, y_min, y_max
        if all(self._manual_points.values()):
            x_min = self._manual_points["x_min"]
            x_max = self._manual_points["x_max"]
            y_min = self._manual_points["y_min"]
            y_max = self._manual_points["y_max"]
            if x_min and x_max and y_min and y_max:
                return (
                    (float(x_min[0]), float(x_min[1])),
                    (float(x_max[0]), float(x_max[1])),
                    (float(y_min[0]), float(y_min[1])),
                    (float(y_max[0]), float(y_max[1])),
                )
        return None

    def _build_affine_mapper(
        self,
        axis_values: Tuple[float, float, float, float],
    ) -> Optional[
        Tuple[
            Tuple[float, float],
            Tuple[float, float],
            Tuple[float, float],
            float,
            float,
            float,
            float,
            float,
            float,
            float,
        ]
    ]:
        points = self._get_axis_points()
        if points is None:
            return None
        x_min_pt, x_max_pt, y_min_pt, y_max_pt = points
        origin = _line_intersection(x_min_pt, x_max_pt, y_min_pt, y_max_pt)
        if origin is None:
            return None
        x_vec = (x_max_pt[0] - x_min_pt[0], x_max_pt[1] - x_min_pt[1])
        y_vec = (y_max_pt[0] - y_min_pt[0], y_max_pt[1] - y_min_pt[1])
        det = x_vec[0] * y_vec[1] - x_vec[1] * y_vec[0]
        if abs(det) < 1e-6:
            return None

        def solve_uv(point: Tuple[float, float]) -> Tuple[float, float]:
            dx = point[0] - origin[0]
            dy = point[1] - origin[1]
            u = (dx * y_vec[1] - dy * y_vec[0]) / det
            v = (x_vec[0] * dy - x_vec[1] * dx) / det
            return u, v

        u_min, _ = solve_uv(x_min_pt)
        u_max, _ = solve_uv(x_max_pt)
        _, v_min = solve_uv(y_min_pt)
        _, v_max = solve_uv(y_max_pt)

        if abs(u_max - u_min) < 1e-6 or abs(v_max - v_min) < 1e-6:
            return None

        x_min_val, x_max_val, y_min_val, y_max_val = axis_values
        x_scale = (x_max_val - x_min_val) / (u_max - u_min)
        y_scale = (y_max_val - y_min_val) / (v_max - v_min)
        return (
            origin,
            x_vec,
            y_vec,
            det,
            x_min_val,
            y_min_val,
            u_min,
            v_min,
            x_scale,
            y_scale,
        )

    def _map_pixel_affine(
        self,
        mapper: Tuple[
            Tuple[float, float],
            Tuple[float, float],
            Tuple[float, float],
            float,
            float,
            float,
            float,
            float,
            float,
            float,
        ],
        x_px: float,
        y_px: float,
    ) -> Tuple[float, float]:
        origin, x_vec, y_vec, det, x_min_val, y_min_val, u_min, v_min, x_scale, y_scale = mapper
        dx = x_px - origin[0]
        dy = y_px - origin[1]
        u = (dx * y_vec[1] - dy * y_vec[0]) / det
        v = (x_vec[0] * dy - x_vec[1] * dx) / det
        x_val = x_min_val + (u - u_min) * x_scale
        y_val = y_min_val + (v - v_min) * y_scale
        return x_val, y_val

    def _all_mask_rects(self) -> list[Tuple[float, float, float, float]]:
        rects: list[Tuple[float, float, float, float]] = []
        for items in self._mask_rects.values():
            rects.extend(items)
        return rects

    def _set_mask_category(self, category: str, rects: list[Tuple[float, float, float, float]]) -> None:
        self._mask_rects[category] = rects
        self.image_tray.set_mask_overlays(self._all_mask_rects())


    def _update_color_swatch(self, color: QtGui.QColor) -> None:
        self.color_swatch.setStyleSheet(
            f"border: 1px solid black; background: rgb({color.red()}, {color.green()}, {color.blue()});"
        )

    def run_axis_detection(self) -> None:
        if self._image is None:
            self.status_label.setText("Load an image first.")
            return
        try:
            result = detect_axis_scale(self._image)
        except AxisDetectionError as exc:
            log_exception("Axis detection", exc)
            self.status_label.setText(str(exc))
            self.image_tray.set_axis_overlays([], [])
            self._axis_result = None
            self._update_axis_menu(None)
            self._update_coord_fields(None)
            return
        except Exception as exc:
            log_exception("Axis detection (unexpected)", exc)
            self.status_label.setText("Axis detection failed. See error log.")
            self.image_tray.set_axis_overlays([], [])
            self._axis_result = None
            self._update_axis_menu(None)
            self._update_coord_fields(None)
            return
        self._axis_result = result
        self.image_tray.set_axis_overlays(result.overlay_points, result.overlay_rects)
        self._update_axis_menu(result)
        self._update_coord_fields(result)
        missing = []
        if result.x_min is None:
            missing.append("X min")
        if result.x_max is None:
            missing.append("X max")
        if result.y_min is None:
            missing.append("Y min")
        if result.y_max is None:
            missing.append("Y max")
        if missing:
            self.status_label.setText("Axis detection partial: missing " + ", ".join(missing) + ".")
        else:
            self.status_label.setText("Axis scale detected.")

    def _update_axis_menu(self, result: Optional[AxisScaleResult]) -> None:
        def fmt(value: Optional[float]) -> str:
            return "--" if value is None else str(value)

        if result is None:
            self.act_axis_x_min.setText("X min: --")
            self.act_axis_x_max.setText("X max: --")
            self.act_axis_y_min.setText("Y min: --")
            self.act_axis_y_max.setText("Y max: --")
            return
        self.act_axis_x_min.setText(f"X min: {fmt(result.x_min)}")
        self.act_axis_x_max.setText(f"X max: {fmt(result.x_max)}")
        self.act_axis_y_min.setText(f"Y min: {fmt(result.y_min)}")
        self.act_axis_y_max.setText(f"Y max: {fmt(result.y_max)}")

    def _update_coord_fields(self, result: Optional[AxisScaleResult]) -> None:
        def fmt(value: Optional[float]) -> str:
            return "" if value is None else str(value)

        if result is None:
            self.coord_x_min.setText("")
            self.coord_x_max.setText("")
            self.coord_y_min.setText("")
            self.coord_y_max.setText("")
            return
        self.coord_x_min.setText(fmt(result.x_min))
        self.coord_x_max.setText(fmt(result.x_max))
        self.coord_y_min.setText(fmt(result.y_min))
        self.coord_y_max.setText(fmt(result.y_max))

    def _reset_calibration_state(self) -> None:
        self._calibration_result = None
        self._calibration_mode = None
        self._manual_stage = 0
        for key in self._manual_points:
            self._manual_points[key] = None

    def start_manual_calibration(self) -> None:
        if self._image is None:
            self.status_label.setText("Load an image first.")
            return
        self._calibration_mode = "manual"
        self._manual_stage = 0
        for key in self._manual_points:
            self._manual_points[key] = None
        self.image_tray.set_calibration_overlays([], None)
        self._calibration_result = None
        if self._pick_color_mode:
            self.act_pick_color.setChecked(False)
            self._pick_color_mode = False
            self.image_tray.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
        self.status_label.setText(
            "Manual calibration: click X min, X max, Y min, Y max (first click per axis is min)."
        )

    def _handle_manual_calibration_click(self, x: int, y: int) -> None:
        if self._manual_stage == 0:
            self._manual_points["x_min"] = (x, y)
            self._manual_stage = 1
            self.status_label.setText("Manual calibration: click X max.")
        elif self._manual_stage == 1:
            self._manual_points["x_max"] = (x, y)
            self._manual_stage = 2
            self.status_label.setText("Manual calibration: click Y min.")
        elif self._manual_stage == 2:
            self._manual_points["y_min"] = (x, y)
            self._manual_stage = 3
            self.status_label.setText("Manual calibration: click Y max.")
        elif self._manual_stage == 3:
            self._manual_points["y_max"] = (x, y)
            self._manual_stage = 4
            self._calibration_mode = None
            self.status_label.setText("Manual calibration complete. Enter values above if needed.")
        points = [pt for pt in self._manual_points.values() if pt is not None]
        box = None
        if all(self._manual_points.values()):
            from Calibration import _box_from_axes

            box = _box_from_axes(
                self._manual_points["x_min"],
                self._manual_points["x_max"],
                self._manual_points["y_min"],
                self._manual_points["y_max"],
            )
        self.image_tray.set_calibration_overlays(points, box)

    def run_coordinate_calibration(self) -> None:
        if self._image is None:
            self.status_label.setText("Load an image first.")
            return
        if self._axis_result is None:
            self.status_label.setText("Run Axis Scale Detection first.")
            return
        try:
            result = coordinate_mediated_calibration(
                self._image,
                self._axis_result.x_min_point,
                self._axis_result.x_max_point,
                self._axis_result.y_min_point,
                self._axis_result.y_max_point,
                exclude_rects=self._axis_result.overlay_rects,
            )
        except Exception as exc:
            log_exception("Coordinate-mediated calibration", exc)
            self.status_label.setText("Coordinate calibration failed. See error log.")
            return
        self._calibration_result = result
        self.image_tray.set_calibration_overlays(result.points(), result.box)
        self.status_label.setText("Coordinate-mediated calibration complete.")

    def run_line_calibration(self) -> None:
        if self._image is None:
            self.status_label.setText("Load an image first.")
            return
        try:
            result = line_mediated_calibration(self._image)
        except Exception as exc:
            log_exception("Line-mediated calibration", exc)
            self.status_label.setText("Line calibration failed. See error log.")
            return
        if result.box is None:
            self.status_label.setText("Line-mediated calibration failed to find a black border.")
            return
        self._calibration_result = result
        self.image_tray.set_calibration_overlays(result.points(), result.box)
        self.status_label.setText("Line-mediated calibration complete.")

    def _auto_select_color(self, image: QtGui.QImage) -> Optional[Tuple[int, int, int]]:
        palette = self._analyze_palette(image)
        self._palette_summary = palette
        if not palette:
            return None
        for rgb, _pct in palette:
            if self._is_near_white(rgb) or self._is_near_black(rgb):
                continue
            return rgb
        return None

    def _analyze_palette(self, image: QtGui.QImage) -> list[Tuple[Tuple[int, int, int], float]]:
        if image.isNull():
            return []
        if image.format() != QtGui.QImage.Format.Format_RGB888:
            image = image.convertToFormat(QtGui.QImage.Format.Format_RGB888)

        width = image.width()
        height = image.height()
        bytes_per_line = image.bytesPerLine()
        bits = image.bits()
        bits.setsize(bytes_per_line * height)
        data = bytes(bits)

        target_samples = 200_000
        total_pixels = width * height
        stride = int((total_pixels / target_samples) ** 0.5) if total_pixels > target_samples else 1
        stride = max(1, stride)

        bucket_size = 16
        counts: dict[Tuple[int, int, int], int] = {}
        sums: dict[Tuple[int, int, int], Tuple[int, int, int]] = {}

        for y in range(0, height, stride):
            row = y * bytes_per_line
            for x in range(0, width, stride):
                idx = row + x * 3
                r = data[idx]
                g = data[idx + 1]
                b = data[idx + 2]
                key = (r // bucket_size, g // bucket_size, b // bucket_size)
                counts[key] = counts.get(key, 0) + 1
                if key in sums:
                    sr, sg, sb = sums[key]
                    sums[key] = (sr + r, sg + g, sb + b)
                else:
                    sums[key] = (r, g, b)

        total = sum(counts.values())
        if total == 0:
            return []

        palette: list[Tuple[Tuple[int, int, int], float]] = []
        for key, count in counts.items():
            sr, sg, sb = sums[key]
            avg = (int(sr / count), int(sg / count), int(sb / count))
            palette.append((avg, count / total))
        palette.sort(key=lambda item: item[1], reverse=True)
        return palette

    @staticmethod
    def _is_near_white(rgb: Tuple[int, int, int], threshold: int = 30) -> bool:
        return rgb[0] >= 255 - threshold and rgb[1] >= 255 - threshold and rgb[2] >= 255 - threshold

    @staticmethod
    def _is_near_black(rgb: Tuple[int, int, int], threshold: int = 30) -> bool:
        return rgb[0] <= threshold and rgb[1] <= threshold and rgb[2] <= threshold


def _line_intersection(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float],
    d: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    rx = b[0] - a[0]
    ry = b[1] - a[1]
    sx = d[0] - c[0]
    sy = d[1] - c[1]
    denom = rx * sy - ry * sx
    if abs(denom) < 1e-6:
        return None
    qpx = c[0] - a[0]
    qpy = c[1] - a[1]
    t = (qpx * sy - qpy * sx) / denom
    return (a[0] + t * rx, a[1] + t * ry)

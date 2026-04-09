from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from AxisReader import AxisDetectionError, AxisScaleResult, detect_axis_scale
from Calibration import CalibrationResult, coordinate_mediated_calibration, line_mediated_calibration
from ErrorLogger import get_csv_path, log_exception, read_recent_entries
from Masking import detect_legend_mask, detect_number_masks, detect_word_masks
from ImageTray import ImageTray
from PointPlacer import PlacedPoint, find_points_by_color, interpolate_points


class ZoomViewport(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._pixmap = QtGui.QPixmap()
        self.setFixedSize(280, 180)

    def set_pixmap(self, pixmap: QtGui.QPixmap) -> None:
        self._pixmap = pixmap
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.fillRect(rect, QtGui.QColor(22, 22, 22))

        if not self._pixmap.isNull():
            scaled = self._pixmap.scaled(
                rect.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            src_x = max(0, (scaled.width() - rect.width()) // 2)
            src_y = max(0, (scaled.height() - rect.height()) // 2)
            src = QtCore.QRect(src_x, src_y, rect.width(), rect.height())
            painter.drawPixmap(rect, scaled, src)

        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 180), 3))
        cx = rect.center().x()
        cy = rect.center().y()
        arm = 10
        painter.drawLine(cx - arm, cy, cx + arm, cy)
        painter.drawLine(cx, cy - arm, cx, cy + arm)

        painter.setPen(QtGui.QPen(QtGui.QColor(255, 70, 70), 1))
        painter.drawLine(cx - arm, cy, cx + arm, cy)
        painter.drawLine(cx, cy - arm, cx, cy + arm)


class CursorZoomPanel(QtWidgets.QFrame):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._zoom_factor = 4.0
        self._zoom_step = 0.5
        self._min_zoom = 1.5
        self._max_zoom = 12.0

        self.setObjectName("cursorZoomPanel")
        self.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        self.setFixedSize(300, 234)
        self.setStyleSheet(
            "#cursorZoomPanel {"
            "background: rgba(255, 255, 255, 235);"
            "border: 1px solid #9aa0a6;"
            "border-radius: 6px;"
            "}"
            "QToolButton { min-width: 24px; }"
        )

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(6)
        title = QtWidgets.QLabel("Cursor Zoom")
        title.setStyleSheet("font-weight: 600;")
        header.addWidget(title)
        header.addStretch(1)

        self.zoom_out_button = QtWidgets.QToolButton()
        self.zoom_out_button.setText("-")
        self.zoom_out_button.setToolTip("Zoom out")
        header.addWidget(self.zoom_out_button)

        self.zoom_value_label = QtWidgets.QLabel()
        self.zoom_value_label.setMinimumWidth(38)
        self.zoom_value_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        header.addWidget(self.zoom_value_label)

        self.zoom_in_button = QtWidgets.QToolButton()
        self.zoom_in_button.setText("+")
        self.zoom_in_button.setToolTip("Zoom in")
        header.addWidget(self.zoom_in_button)

        outer.addLayout(header)

        self.viewport = ZoomViewport(self)
        outer.addWidget(self.viewport)

        self.zoom_out_button.clicked.connect(lambda: self._adjust_zoom(-self._zoom_step))
        self.zoom_in_button.clicked.connect(lambda: self._adjust_zoom(self._zoom_step))

        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self.refresh_view)
        self._timer.start()
        self._update_zoom_label()
        self.refresh_view()

    def _adjust_zoom(self, delta: float) -> None:
        new_factor = self._zoom_factor + delta
        self._zoom_factor = max(self._min_zoom, min(self._max_zoom, new_factor))
        self._update_zoom_label()
        self.refresh_view()

    def _update_zoom_label(self) -> None:
        self.zoom_value_label.setText(f"{self._zoom_factor:.1f}x")

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        if not self._timer.isActive():
            self._timer.start()

    def hideEvent(self, event: QtGui.QHideEvent) -> None:
        super().hideEvent(event)
        self._timer.stop()

    def refresh_view(self) -> None:
        if not self.isVisible():
            return
        cursor_pos = QtGui.QCursor.pos()
        screen = QtGui.QGuiApplication.screenAt(cursor_pos)
        if screen is None:
            screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return

        sample_w = max(24, int(round(self.viewport.width() / self._zoom_factor)))
        sample_h = max(24, int(round(self.viewport.height() / self._zoom_factor)))
        sample_x = int(round(cursor_pos.x() - sample_w / 2))
        sample_y = int(round(cursor_pos.y() - sample_h / 2))

        pixmap = screen.grabWindow(0, sample_x, sample_y, sample_w, sample_h)
        if pixmap.isNull():
            return
        self.viewport.set_pixmap(pixmap)


class PersistentSelectionMenu(QtWidgets.QMenu):
    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        action = self.actionAt(event.position().toPoint())
        if action is not None and action.isEnabled() and action.menu() is None:
            action.trigger()
            event.accept()
            return
        super().mouseReleaseEvent(event)


@dataclass
class ColorExtractionState:
    color: Optional[Tuple[int, int, int]] = None
    base_points: list[PlacedPoint] = field(default_factory=list)
    points: list[PlacedPoint] = field(default_factory=list)
    interpolate_enabled: bool = False
    limit_points_to_calib: bool = False
    pick_color_mode: bool = False




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
        self._color_states: list[ColorExtractionState] = [ColorExtractionState()]
        self._active_color_index = 0
        self._export_selected_color_indices: set[int] = set()
        self._slot_buttons: list[QtWidgets.QAbstractButton] = []
        self.zoom_panel: Optional[CursorZoomPanel] = None

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
        self.export_menu.addSeparator()
        self.export_selection_menu = PersistentSelectionMenu("Colors to Export", self.export_menu)
        self.export_selection_menu.aboutToShow.connect(self._refresh_export_selection_menu)
        self.export_menu.addMenu(self.export_selection_menu)
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
        self.color_row.addWidget(self.selected_color_label)
        self.color_slots_widget = QtWidgets.QWidget()
        self.color_slots_layout = QtWidgets.QHBoxLayout(self.color_slots_widget)
        self.color_slots_layout.setContentsMargins(0, 0, 0, 0)
        self.color_slots_layout.setSpacing(4)
        self.color_row.addWidget(self.color_slots_widget)
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

        self.zoom_panel = CursorZoomPanel(central)
        self.zoom_panel.show()
        self._position_zoom_panel()
        self.zoom_panel.raise_()
        QtCore.QTimer.singleShot(0, self._position_zoom_panel)
        self._rebuild_color_slot_buttons()
        self._sync_active_color_actions()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._position_zoom_panel()

    def _position_zoom_panel(self) -> None:
        if self.zoom_panel is None:
            return
        central = self.centralWidget()
        if central is None:
            return
        margin = 10
        x = max(margin, central.width() - self.zoom_panel.width() - margin)
        self.zoom_panel.move(x, margin)
        self.zoom_panel.raise_()

    def _active_color_state(self) -> ColorExtractionState:
        return self._color_states[self._active_color_index]

    def _save_active_color_state(self) -> None:
        state = self._active_color_state()
        state.color = self._selected_color
        state.base_points = list(self._base_points)
        state.points = list(self._points)
        state.interpolate_enabled = self._interpolate_enabled
        state.limit_points_to_calib = self._limit_points_to_calib
        state.pick_color_mode = self._pick_color_mode

    def _restore_active_color_state(self) -> None:
        state = self._active_color_state()
        self._selected_color = state.color
        self._base_points = list(state.base_points)
        self._points = list(state.points)
        self._interpolate_enabled = state.interpolate_enabled
        self._limit_points_to_calib = state.limit_points_to_calib
        self._pick_color_mode = state.pick_color_mode

    def _color_slot_label(self, index: int) -> str:
        state = self._color_states[index]
        if state.color is None:
            return f"Color {index + 1}"
        return f"Color {index + 1} ({state.color[0]}, {state.color[1]}, {state.color[2]})"

    def _available_color_indices(self, require_points: bool = False) -> list[int]:
        indices: list[int] = []
        for index, state in enumerate(self._color_states):
            if state.color is None:
                continue
            if require_points and not state.points:
                continue
            indices.append(index)
        return indices

    def _normalize_export_color_selection(self) -> None:
        available = set(self._available_color_indices())
        self._export_selected_color_indices &= available
        if not self._export_selected_color_indices and available:
            if self._active_color_index in available:
                self._export_selected_color_indices = {self._active_color_index}
            else:
                self._export_selected_color_indices = {min(available)}

    def _refresh_export_selection_menu(self) -> None:
        self._save_active_color_state()
        self._normalize_export_color_selection()
        self.export_selection_menu.clear()

        active_action = self.export_selection_menu.addAction("Active Color Only")
        active_action.triggered.connect(self._set_export_active_color_only)
        all_action = self.export_selection_menu.addAction("All Configured Colors")
        all_action.triggered.connect(self._set_export_all_colors)
        self.export_selection_menu.addSeparator()

        available = self._available_color_indices()
        if not available:
            placeholder = self.export_selection_menu.addAction("No colors configured")
            placeholder.setEnabled(False)
            return

        for index in available:
            action = self.export_selection_menu.addAction(self._color_slot_label(index))
            action.setCheckable(True)
            action.setChecked(index in self._export_selected_color_indices)
            action.toggled.connect(lambda checked, color_index=index: self._toggle_export_color(color_index, checked))

    def _set_export_active_color_only(self) -> None:
        if self._selected_color is None:
            self._export_selected_color_indices.clear()
        else:
            self._export_selected_color_indices = {self._active_color_index}
        if self.export_selection_menu.isVisible():
            QtCore.QTimer.singleShot(0, self._refresh_export_selection_menu)

    def _set_export_all_colors(self) -> None:
        self._export_selected_color_indices = set(self._available_color_indices())
        if self.export_selection_menu.isVisible():
            QtCore.QTimer.singleShot(0, self._refresh_export_selection_menu)

    def _toggle_export_color(self, index: int, checked: bool) -> None:
        if checked:
            self._export_selected_color_indices.add(index)
        else:
            self._export_selected_color_indices.discard(index)

    def _slot_button_style(
        self,
        is_active: bool,
        color: Optional[Tuple[int, int, int]],
        is_add_button: bool = False,
    ) -> str:
        border_color = "#1a7f37" if is_active else "#000000"
        border_width = "2px" if is_active else "1px"
        if color is None:
            background = "#ffffff" if is_add_button else "#f1f3f4"
        else:
            background = f"rgb({color[0]}, {color[1]}, {color[2]})"
        text_color = "#1a7f37" if is_add_button else "#000000"
        return (
            "QToolButton {"
            f"border: {border_width} solid {border_color};"
            "border-radius: 2px;"
            f"background: {background};"
            f"color: {text_color};"
            "font-weight: 600;"
            "font-size: 14px;"
            "padding: 0px;"
            "margin: 0px;"
            "}"
        )

    def _set_active_color_index(self, index: int) -> None:
        if index < 0 or index >= len(self._color_states) or index == self._active_color_index:
            return
        self._save_active_color_state()
        self._active_color_index = index
        self._restore_active_color_state()
        self._sync_active_color_actions()
        self._rebuild_color_slot_buttons()
        self._refresh_point_overlays()
        self._set_status_for_active_color()

    def _set_status_for_active_color(self, prefix: str = "") -> None:
        label = self._color_slot_label(self._active_color_index)
        if self._selected_color is None:
            message = f"{label} is active. Use Pick Color to assign this slot."
        else:
            point_suffix = f" Points: {len(self._points)}."
            message = f"{label} is active.{point_suffix}"
        if prefix:
            message = prefix + " " + message
        self.status_label.setText(message)

    def _add_color_slot(self) -> None:
        self._save_active_color_state()
        self._color_states.append(ColorExtractionState())
        self._active_color_index = len(self._color_states) - 1
        self._restore_active_color_state()
        self._sync_active_color_actions()
        self._rebuild_color_slot_buttons()
        self._refresh_point_overlays()
        self._set_status_for_active_color("Added a new color slot.")

    def _rebuild_color_slot_buttons(self) -> None:
        while self.color_slots_layout.count():
            item = self.color_slots_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        self._slot_buttons = []
        swatch_size = max(20, self.selected_color_label.sizeHint().height())
        for index, state in enumerate(self._color_states):
            button = QtWidgets.QToolButton()
            button.setFixedSize(swatch_size, swatch_size)
            button.setToolTip(self._color_slot_label(index))
            button.setText("")
            button.setStyleSheet(self._slot_button_style(index == self._active_color_index, state.color))
            button.clicked.connect(lambda _checked=False, color_index=index: self._set_active_color_index(color_index))
            self.color_slots_layout.addWidget(button)
            self._slot_buttons.append(button)
            if index < len(self._color_states) - 1:
                comma_label = QtWidgets.QLabel(",")
                comma_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                self.color_slots_layout.addWidget(comma_label)

        plus_button = QtWidgets.QToolButton()
        plus_button.setFixedSize(swatch_size, swatch_size)
        plus_button.setText("+")
        plus_button.setToolTip("Add color slot")
        plus_button.setStyleSheet(self._slot_button_style(False, None, is_add_button=True))
        plus_button.clicked.connect(self._add_color_slot)
        self.color_slots_layout.addWidget(plus_button)

    def _sync_active_color_actions(self) -> None:
        for action, checked in (
            (self.act_interpolate, self._interpolate_enabled),
            (self.act_place_points_limit, self._limit_points_to_calib),
            (self.act_pick_color, self._pick_color_mode),
        ):
            action.blockSignals(True)
            action.setChecked(checked)
            action.blockSignals(False)
        if self._pick_color_mode:
            self.image_tray.setCursor(QtCore.Qt.CursorShape.CrossCursor)
        else:
            self.image_tray.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

    def _inverse_rgb(self, color: Tuple[int, int, int]) -> Tuple[int, int, int]:
        return 255 - color[0], 255 - color[1], 255 - color[2]

    def _refresh_point_overlays(self) -> None:
        groups: list[tuple[list[tuple[int, int]], tuple[int, int, int], bool]] = []
        for index, state in enumerate(self._color_states):
            if state.color is None or not state.points:
                continue
            groups.append(
                (
                    [(pt.x, pt.y) for pt in state.points],
                    self._inverse_rgb(state.color),
                    index == self._active_color_index,
                )
            )
        self.image_tray.set_point_groups(groups)

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
        self.image_tray.set_point_groups([])
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
        self._selected_color = auto_color
        self._interpolate_enabled = False
        self._limit_points_to_calib = False
        self._pick_color_mode = False
        self._color_states = [
            ColorExtractionState(
                color=auto_color,
                base_points=[],
                points=[],
                interpolate_enabled=False,
                limit_points_to_calib=False,
                pick_color_mode=False,
            )
        ]
        self._active_color_index = 0
        self._export_selected_color_indices = {0} if auto_color is not None else set()
        self._restore_active_color_state()
        self._sync_active_color_actions()
        self._rebuild_color_slot_buttons()
        self._refresh_point_overlays()
        if auto_color:
            self._update_color_swatch(QtGui.QColor(*auto_color))
            self.status_label.setText(f"Image loaded. Auto-selected color: {auto_color}")
        else:
            self._update_color_swatch(QtGui.QColor(255, 255, 255))
            self.status_label.setText("Image loaded. Pick a color to start.")

    def toggle_pick_color(self, checked: bool) -> None:
        self._pick_color_mode = checked
        self._active_color_state().pick_color_mode = checked
        if checked:
            self.image_tray.setCursor(QtCore.Qt.CursorShape.CrossCursor)
            self.status_label.setText(f"Pick color for {self._color_slot_label(self._active_color_index)}: click on the image.")
        else:
            self.image_tray.setCursor(QtCore.Qt.CursorShape.ArrowCursor)
            self._set_status_for_active_color()

    def on_image_clicked(self, x: int, y: int) -> None:
        if self._image is None:
            return
        if self._calibration_mode == "manual":
            self._handle_manual_calibration_click(x, y)
            return
        if self._pick_color_mode:
            color = self._image.pixelColor(x, y)
            self._selected_color = (color.red(), color.green(), color.blue())
            state = self._active_color_state()
            state.color = self._selected_color
            state.base_points = []
            state.points = []
            self._base_points = []
            self._points = []
            self._update_color_swatch(color)
            self._refresh_point_overlays()
            self._rebuild_color_slot_buttons()
            self.act_pick_color.setChecked(False)
            self.status_label.setText(f"Assigned {self._selected_color} to {self._color_slot_label(self._active_color_index)}.")

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
        self.status_label.setText(
            f"{self._color_slot_label(self._active_color_index)} placed {len(self._points)} points{suffix}{limit_suffix}."
        )

    def toggle_interpolation(self, checked: bool) -> None:
        self._interpolate_enabled = checked
        self._active_color_state().interpolate_enabled = checked
        if self._image is None or self._selected_color is None:
            state = "on" if checked else "off"
            self.status_label.setText(f"Interpolation {state}. Load an image and pick a color.")
            return
        self._compute_points()
        state = "on" if checked else "off"
        self.status_label.setText(f"Interpolation {state} for {self._color_slot_label(self._active_color_index)}. Points: {len(self._points)}.")

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
        self._active_color_state().limit_points_to_calib = checked
        if self._image is None or self._selected_color is None:
            state = "on" if checked else "off"
            self.status_label.setText(f"Limit to calibration {state}.")
            return
        self._compute_points()
        state = "on" if checked else "off"
        self.status_label.setText(
            f"Limit to calibration {state} for {self._color_slot_label(self._active_color_index)}. Points: {len(self._points)}."
        )

    def _apply_interpolation(self, points: list[PlacedPoint], enabled: Optional[bool] = None) -> list[PlacedPoint]:
        use_interpolation = self._interpolate_enabled if enabled is None else enabled
        if use_interpolation:
            return interpolate_points(points, segment_len=5.0, points_per_segment=3)
        return list(points)

    def _compute_points_for_state(self, state: ColorExtractionState) -> None:
        if self._image is None or state.color is None:
            state.base_points = []
            state.points = []
            return
        min_chroma = self._chroma_min if self._chroma_filter_enabled else None
        base_points = find_points_by_color(
            self._image,
            state.color,
            min_chroma=min_chroma,
            exclude_rects=self._all_mask_rects(),
        )
        points = base_points
        if state.limit_points_to_calib:
            box = self._get_calibration_box()
            if box is not None:
                left, right, top, bottom = box
                points = [pt for pt in points if left <= pt.x <= right and top <= pt.y <= bottom]
        state.base_points = list(base_points)
        state.points = self._apply_interpolation(points, enabled=state.interpolate_enabled)

    def _compute_points(self) -> None:
        state = self._active_color_state()
        self._compute_points_for_state(state)
        self._restore_active_color_state()
        self._refresh_point_overlays()

    def run_mask_words(self) -> None:
        if self._image is None:
            self.status_label.setText("Load an image first.")
            return
        try:
            rects = detect_word_masks(self._image)
        except Exception as exc:
            log_exception("Mask words", exc)
            detail = str(exc).strip()
            if detail:
                self.status_label.setText(f"Mask words failed: {detail}")
            else:
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
            detail = str(exc).strip()
            if detail:
                self.status_label.setText(f"Mask numbers failed: {detail}")
            else:
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
            detail = str(exc).strip()
            if detail:
                self.status_label.setText(f"Mask legend failed: {detail}")
            else:
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
        self._save_active_color_state()
        self._normalize_export_color_selection()
        selected_indices = sorted(self._export_selected_color_indices)
        if not selected_indices:
            self.status_label.setText("Choose at least one configured color in Export -> Colors to Export.")
            return

        headers = ["color_slot", "color_r", "color_g", "color_b", "x", "y", "x_px", "y_px"]
        if normalize_y:
            headers.append("y_norm")

        rows: list[list[float]] = []
        exported_indices: list[int] = []
        for index in selected_indices:
            state = self._color_states[index]
            if state.color is None:
                continue
            if not state.points:
                self._compute_points_for_state(state)
            if not state.points:
                continue
            exported_indices.append(index)
            color_r, color_g, color_b = state.color
            for pt in state.points:
                x_px = pt.x
                y_px = pt.y
                if mapper is not None:
                    x_val, y_val = self._map_pixel_affine(mapper, x_px, y_px)
                else:
                    x_val = x_min_val + (x_px - left) * (x_max_val - x_min_val) / (right - left)
                    y_val = y_min_val + (bottom - y_px) * (y_max_val - y_min_val) / (bottom - top)
                row: list[float] = [index + 1, color_r, color_g, color_b, x_val, y_val, x_px, y_px]
                if normalize_y:
                    denom = y_max_val - y_min_val
                    y_norm = (y_val - y_min_val) / denom if denom != 0 else 0.0
                    row.append(y_norm)
                rows.append(row)

        self._restore_active_color_state()
        self._refresh_point_overlays()

        if not rows:
            self.status_label.setText("No points available to export for the selected colors.")
            return

        saved = False
        if kind == "csv":
            saved = self._export_csv(headers, rows)
        else:
            saved = self._export_excel(headers, rows)
        if not saved:
            return
        suffix = " (normalized)" if normalize_y else ""
        msg = f"Exported {len(rows)} points across {len(exported_indices)} color slot(s){suffix}."
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
        self._rebuild_color_slot_buttons()

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

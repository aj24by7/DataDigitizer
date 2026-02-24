from __future__ import annotations

from typing import Iterable, Optional, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets


class ImageTray(QtWidgets.QWidget):
    image_clicked = QtCore.pyqtSignal(int, int)
    mask_rect_created = QtCore.pyqtSignal(float, float, float, float)
    mask_remove_requested = QtCore.pyqtSignal(int, int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._image: Optional[QtGui.QImage] = None
        self._points: list[Tuple[int, int]] = []
        self._axis_points: list[Tuple[float, float]] = []
        self._axis_rects: list[Tuple[float, float, float, float]] = []
        self._calibration_points: list[Tuple[float, float]] = []
        self._calibration_box: Optional[list[Tuple[float, float]]] = None
        self._mask_rects: list[Tuple[float, float, float, float]] = []
        self._mask_draw_enabled = False
        self._mask_dragging = False
        self._mask_drag_start: Optional[Tuple[int, int]] = None
        self._mask_preview: Optional[Tuple[float, float, float, float]] = None
        self._dragging_line: Optional[int] = None
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self._set_white_background()

    def _set_white_background(self) -> None:
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(255, 255, 255))
        self.setPalette(palette)

    def set_image(self, image: QtGui.QImage) -> None:
        self._image = image
        self._points = []
        self._axis_points = []
        self._axis_rects = []
        self._calibration_points = []
        self._calibration_box = None
        self._mask_rects = []
        self._mask_dragging = False
        self._mask_drag_start = None
        self._mask_preview = None
        self.update()

    def set_points(self, points: Iterable[Tuple[float, float]]) -> None:
        self._points = list(points)
        self.update()

    def set_axis_overlays(
        self,
        points: Iterable[Tuple[float, float]],
        rects: Iterable[Tuple[float, float, float, float]],
    ) -> None:
        self._axis_points = list(points)
        self._axis_rects = list(rects)
        self.update()

    def set_calibration_overlays(
        self,
        points: Iterable[Tuple[float, float]],
        box: Optional[Iterable[Tuple[float, float]]],
    ) -> None:
        self._calibration_points = list(points)
        self._calibration_box = list(box) if box else None
        self.update()

    def set_mask_overlays(self, rects: Iterable[Tuple[float, float, float, float]]) -> None:
        self._mask_rects = list(rects)
        self.update()

    def set_mask_draw_enabled(self, enabled: bool) -> None:
        self._mask_draw_enabled = enabled
        if not enabled:
            self._mask_dragging = False
            self._mask_drag_start = None
            self._mask_preview = None
            self.unsetCursor()
        self.update()

    def get_display_size(self) -> Optional[Tuple[int, int]]:
        if self._image is None or self._image.isNull():
            return None
        target = self._fit_rect(self._image.size(), self.rect())
        width = max(1, int(round(target.width())))
        height = max(1, int(round(target.height())))
        return width, height

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), QtGui.QColor(255, 255, 255))
        if self._image is None or self._image.isNull():
            return

        target = self._fit_rect(self._image.size(), self.rect())
        painter.drawImage(target, self._image)
        if self._mask_rects:
            self._draw_mask_overlays(painter, target)
        if self._mask_preview:
            self._draw_mask_preview(painter, target)
        if self._points:
            self._draw_points(painter, target)
        if self._axis_points or self._axis_rects:
            self._draw_axis_overlays(painter, target)
        if self._calibration_points or self._calibration_box:
            self._draw_calibration_overlays(painter, target)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._image is None or self._image.isNull():
            return
        target = self._fit_rect(self._image.size(), self.rect())
        if not target.contains(event.position()):
            return
        x, y = self._map_to_image(event.position(), target)
        if self._mask_draw_enabled:
            if event.modifiers() & QtCore.Qt.KeyboardModifier.ShiftModifier:
                if self._point_in_mask(x, y):
                    self.mask_remove_requested.emit(x, y)
                    return
            self._mask_dragging = True
            self._mask_drag_start = (x, y)
            self._mask_preview = (x, y, 0.0, 0.0)
            return
        hit = self._hit_calibration_line(x, y)
        if hit is not None:
            self._dragging_line = hit
            return
        if 0 <= x < self._image.width() and 0 <= y < self._image.height():
            self.image_clicked.emit(x, y)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._image is None or self._image.isNull():
            return
        target = self._fit_rect(self._image.size(), self.rect())
        if not target.contains(event.position()):
            return
        x, y = self._map_to_image(event.position(), target)
        if self._mask_draw_enabled and self._mask_dragging and self._mask_drag_start:
            self._mask_preview = _rect_from_points(self._mask_drag_start, (x, y))
            self.update()
            return
        if self._dragging_line is not None and self._calibration_box:
            self._adjust_calibration_box(x, y)
            self.update()
            return
        hit = self._hit_calibration_line(x, y)
        if hit is None:
            self.unsetCursor()
        elif hit in (0, 2):
            self.setCursor(QtCore.Qt.CursorShape.SizeVerCursor)
        else:
            self.setCursor(QtCore.Qt.CursorShape.SizeHorCursor)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._mask_draw_enabled and self._mask_dragging and self._mask_preview:
            x, y, w, h = self._mask_preview
            if w >= 3 and h >= 3:
                self.mask_rect_created.emit(x, y, w, h)
            self._mask_dragging = False
            self._mask_drag_start = None
            self._mask_preview = None
            self.update()
            return
        if self._dragging_line is not None:
            self._dragging_line = None
            self.unsetCursor()

    def _fit_rect(self, image_size: QtCore.QSize, bounds: QtCore.QRect) -> QtCore.QRectF:
        img_w = image_size.width()
        img_h = image_size.height()
        if img_w <= 0 or img_h <= 0:
            return QtCore.QRectF()
        scale = min(bounds.width() / img_w, bounds.height() / img_h)
        draw_w = img_w * scale
        draw_h = img_h * scale
        x = bounds.x() + (bounds.width() - draw_w) / 2
        y = bounds.y() + (bounds.height() - draw_h) / 2
        return QtCore.QRectF(x, y, draw_w, draw_h)

    def _map_to_image(
        self,
        pos: QtCore.QPointF,
        target: QtCore.QRectF,
    ) -> Tuple[int, int]:
        if target.width() == 0 or target.height() == 0:
            return 0, 0
        rel_x = (pos.x() - target.left()) / target.width()
        rel_y = (pos.y() - target.top()) / target.height()
        x = int(rel_x * self._image.width())
        y = int(rel_y * self._image.height())
        x = max(0, min(x, self._image.width() - 1))
        y = max(0, min(y, self._image.height() - 1))
        return x, y

    def _draw_points(self, painter: QtGui.QPainter, target: QtCore.QRectF) -> None:
        pen = QtGui.QPen(QtGui.QColor(0, 120, 255))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 120, 255)))
        scale_x = target.width() / self._image.width()
        scale_y = target.height() / self._image.height()
        for x, y in self._points:
            px = target.left() + x * scale_x
            py = target.top() + y * scale_y
            painter.drawEllipse(QtCore.QPointF(px, py), 3, 3)

    def _draw_axis_overlays(self, painter: QtGui.QPainter, target: QtCore.QRectF) -> None:
        pen = QtGui.QPen(QtGui.QColor(255, 140, 0))
        pen.setWidth(1)
        pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        scale_x = target.width() / self._image.width()
        scale_y = target.height() / self._image.height()

        for x, y, w, h in self._axis_rects:
            rx = target.left() + x * scale_x
            ry = target.top() + y * scale_y
            rw = w * scale_x
            rh = h * scale_y
            painter.drawRect(QtCore.QRectF(rx, ry, rw, rh))

        painter.setPen(QtGui.QPen(QtGui.QColor(255, 140, 0), 1))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 140, 0)))
        for x, y in self._axis_points:
            px = target.left() + x * scale_x
            py = target.top() + y * scale_y
            painter.drawEllipse(QtCore.QPointF(px, py), 2, 2)

    def _draw_calibration_overlays(self, painter: QtGui.QPainter, target: QtCore.QRectF) -> None:
        scale_x = target.width() / self._image.width()
        scale_y = target.height() / self._image.height()

        pen = QtGui.QPen(QtGui.QColor(0, 160, 100))
        pen.setWidth(2)
        pen.setStyle(QtCore.Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)

        if self._calibration_box and len(self._calibration_box) == 4:
            mapped = []
            for x, y in self._calibration_box:
                px = target.left() + x * scale_x
                py = target.top() + y * scale_y
                mapped.append(QtCore.QPointF(px, py))
            for i in range(4):
                painter.drawLine(mapped[i], mapped[(i + 1) % 4])

        painter.setPen(QtGui.QPen(QtGui.QColor(0, 160, 100), 1))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 160, 100)))
        for x, y in self._calibration_points:
            px = target.left() + x * scale_x
            py = target.top() + y * scale_y
            painter.drawEllipse(QtCore.QPointF(px, py), 3, 3)

    def _draw_mask_overlays(self, painter: QtGui.QPainter, target: QtCore.QRectF) -> None:
        color = QtGui.QColor(255, 0, 0, 80)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QBrush(color))
        scale_x = target.width() / self._image.width()
        scale_y = target.height() / self._image.height()
        for x, y, w, h in self._mask_rects:
            rx = target.left() + x * scale_x
            ry = target.top() + y * scale_y
            rw = w * scale_x
            rh = h * scale_y
            painter.drawRect(QtCore.QRectF(rx, ry, rw, rh))

    def _draw_mask_preview(self, painter: QtGui.QPainter, target: QtCore.QRectF) -> None:
        if not self._mask_preview:
            return
        x, y, w, h = self._mask_preview
        color = QtGui.QColor(255, 0, 0, 60)
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 0, 0, 120), 1))
        painter.setBrush(QtGui.QBrush(color))
        scale_x = target.width() / self._image.width()
        scale_y = target.height() / self._image.height()
        rx = target.left() + x * scale_x
        ry = target.top() + y * scale_y
        rw = w * scale_x
        rh = h * scale_y
        painter.drawRect(QtCore.QRectF(rx, ry, rw, rh))

    def _point_in_mask(self, x: int, y: int) -> bool:
        for left, top, w, h in self._mask_rects:
            if left <= x <= left + w and top <= y <= top + h:
                return True
        return False

    def _box_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        if not self._calibration_box or len(self._calibration_box) != 4:
            return None
        xs = [p[0] for p in self._calibration_box]
        ys = [p[1] for p in self._calibration_box]
        return min(xs), max(xs), min(ys), max(ys)

    def _hit_calibration_line(self, x: int, y: int, tol: int = 5) -> Optional[int]:
        bounds = self._box_bounds()
        if bounds is None:
            return None
        left, right, top, bottom = bounds
        if left <= x <= right and abs(y - top) <= tol:
            return 0
        if top <= y <= bottom and abs(x - right) <= tol:
            return 1
        if left <= x <= right and abs(y - bottom) <= tol:
            return 2
        if top <= y <= bottom and abs(x - left) <= tol:
            return 3
        return None

    def _adjust_calibration_box(self, x: int, y: int) -> None:
        if self._calibration_box is None or self._dragging_line is None:
            return
        bounds = self._box_bounds()
        if bounds is None:
            return
        left, right, top, bottom = bounds
        min_size = 5
        old_left, old_right, old_top, old_bottom = left, right, top, bottom

        if self._dragging_line == 0:
            top = min(y, bottom - min_size)
        elif self._dragging_line == 1:
            right = max(x, left + min_size)
        elif self._dragging_line == 2:
            bottom = max(y, top + min_size)
        elif self._dragging_line == 3:
            left = min(x, right - min_size)

        self._calibration_box = [
            (left, top),
            (right, top),
            (right, bottom),
            (left, bottom),
        ]

        updated = []
        for px, py in self._calibration_points:
            if self._dragging_line in (0, 2) and abs(py - (old_top if self._dragging_line == 0 else old_bottom)) <= 2:
                py = top if self._dragging_line == 0 else bottom
            if self._dragging_line in (1, 3) and abs(px - (old_right if self._dragging_line == 1 else old_left)) <= 2:
                px = right if self._dragging_line == 1 else left
            updated.append((px, py))
        self._calibration_points = updated


def _rect_from_points(
    p0: Tuple[int, int],
    p1: Tuple[int, int],
) -> Tuple[float, float, float, float]:
    x0, y0 = p0
    x1, y1 = p1
    left = float(min(x0, x1))
    top = float(min(y0, y1))
    right = float(max(x0, x1))
    bottom = float(max(y0, y1))
    return left, top, right - left, bottom - top

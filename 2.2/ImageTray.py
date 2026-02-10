from __future__ import annotations

from typing import Iterable, Optional, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets


class ImageTray(QtWidgets.QWidget):
    image_clicked = QtCore.pyqtSignal(int, int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._image: Optional[QtGui.QImage] = None
        self._points: list[Tuple[int, int]] = []
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
        self.update()

    def set_points(self, points: Iterable[Tuple[float, float]]) -> None:
        self._points = list(points)
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
        if self._points:
            self._draw_points(painter, target)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._image is None or self._image.isNull():
            return
        target = self._fit_rect(self._image.size(), self.rect())
        if not target.contains(event.position()):
            return
        x, y = self._map_to_image(event.position(), target)
        if 0 <= x < self._image.width() and 0 <= y < self._image.height():
            self.image_clicked.emit(x, y)

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

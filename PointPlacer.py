from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from PyQt6 import QtGui

try:  # optional, faster path
    import numpy as np
except ImportError:  # pragma: no cover - optional
    np = None


@dataclass(frozen=True)
class PlacedPoint:
    x: float
    y: float


def find_points_by_color(
    image: QtGui.QImage,
    target_rgb: Tuple[int, int, int],
    tol: int = 40,
) -> List[PlacedPoint]:
    if image.isNull():
        return []
    if image.format() != QtGui.QImage.Format.Format_RGB888:
        image = image.convertToFormat(QtGui.QImage.Format.Format_RGB888)

    width = image.width()
    height = image.height()
    bytes_per_line = image.bytesPerLine()
    bits = image.bits()
    bits.setsize(bytes_per_line * height)

    if np is None:
        data = bytes(bits)
        return _find_points_py(data, width, height, bytes_per_line, target_rgb, tol)

    try:
        flat = np.frombuffer(bits, dtype=np.uint8)
    except TypeError:
        flat = np.frombuffer(bytes(bits), dtype=np.uint8)
    rows = flat.reshape((height, bytes_per_line))
    rgb = rows[:, : width * 3].reshape((height, width, 3))

    if tol <= 0:
        mask = (rgb == target_rgb).all(axis=2)
    else:
        target = np.array(target_rgb, dtype=np.int16)
        diff = np.abs(rgb.astype(np.int16) - target)
        mask = np.all(diff <= tol, axis=2)

    if not mask.any():
        return []

    col_has = mask.any(axis=0)
    top = mask.argmax(axis=0)
    bottom = (height - 1) - mask[::-1, :].argmax(axis=0)
    mid = (top + bottom) / 2.0

    xs = np.nonzero(col_has)[0]
    return [PlacedPoint(x=float(x), y=float(mid[x])) for x in xs]


def _find_points_py(
    data: bytes,
    width: int,
    height: int,
    bytes_per_line: int,
    target_rgb: Tuple[int, int, int],
    tol: int,
) -> List[PlacedPoint]:
    points: list[PlacedPoint] = []
    for x in range(width):
        y_top = _scan_column(data, bytes_per_line, height, x, target_rgb, tol, True)
        if y_top is None:
            continue
        y_bottom = _scan_column(data, bytes_per_line, height, x, target_rgb, tol, False)
        if y_bottom is None:
            continue
        y_mid = (y_top + y_bottom) / 2.0
        points.append(PlacedPoint(x=float(x), y=float(y_mid)))
    return points


def _scan_column(
    data: bytes,
    bytes_per_line: int,
    height: int,
    x: int,
    target_rgb: Tuple[int, int, int],
    tol: int,
    top_down: bool,
) -> Optional[int]:
    if top_down:
        y_iter = range(height)
    else:
        y_iter = range(height - 1, -1, -1)
    for y in y_iter:
        idx = y * bytes_per_line + x * 3
        r = data[idx]
        g = data[idx + 1]
        b = data[idx + 2]
        if _match_rgb(r, g, b, target_rgb, tol):
            return y
    return None


def _match_rgb(
    r: int,
    g: int,
    b: int,
    target_rgb: Tuple[int, int, int],
    tol: int,
) -> bool:
    return (
        abs(r - target_rgb[0]) <= tol
        and abs(g - target_rgb[1]) <= tol
        and abs(b - target_rgb[2]) <= tol
    )


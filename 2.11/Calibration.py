from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from PyQt6 import QtGui

try:  # optional faster path
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


@dataclass(frozen=True)
class CalibrationResult:
    x_min_point: Optional[Tuple[int, int]]
    x_max_point: Optional[Tuple[int, int]]
    y_min_point: Optional[Tuple[int, int]]
    y_max_point: Optional[Tuple[int, int]]
    box: Optional[List[Tuple[int, int]]]

    def points(self) -> List[Tuple[int, int]]:
        out = []
        for pt in (self.x_min_point, self.x_max_point, self.y_min_point, self.y_max_point):
            if pt is not None:
                out.append(pt)
        return out


def coordinate_mediated_calibration(
    image: QtGui.QImage,
    x_min_point: Optional[Tuple[float, float]],
    x_max_point: Optional[Tuple[float, float]],
    y_min_point: Optional[Tuple[float, float]],
    y_max_point: Optional[Tuple[float, float]],
    exclude_rects: Optional[List[Tuple[float, float, float, float]]] = None,
    exclude_margin: int = 3,
    black_threshold: int = 30,
) -> CalibrationResult:
    data = _image_bytes(image)
    if data is None:
        return CalibrationResult(None, None, None, None, None)

    bytes_data, width, height, bytes_per_line = data

    expanded_rects = _expand_rects(exclude_rects or [], exclude_margin)

    def snap_up(point: Optional[Tuple[float, float]]) -> Optional[Tuple[int, int]]:
        if point is None:
            return None
        x = int(round(point[0]))
        y = int(round(point[1]))
        return _snap_to_black(
            bytes_data,
            width,
            height,
            bytes_per_line,
            x,
            y,
            "up",
            black_threshold,
            expanded_rects,
        )

    def snap_right(point: Optional[Tuple[float, float]]) -> Optional[Tuple[int, int]]:
        if point is None:
            return None
        x = int(round(point[0]))
        y = int(round(point[1]))
        return _snap_to_black(
            bytes_data,
            width,
            height,
            bytes_per_line,
            x,
            y,
            "right",
            black_threshold,
            expanded_rects,
        )

    x_min = snap_up(x_min_point)
    x_max = snap_up(x_max_point)
    y_min = snap_right(y_min_point)
    y_max = snap_right(y_max_point)

    box = _box_from_axes(x_min, x_max, y_min, y_max)
    return CalibrationResult(x_min, x_max, y_min, y_max, box)


def line_mediated_calibration(
    image: QtGui.QImage,
    black_threshold: int = 30,
) -> CalibrationResult:
    box = _find_black_border_box(image, black_threshold)
    if box is None:
        return CalibrationResult(None, None, None, None, None)

    # box order: top-left, top-right, bottom-right, bottom-left
    top_left, top_right, bottom_right, bottom_left = box
    x_min = bottom_left
    x_max = bottom_right
    y_min = bottom_left
    y_max = top_left
    return CalibrationResult(x_min, x_max, y_min, y_max, box)


def _box_from_axes(
    x_min: Optional[Tuple[int, int]],
    x_max: Optional[Tuple[int, int]],
    y_min: Optional[Tuple[int, int]],
    y_max: Optional[Tuple[int, int]],
) -> Optional[List[Tuple[int, int]]]:
    if x_min is None or x_max is None or y_min is None or y_max is None:
        return None
    left_x = y_min[0]
    right_x = x_max[0]
    bottom_y = x_min[1]
    top_y = y_max[1]
    if right_x <= left_x or bottom_y <= top_y:
        return None
    return [(left_x, top_y), (right_x, top_y), (right_x, bottom_y), (left_x, bottom_y)]


def _image_bytes(
    image: QtGui.QImage,
) -> Optional[Tuple[bytes, int, int, int]]:
    if image.isNull():
        return None
    if image.format() != QtGui.QImage.Format.Format_RGB888:
        image = image.convertToFormat(QtGui.QImage.Format.Format_RGB888)
    width = image.width()
    height = image.height()
    bytes_per_line = image.bytesPerLine()
    bits = image.bits()
    bits.setsize(bytes_per_line * height)
    data = bytes(bits)
    return data, width, height, bytes_per_line


def _snap_to_black(
    data: bytes,
    width: int,
    height: int,
    bytes_per_line: int,
    start_x: int,
    start_y: int,
    direction: str,
    threshold: int,
    exclude_rects: List[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int]]:
    start_x = max(0, min(start_x, width - 1))
    start_y = max(0, min(start_y, height - 1))
    if direction == "right":
        for x in range(start_x, width):
            if _is_excluded(x, start_y, exclude_rects):
                continue
            if _is_black(data, bytes_per_line, x, start_y, threshold):
                return (x, start_y)
    elif direction == "up":
        for y in range(start_y, -1, -1):
            if _is_excluded(start_x, y, exclude_rects):
                continue
            if _is_black(data, bytes_per_line, start_x, y, threshold):
                return (start_x, y)
    return None


def _find_black_border_box(
    image: QtGui.QImage,
    threshold: int,
) -> Optional[List[Tuple[int, int]]]:
    data = _image_bytes(image)
    if data is None:
        return None
    bytes_data, width, height, bytes_per_line = data

    if np is not None:
        flat = np.frombuffer(bytes_data, dtype=np.uint8)
        rows = flat.reshape((height, bytes_per_line))
        rgb = rows[:, : width * 3].reshape((height, width, 3))
        mask = np.all(rgb <= threshold, axis=2)
        if not mask.any():
            return None

        # Prefer long straight border lines: find rows/cols with long black runs.
        row_max = _max_run_lengths(mask, axis=1)
        col_max = _max_run_lengths(mask, axis=0)
        row_hits = np.where(row_max >= int(width * 0.6))[0]
        col_hits = np.where(col_max >= int(height * 0.6))[0]
        if row_hits.size >= 2 and col_hits.size >= 2:
            min_y = int(row_hits.min())
            max_y = int(row_hits.max())
            min_x = int(col_hits.min())
            max_x = int(col_hits.max())
            return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

        # Fallback: bounding box of all black pixels.
        ys, xs = np.where(mask)
        min_x = int(xs.min())
        max_x = int(xs.max())
        min_y = int(ys.min())
        max_y = int(ys.max())
        return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

    row_hits: List[int] = []
    col_hits: List[int] = []
    row_run_min = int(width * 0.6)
    col_run_min = int(height * 0.6)

    for y in range(height):
        max_run = 0
        current = 0
        row = y * bytes_per_line
        for x in range(width):
            idx = row + x * 3
            r = bytes_data[idx]
            g = bytes_data[idx + 1]
            b = bytes_data[idx + 2]
            is_black = r <= threshold and g <= threshold and b <= threshold
            if is_black:
                current += 1
                if current > max_run:
                    max_run = current
            else:
                current = 0
        if max_run >= row_run_min:
            row_hits.append(y)

    for x in range(width):
        max_run = 0
        current = 0
        for y in range(height):
            idx = y * bytes_per_line + x * 3
            r = bytes_data[idx]
            g = bytes_data[idx + 1]
            b = bytes_data[idx + 2]
            is_black = r <= threshold and g <= threshold and b <= threshold
            if is_black:
                current += 1
                if current > max_run:
                    max_run = current
            else:
                current = 0
        if max_run >= col_run_min:
            col_hits.append(x)

    if len(row_hits) >= 2 and len(col_hits) >= 2:
        min_y = min(row_hits)
        max_y = max(row_hits)
        min_x = min(col_hits)
        max_x = max(col_hits)
        return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]

    min_x = width
    max_x = -1
    min_y = height
    max_y = -1
    for y in range(height):
        row = y * bytes_per_line
        for x in range(width):
            idx = row + x * 3
            r = bytes_data[idx]
            g = bytes_data[idx + 1]
            b = bytes_data[idx + 2]
            if r <= threshold and g <= threshold and b <= threshold:
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
    if max_x < 0:
        return None
    return [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]


def _max_run_lengths(mask: "np.ndarray", axis: int) -> "np.ndarray":
    # axis=1 -> per row, axis=0 -> per column
    if axis == 1:
        runs = []
        for row in mask:
            runs.append(_max_run_1d(row))
        return np.array(runs)
    runs = []
    for col in mask.T:
        runs.append(_max_run_1d(col))
    return np.array(runs)


def _max_run_1d(values: "np.ndarray") -> int:
    # longest consecutive True segment
    best = 0
    current = 0
    for v in values:
        if v:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return best


def _is_black(
    data: bytes,
    bytes_per_line: int,
    x: int,
    y: int,
    threshold: int,
) -> bool:
    idx = y * bytes_per_line + x * 3
    r = data[idx]
    g = data[idx + 1]
    b = data[idx + 2]
    return r <= threshold and g <= threshold and b <= threshold


def _expand_rects(
    rects: List[Tuple[float, float, float, float]],
    margin: int,
) -> List[Tuple[int, int, int, int]]:
    expanded: List[Tuple[int, int, int, int]] = []
    for left, top, width, height in rects:
        x0 = int(left) - margin
        y0 = int(top) - margin
        x1 = int(left + width) + margin
        y1 = int(top + height) + margin
        expanded.append((x0, y0, x1, y1))
    return expanded


def _is_excluded(x: int, y: int, rects: List[Tuple[int, int, int, int]]) -> bool:
    for x0, y0, x1, y1 in rects:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False

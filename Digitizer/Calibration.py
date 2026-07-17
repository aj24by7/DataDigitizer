from __future__ import annotations

import math
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

    # A snap can land on the wrong line entirely: exclude_rects hides every OCR text bbox,
    # and an oversized/misplaced one (the rotated y-label pass produces these) can cover the
    # real axis under a tick label, sending snap_up past it to the top spine. The pair then
    # spans a diagonal instead of the axis, and _build_affine_mapper skews the whole frame.
    # The mapper only needs each pair to be PARALLEL to its axis (a constant offset of the
    # line cancels out of the u/v solve), so pairs that already agree are left untouched.
    rows, cols = _axis_line_bands(bytes_data, width, height, bytes_per_line, black_threshold)
    x_min, x_max = _repair_axis_pair(
        x_min, x_max, x_min_point, x_max_point, rows, 1, height, "up",
        bytes_data, bytes_per_line, width, height, black_threshold,
    )
    y_min, y_max = _repair_axis_pair(
        y_min, y_max, y_min_point, y_max_point, cols, 0, width, "right",
        bytes_data, bytes_per_line, width, height, black_threshold,
    )

    # Draw the dashed calibration window from the RAW (un-snapped) input points, exactly
    # the way manual calibration builds it from the four clicked points. The snapped
    # points above are still what the coordinate math consumes (the affine mapper reads
    # CalibrationResult.x_min_point ... via DigitizerWindow._get_axis_points), but the
    # *box* must hang at the tick-label extents instead of being pulled onto the black
    # axis lines — that snapping is why the old box hugged the axes and "touched" the
    # min/max dots. Using the raw points makes this box identical in spirit to manual's.
    def _round_point(point):
        if point is None:
            return None
        return (int(round(point[0])), int(round(point[1])))

    box = _box_from_axes(
        _round_point(x_min_point),
        _round_point(x_max_point),
        _round_point(y_min_point),
        _round_point(y_max_point),
    )
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
    # GUI-only box geometry. Build the dashed rectangle from the DATA EXTENT of the four
    # axis points: X-min/X-max fix the left/right edges, Y-min/Y-max fix the bottom/top.
    # This is the same construction manual calibration uses (see
    # DigitizerWindow._manual_box_corners), so the coordinate-mediated box now "hangs" at
    # the tick extents instead of hugging the y-axis/x-axis lines and touching the snapped
    # min/max dots. Only the drawn overlay is affected — export uses the affine mapper built
    # from the snapped points (DigitizerWindow._get_axis_points), not this box.
    left_x = x_min[0]
    right_x = x_max[0]
    bottom_y = y_min[1]
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


# A row/column must be dark across this fraction of the image to count as an axis line.
_AXIS_LINE_COVERAGE = 0.5
# A plain frame has at most 4 such bands per orientation. More than that means we are not
# looking at a frame (black gridlines, a filled region, a dark photo), so the line evidence
# is not trustworthy and arbitration falls through to snap travel instead.
_MAX_FRAME_BANDS = 4
# Tilt between a tick pair beyond which the pair cannot be a scanned/rotated axis and one
# of the two snaps must have stopped on the wrong line.
_MAX_TILT_DEG = 6.0


def _axis_line_bands(
    data: bytes,
    width: int,
    height: int,
    bytes_per_line: int,
    threshold: int,
) -> Tuple[List[int], List[int]]:
    """Find full-length dark rows/columns (plot spines), one representative each.

    Returns ([], []) when numpy is unavailable or nothing qualifies; callers must treat
    that as "no evidence" and fall back, never as a failure.
    """
    if np is None:
        return [], []
    try:
        flat = np.frombuffer(data, dtype=np.uint8)
        rows_view = flat.reshape((height, bytes_per_line))
        rgb = rows_view[:, : width * 3].reshape((height, width, 3))
        mask = np.all(rgb <= threshold, axis=2)
    except ValueError:  # pragma: no cover - defensive against odd strides
        return [], []
    rows = _bands(np.where(mask.sum(axis=1) >= _AXIS_LINE_COVERAGE * width)[0].tolist())
    cols = _bands(np.where(mask.sum(axis=0) >= _AXIS_LINE_COVERAGE * height)[0].tolist())
    return rows, cols


def _bands(indices: List[int]) -> List[int]:
    """Collapse runs of adjacent indices (an axis line is a few pixels thick) to a midpoint."""
    if not indices:
        return []
    out: List[int] = []
    run = [indices[0]]
    for value in indices[1:]:
        if value - run[-1] <= 2:
            run.append(value)
        else:
            out.append(run[len(run) // 2])
            run = [value]
    out.append(run[len(run) // 2])
    return out


def _repair_axis_pair(
    a: Optional[Tuple[int, int]],
    b: Optional[Tuple[int, int]],
    a_start: Optional[Tuple[float, float]],
    b_start: Optional[Tuple[float, float]],
    bands: List[int],
    comp: int,
    span: int,
    direction: str,
    data: bytes,
    bytes_per_line: int,
    width: int,
    height: int,
    threshold: int,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    """Force a tick pair back onto a common row (comp=1) or column (comp=0).

    Fires only when the pair is tilted far past anything a real scan could produce, i.e.
    when one snap demonstrably stopped on the wrong line. A mild tilt is left alone: the
    affine mapper handles genuinely rotated plots, and clobbering that would be a
    regression. Anything ambiguous is returned untouched.
    """
    if a is None or b is None:
        return a, b
    other = 1 - comp
    run = abs(a[other] - b[other])
    if run < max(8, 0.02 * span):
        return a, b  # ticks too close together for the tilt to mean anything
    if math.degrees(math.atan2(abs(a[comp] - b[comp]), run)) <= _MAX_TILT_DEG:
        return a, b

    project_tol = max(4, int(round(0.01 * span)))
    line_tol = max(15, int(round(0.03 * span)))
    a_good: Optional[bool] = None

    if bands and len(bands) <= _MAX_FRAME_BANDS:
        # Ask each point whether it stopped at the line it SHOULD have hit: the first band
        # lying in its own scan direction. Merely being near some band is not evidence — a
        # snap that overshoots to the far spine is also "near a band".
        def on_expected(point: Tuple[int, int], start: Optional[Tuple[float, float]]) -> Optional[bool]:
            if start is None:
                return None
            origin = int(round(start[comp]))
            ahead = [n for n in bands if (n <= origin if direction == "up" else n >= origin)]
            if not ahead:
                return None
            expected = max(ahead) if direction == "up" else min(ahead)
            return abs(point[comp] - expected) <= line_tol

        near_a = on_expected(a, a_start)
        near_b = on_expected(b, b_start)
        if near_a is not None and near_b is not None and near_a != near_b:
            a_good = near_a

    if a_good is None and a_start is not None and b_start is not None:
        # No usable line evidence: the point that snapped a short way from its own tick
        # label is the plausible one; a snap that ran a long way crossed the axis it wanted.
        travel_a = abs(a[comp] - int(round(a_start[comp])))
        travel_b = abs(b[comp] - int(round(b_start[comp])))
        if abs(travel_a - travel_b) > line_tol:
            a_good = travel_a < travel_b

    if a_good is None:
        # Past the tilt gate one of these IS wrong, and the pair only has to end up
        # parallel to the axis — the mapper cancels a constant offset of the line, so an
        # arbitrary-but-consistent choice still maps correctly. Keep the first point.
        a_good = True

    winner, loser = (a, b) if a_good else (b, a)
    fixed = _project_onto_line(
        loser, winner[comp], comp, project_tol, data, bytes_per_line, width, height, threshold
    )
    return (winner, fixed) if a_good else (fixed, winner)


def _project_onto_line(
    point: Tuple[int, int],
    target: int,
    comp: int,
    tol: int,
    data: bytes,
    bytes_per_line: int,
    width: int,
    height: int,
    threshold: int,
) -> Tuple[int, int]:
    """Move point's comp coordinate to target, preferring the nearest dark pixel within tol."""
    def at(value: int) -> Tuple[int, int]:
        return (point[0], value) if comp == 1 else (value, point[1])

    limit = height if comp == 1 else width
    for offset in range(0, tol + 1):
        for candidate in ({target - offset, target + offset} if offset else {target}):
            if not 0 <= candidate < limit:
                continue
            x, y = at(candidate)
            if _is_black(data, bytes_per_line, x, y, threshold):
                return (x, y)
    return at(target)


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

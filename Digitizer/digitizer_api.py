from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from UI import DigitizerWindow


Point = Tuple[int, int]
AxisValues = Tuple[float, float, float, float]
ColorRgb = Tuple[int, int, int]


class DigitizerCliError(RuntimeError):
    """Raised when the CLI wrapper cannot complete a digitization run."""


@dataclass(frozen=True)
class DigitizerOutputs:
    csv_path: str
    overlay_path: str
    point_count: int
    color_rgb: ColorRgb
    axis_values: AxisValues
    tick_points: Tuple[Point, Point, Point, Point]
    used_ocr: bool
    # Populated for the richer CLI output; safe defaults keep older callers working.
    ocr_confidence: Optional[float] = None
    log_path: Optional[str] = None
    elapsed_seconds: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def digitize_image(
    pic_dir: str | Path,
    color_rgb: Optional[Sequence[int]] = None,
    tick_points: Optional[Sequence[Sequence[int | float]]] = None,
    axis_values: Optional[Sequence[int | float]] = None,
    output_dir: str | Path | None = None,
    normalize_y: bool = False,
    limit_to_calibration: bool = True,
    verbose: int = 0,
) -> DigitizerOutputs:
    """Digitize a plot image using the existing GUI algorithms.

    `tick_points` order is x_min, x_max, y_min, y_max in image pixel
    coordinates. `axis_values` order is x_min, x_max, y_min, y_max in plot
    coordinates. `verbose` only changes which extra artifacts are written: at
    `verbose >= 1` a `<image>_log.txt` run record is written alongside the CSV.
    """

    start = time.perf_counter()

    app = _ensure_qt_app()
    _ = app  # Keep a live reference for Qt object lifetime.

    image_path = _resolve_image_path(pic_dir)
    out_dir = _resolve_output_dir(output_dir)

    image = QtGui.QImage(str(image_path))
    if image.isNull():
        raise DigitizerCliError(f"Failed to load image: {image_path}")

    window = DigitizerWindow()
    window.set_image(image)

    requested_color = _normalize_color(color_rgb) if color_rgb is not None else None
    if requested_color is not None:
        _apply_color(window, requested_color)

    if window._selected_color is None:
        raise DigitizerCliError("No color was supplied and auto color selection did not find a usable color.")

    provided_ticks = _normalize_ticks(tick_points) if tick_points is not None else None
    provided_axis = _normalize_axis_values(axis_values) if axis_values is not None else None
    used_ocr = False

    if provided_ticks is None or provided_axis is None:
        used_ocr = True
        _run_axis_detection(window)

    if provided_axis is None:
        provided_axis = _axis_values_from_detection(window)

    _apply_axis_values(window, provided_axis)

    if provided_ticks is not None:
        _apply_manual_ticks(window, provided_ticks)
    else:
        _run_coordinate_calibration(window)

    window._limit_points_to_calib = bool(limit_to_calibration)
    window._active_color_state().limit_points_to_calib = bool(limit_to_calibration)
    window._export_selected_color_indices = {window._active_color_index}
    window._compute_points()

    headers, rows = _build_export_rows(window, normalize_y=normalize_y)
    if not rows:
        raise DigitizerCliError("Digitization completed but produced no points.")

    stem = image_path.stem
    csv_path = out_dir / f"{stem}_digitized_points.csv"
    overlay_path = out_dir / f"{stem}_digitized_overlay.png"

    _write_csv(csv_path, headers, rows)
    _write_overlay(window, overlay_path)

    color_rgb = _normalize_color(window._selected_color)
    tick_points = _current_tick_points(window)
    ocr_confidence = _detection_confidence(window) if used_ocr else None
    elapsed_seconds = round(time.perf_counter() - start, 3)

    log_path: Optional[Path] = None
    if verbose >= 1:
        log_path = out_dir / f"{stem}_log.txt"
        _write_log(
            log_path,
            image_path=image_path,
            color_rgb=color_rgb,
            tick_points=tick_points,
            axis_values=provided_axis,
            used_ocr=used_ocr,
            point_count=len(rows),
            ocr_confidence=ocr_confidence,
            elapsed_seconds=elapsed_seconds,
            csv_path=csv_path,
            overlay_path=overlay_path,
        )

    return DigitizerOutputs(
        csv_path=str(csv_path),
        overlay_path=str(overlay_path),
        point_count=len(rows),
        color_rgb=color_rgb,
        axis_values=provided_axis,
        tick_points=tick_points,
        used_ocr=used_ocr,
        ocr_confidence=ocr_confidence,
        log_path=str(log_path) if log_path is not None else None,
        elapsed_seconds=elapsed_seconds,
    )


def _ensure_qt_app() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(["DataDigitizerCLI"])
    return app


def _resolve_image_path(pic_dir: str | Path) -> Path:
    raw = Path(str(pic_dir)).expanduser()
    candidates = [raw]
    # A bare filename (e.g. "plot2.png") is also looked up in the Downloads folder,
    # so a short one-liner works for images saved there without the full path.
    if len(raw.parts) == 1:
        candidates.append(_downloads_dir() / raw.name)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    first = candidates[0].resolve()
    if first.is_dir():
        raise DigitizerCliError("pic_dir must point to an image file, not a directory.")
    locations = " or ".join(str(c) for c in candidates)
    raise DigitizerCliError(f"Image file not found: {locations}")


def _resolve_output_dir(output_dir: str | Path | None) -> Path:
    # Default output goes to Downloads so a no-options run lands somewhere obvious.
    out_dir = Path(str(output_dir)).expanduser().resolve() if output_dir else _downloads_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _downloads_dir() -> Path:
    downloads = Path.home() / "Downloads"
    return downloads if downloads.is_dir() else Path.home()


def _normalize_color(color_rgb: Sequence[int] | ColorRgb) -> ColorRgb:
    if len(color_rgb) != 3:
        raise DigitizerCliError("color must contain exactly 3 RGB values.")
    values = tuple(int(v) for v in color_rgb)
    if any(v < 0 or v > 255 for v in values):
        raise DigitizerCliError("RGB values must be between 0 and 255.")
    return values  # type: ignore[return-value]


def _normalize_axis_values(axis_values: Sequence[int | float]) -> AxisValues:
    if len(axis_values) != 4:
        raise DigitizerCliError("axis values must be xmin,xmax,ymin,ymax.")
    x_min, x_max, y_min, y_max = (float(v) for v in axis_values)
    if x_min == x_max:
        raise DigitizerCliError("x_min and x_max cannot be equal.")
    if y_min == y_max:
        raise DigitizerCliError("y_min and y_max cannot be equal.")
    return x_min, x_max, y_min, y_max


def _normalize_ticks(tick_points: Sequence[Sequence[int | float]]) -> Tuple[Point, Point, Point, Point]:
    if len(tick_points) != 4:
        raise DigitizerCliError("tick setting must contain 4 points: x_min,x_max,y_min,y_max.")
    normalized: list[Point] = []
    for point in tick_points:
        if len(point) != 2:
            raise DigitizerCliError("each tick point must contain x,y pixel coordinates.")
        normalized.append((int(round(float(point[0]))), int(round(float(point[1])))))
    return normalized[0], normalized[1], normalized[2], normalized[3]


def _apply_color(window: DigitizerWindow, color: ColorRgb) -> None:
    state = window._active_color_state()
    state.color = color
    state.base_points = []
    state.points = []
    window._selected_color = color
    window._base_points = []
    window._points = []
    window._export_selected_color_indices = {window._active_color_index}
    window._restore_active_color_state()


def _run_axis_detection(window: DigitizerWindow) -> None:
    window.run_axis_detection()
    if window._axis_result is None:
        raise DigitizerCliError(window.status_label.text() or "Axis OCR detection failed.")


def _axis_values_from_detection(window: DigitizerWindow) -> AxisValues:
    result = window._axis_result
    if result is None:
        raise DigitizerCliError("Axis OCR detection has not run.")
    values = (result.x_min, result.x_max, result.y_min, result.y_max)
    if any(value is None for value in values):
        raise DigitizerCliError("Axis OCR did not find all x/y min/max values.")
    return values  # type: ignore[return-value]


def _apply_axis_values(window: DigitizerWindow, axis_values: AxisValues) -> None:
    x_min, x_max, y_min, y_max = axis_values
    window.coord_x_min.setText(str(x_min))
    window.coord_x_max.setText(str(x_max))
    window.coord_y_min.setText(str(y_min))
    window.coord_y_max.setText(str(y_max))


def _apply_manual_ticks(window: DigitizerWindow, ticks: Tuple[Point, Point, Point, Point]) -> None:
    keys = ("x_min", "x_max", "y_min", "y_max")
    for key, point in zip(keys, ticks):
        window._manual_points[key] = point
    window._manual_stage = 4
    window._calibration_mode = None
    window._calibration_result = None
    box = window._get_calibration_box()
    window.image_tray.set_calibration_overlays(list(ticks), _box_to_points(box))


def _run_coordinate_calibration(window: DigitizerWindow) -> None:
    result = window._axis_result
    if result is None:
        raise DigitizerCliError("Coordinate calibration requires axis OCR detection.")
    required_points = (result.x_min_point, result.x_max_point, result.y_min_point, result.y_max_point)
    if any(point is None for point in required_points):
        raise DigitizerCliError("Axis OCR did not find all tick point positions needed for calibration.")
    window.run_coordinate_calibration()
    if window._calibration_result is None or window._get_axis_points() is None:
        raise DigitizerCliError(window.status_label.text() or "Coordinate-mediated calibration failed.")


def _build_export_rows(window: DigitizerWindow, normalize_y: bool) -> tuple[list[str], list[list[float]]]:
    axis_values = window._get_axis_values()
    if axis_values is None:
        raise DigitizerCliError("Export requires X/Y min/max values.")
    calibration_box = window._get_calibration_box()
    mapper = window._build_affine_mapper(axis_values)
    if calibration_box is None and mapper is None:
        raise DigitizerCliError("Export requires calibration.")

    if calibration_box is not None:
        left, right, top, bottom = calibration_box
        if right <= left or bottom <= top:
            raise DigitizerCliError("Calibration box is invalid.")

    headers = ["color_slot", "color_r", "color_g", "color_b", "x", "y", "x_px", "y_px"]
    if normalize_y:
        headers.append("y_norm")

    rows: list[list[float]] = []
    selected_indices = sorted(window._export_selected_color_indices or {window._active_color_index})
    x_min_val, x_max_val, y_min_val, y_max_val = axis_values
    _ = x_min_val, x_max_val

    for index in selected_indices:
        state = window._color_states[index]
        if state.color is None:
            continue
        if not state.points:
            window._compute_points_for_state(state)
        color_r, color_g, color_b = state.color
        for pt in state.points:
            x_px = pt.x
            y_px = pt.y
            if mapper is not None:
                x_val, y_val = window._map_pixel_affine(mapper, x_px, y_px)
            else:
                x_val = x_min_val + (x_px - left) * (x_max_val - x_min_val) / (right - left)
                y_val = y_min_val + (bottom - y_px) * (y_max_val - y_min_val) / (bottom - top)
            row: list[float] = [index + 1, color_r, color_g, color_b, x_val, y_val, x_px, y_px]
            if normalize_y:
                denom = y_max_val - y_min_val
                row.append((y_val - y_min_val) / denom if denom else 0.0)
            rows.append(row)

    return headers, rows


def _write_csv(path: Path, headers: list[str], rows: list[list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([round(value, 6) if isinstance(value, float) else value for value in row])


def _write_overlay(window: DigitizerWindow, path: Path) -> None:
    image = window._image
    if image is None or image.isNull():
        raise DigitizerCliError("Cannot render overlay without a loaded image.")
    canvas = image.convertToFormat(QtGui.QImage.Format.Format_ARGB32)
    painter = QtGui.QPainter(canvas)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

    box = window._get_calibration_box()
    if box is not None:
        left, right, top, bottom = box
        pen = QtGui.QPen(QtGui.QColor(0, 120, 255, 210), 2)
        painter.setPen(pen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(0, 120, 255, 30)))
        painter.drawRect(left, top, right - left, bottom - top)

    axis_result = window._axis_result
    if axis_result is not None:
        painter.setPen(QtGui.QPen(QtGui.QColor(255, 145, 0, 170), 1))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 145, 0, 30)))
        for left, top, width, height in axis_result.overlay_rects:
            painter.drawRect(int(left), int(top), int(width), int(height))

    for index, state in enumerate(window._color_states):
        if state.color is None or not state.points:
            continue
        point_color = _inverse_rgb(state.color)
        painter.setPen(QtGui.QPen(QtGui.QColor(*point_color), 2 if index == window._active_color_index else 1))
        painter.setBrush(QtGui.QBrush(QtGui.QColor(*point_color, 170)))
        for pt in state.points:
            painter.drawEllipse(QtCore.QPointF(pt.x, pt.y), 2.2, 2.2)

    painter.end()
    if not canvas.save(str(path), "PNG"):
        raise DigitizerCliError(f"Failed to save overlay PNG: {path}")


def _detection_confidence(window: DigitizerWindow) -> Optional[float]:
    result = getattr(window, "_axis_result", None)
    confidence = getattr(result, "confidence", None)
    if confidence is None:
        return None
    return round(float(confidence), 1)


def _write_log(
    path: Path,
    *,
    image_path: Path,
    color_rgb: ColorRgb,
    tick_points: Tuple[Point, Point, Point, Point],
    axis_values: AxisValues,
    used_ocr: bool,
    point_count: int,
    ocr_confidence: Optional[float],
    elapsed_seconds: float,
    csv_path: Path,
    overlay_path: Path,
) -> None:
    x_min_pt, x_max_pt, y_min_pt, y_max_pt = tick_points
    x_min, x_max, y_min, y_max = axis_values
    axis_source = "OCR" if used_ocr else "provided manually"
    conf_text = f"{ocr_confidence:.1f}%" if ocr_confidence is not None else "n/a (axes not read by OCR)"
    lines = [
        "Data Digitizer 2.12 - run log",
        f"time             : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"elapsed (s)      : {elapsed_seconds:.2f}",
        f"image            : {image_path}",
        "",
        f"color (r,g,b)    : {color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}",
        "pixel coords (px) - tick positions in the image:",
        f"    x_min tick   : {x_min_pt}",
        f"    x_max tick   : {x_max_pt}",
        f"    y_min tick   : {y_min_pt}",
        f"    y_max tick   : {y_max_pt}",
        f"tick -> values   : x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}  ({axis_source})",
        f"OCR confidence   : {conf_text}",
        f"num points       : {point_count}",
        "",
        f"CSV              : {csv_path}",
        f"overlay          : {overlay_path}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _inverse_rgb(color: ColorRgb) -> ColorRgb:
    return 255 - color[0], 255 - color[1], 255 - color[2]


def _box_to_points(box: Optional[Tuple[int, int, int, int]]) -> Optional[list[Point]]:
    if box is None:
        return None
    left, right, top, bottom = box
    return [(left, top), (right, top), (right, bottom), (left, bottom)]


def _current_tick_points(window: DigitizerWindow) -> Tuple[Point, Point, Point, Point]:
    points = window._get_axis_points()
    if points is None:
        raise DigitizerCliError("No calibration tick points are available.")
    normalized = []
    for x, y in points:
        normalized.append((int(round(x)), int(round(y))))
    return normalized[0], normalized[1], normalized[2], normalized[3]

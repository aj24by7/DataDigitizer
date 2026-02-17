from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Optional, Tuple

from PyQt6 import QtGui

try:
    import pytesseract
    from pytesseract import Output
except ImportError:  # pragma: no cover
    pytesseract = None
    Output = None

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
except ImportError:  # pragma: no cover
    Image = None
    ImageEnhance = None
    ImageFilter = None
    ImageOps = None


@dataclass(frozen=True)
class DetectedNumber:
    value: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]


@dataclass(frozen=True)
class AxisScaleResult:
    x_min: Optional[float]
    x_max: Optional[float]
    y_min: Optional[float]
    y_max: Optional[float]
    x_min_point: Optional[Tuple[float, float]]
    x_max_point: Optional[Tuple[float, float]]
    y_min_point: Optional[Tuple[float, float]]
    y_max_point: Optional[Tuple[float, float]]
    overlay_points: List[Tuple[float, float]]
    overlay_rects: List[Tuple[float, float, float, float]]


class AxisDetectionError(RuntimeError):
    pass


def detect_axis_scale(image: QtGui.QImage) -> AxisScaleResult:
    if pytesseract is None or Image is None:
        raise AxisDetectionError("Axis detection requires pytesseract + pillow installed.")
    _configure_tesseract()
    if image.isNull():
        raise AxisDetectionError("No image loaded.")

    pil_image = _qimage_to_pil(image)
    if pil_image is None:
        raise AxisDetectionError("Failed to convert image for OCR.")

    numbers_orig = _extract_numbers(_preprocess_for_ocr(pil_image))

    # Rotated scan (always run to catch vertical labels)
    rotated = pil_image.rotate(-90, expand=True)
    y_numbers_rot = _extract_numbers(_preprocess_for_ocr(rotated))
    numbers_rotated = [_map_rotated_number_to_original(det, pil_image.height) for det in y_numbers_rot]

    all_numbers = numbers_orig + numbers_rotated

    x_cluster = _select_best_cluster(_cluster_by_band(all_numbers, axis="y", tol=10), axis_range="x")
    y_cluster = _select_best_cluster(_cluster_by_band(all_numbers, axis="x", tol=10), axis_range="y")

    x_min_val, x_max_val, x_min_det, x_max_det = _min_max_from_cluster(x_cluster, axis="x")
    y_min_val, y_max_val, y_min_det, y_max_det = _min_max_from_cluster(y_cluster, axis="y")

    overlay_numbers = all_numbers
    overlay_points = [det.center for det in overlay_numbers]
    overlay_rects = [_bbox_to_rect(det.bbox) for det in overlay_numbers]

    return AxisScaleResult(
        x_min=x_min_val,
        x_max=x_max_val,
        y_min=y_min_val,
        y_max=y_max_val,
        x_min_point=x_min_det.center if x_min_det else None,
        x_max_point=x_max_det.center if x_max_det else None,
        y_min_point=y_min_det.center if y_min_det else None,
        y_max_point=y_max_det.center if y_max_det else None,
        overlay_points=overlay_points,
        overlay_rects=overlay_rects,
    )


def _configure_tesseract() -> None:
    if pytesseract is None:
        return
    local_exe = _find_local_tesseract()
    if local_exe is not None:
        pytesseract.pytesseract.tesseract_cmd = str(local_exe)
        return


def _find_local_tesseract() -> Optional[Path]:
    here = Path(__file__).resolve()
    candidates = [
        here.parent / "vendor" / "tesseract" / "tesseract.exe",
        here.parent.parent / "vendor" / "tesseract" / "tesseract.exe",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _qimage_to_pil(image: QtGui.QImage) -> Optional["Image.Image"]:
    if Image is None:
        return None
    if image.format() != QtGui.QImage.Format.Format_RGB888:
        image = image.convertToFormat(QtGui.QImage.Format.Format_RGB888)
    width = image.width()
    height = image.height()
    bytes_per_line = image.bytesPerLine()
    bits = image.bits()
    bits.setsize(bytes_per_line * height)
    data = bytes(bits)
    row_bytes = width * 3
    if bytes_per_line == row_bytes:
        return Image.frombytes("RGB", (width, height), data)
    rows = []
    for y in range(height):
        start = y * bytes_per_line
        rows.append(data[start : start + row_bytes])
    return Image.frombytes("RGB", (width, height), b"".join(rows))


def _preprocess_for_ocr(image: "Image.Image") -> "Image.Image":
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray)
    gray = ImageEnhance.Contrast(gray).enhance(1.6)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=1.2, percent=180, threshold=3))
    return gray


def _extract_numbers(image: "Image.Image") -> List[DetectedNumber]:
    config = "--psm 6 -c tessedit_char_whitelist=0123456789.-+"
    data = pytesseract.image_to_data(image, output_type=Output.DICT, config=config)
    results: list[DetectedNumber] = []
    count = len(data.get("text", []))
    for i in range(count):
        text = data["text"][i].strip()
        if not text:
            continue
        text = text.replace(",", "").replace("−", "-")
        if any(ch.isalpha() for ch in text):
            continue
        value = _parse_number(text)
        if value is None:
            continue
        left = int(data["left"][i])
        top = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])
        if width <= 0 or height <= 0:
            continue
        center = (left + width / 2.0, top + height / 2.0)
        results.append(DetectedNumber(value=value, bbox=(left, top, width, height), center=center))
    return results


def _parse_number(text: str) -> Optional[float]:
    match = re.search(r"[-+]?\d*\.?\d+", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _cluster_by_band(
    numbers: List[DetectedNumber],
    axis: str,
    tol: float,
) -> List[List[DetectedNumber]]:
    if not numbers:
        return []
    key = (lambda n: n.center[0]) if axis == "x" else (lambda n: n.center[1])
    sorted_nums = sorted(numbers, key=key)
    clusters: List[List[DetectedNumber]] = []
    current: List[DetectedNumber] = [sorted_nums[0]]
    current_center = key(sorted_nums[0])
    for det in sorted_nums[1:]:
        pos = key(det)
        if abs(pos - current_center) <= tol:
            current.append(det)
            current_center = sum(key(n) for n in current) / len(current)
        else:
            clusters.append(current)
            current = [det]
            current_center = pos
    clusters.append(current)
    return clusters


def _select_best_cluster(
    clusters: List[List[DetectedNumber]],
    axis_range: str,
) -> List[DetectedNumber]:
    if not clusters:
        return []
    best = clusters[0]
    best_size = len(best)
    best_span = _cluster_span(best, axis_range)
    for cluster in clusters[1:]:
        size = len(cluster)
        span = _cluster_span(cluster, axis_range)
        if size > best_size or (size == best_size and span > best_span):
            best = cluster
            best_size = size
            best_span = span
    return best


def _cluster_span(cluster: List[DetectedNumber], axis: str) -> float:
    if not cluster:
        return 0.0
    if axis == "x":
        positions = [n.center[0] for n in cluster]
    else:
        positions = [n.center[1] for n in cluster]
    return max(positions) - min(positions)


def _min_max_from_cluster(
    cluster: List[DetectedNumber],
    axis: str,
) -> Tuple[Optional[float], Optional[float], Optional[DetectedNumber], Optional[DetectedNumber]]:
    if not cluster:
        return None, None, None, None
    if len(cluster) == 1:
        return cluster[0].value, None, cluster[0], None
    if axis == "x":
        ordered = sorted(cluster, key=lambda n: n.center[0])
        return ordered[0].value, ordered[-1].value, ordered[0], ordered[-1]
    ordered = sorted(cluster, key=lambda n: n.center[1])
    # larger y is lower on screen -> y min at bottom
    return ordered[-1].value, ordered[0].value, ordered[-1], ordered[0]


def _map_rotated_number_to_original(det: DetectedNumber, original_height: int) -> DetectedNumber:
    left, top, width, height = det.bbox
    corners = [
        (left, top),
        (left + width, top),
        (left, top + height),
        (left + width, top + height),
    ]
    mapped = [_rotated_to_original(pt, original_height) for pt in corners]
    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    new_left = int(min(xs))
    new_top = int(min(ys))
    new_width = int(max(xs) - new_left)
    new_height = int(max(ys) - new_top)
    center_rot = det.center
    center_orig = _rotated_to_original(center_rot, original_height)
    return DetectedNumber(
        value=det.value,
        bbox=(new_left, new_top, new_width, new_height),
        center=center_orig,
    )


def _rotated_to_original(point: Tuple[float, float], original_height: int) -> Tuple[float, float]:
    xr, yr = point
    xo = yr
    yo = original_height - 1 - xr
    return xo, yo


def _bbox_to_rect(bbox: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
    left, top, width, height = bbox
    return float(left), float(top), float(width), float(height)

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
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
    conf: float = -1.0


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
    # Mean Tesseract confidence (0-100) of the four numbers used for the axis
    # min/max; None when no usable confidence was reported.
    confidence: Optional[float] = None


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

    try:
        numbers_orig = _extract_numbers(_preprocess_for_ocr(pil_image))

        # Rotated scan (always run to catch vertical labels)
        rotated = pil_image.rotate(-90, expand=True)
        y_numbers_rot = _extract_numbers(_preprocess_for_ocr(rotated))
        numbers_rotated = [_map_rotated_number_to_original(det, pil_image.height) for det in y_numbers_rot]
    except Exception as exc:
        if _is_tesseract_not_found(exc):
            raise AxisDetectionError(_missing_tesseract_message()) from exc
        raise

    all_numbers = numbers_orig + numbers_rotated

    x_cluster = _select_best_cluster(_cluster_by_band(all_numbers, axis="y", tol=10), axis_range="x")
    y_cluster = _select_best_cluster(_cluster_by_band(all_numbers, axis="x", tol=10), axis_range="y")

    x_min_val, x_max_val, x_min_det, x_max_det = _robust_endpoints(x_cluster, axis="x")
    y_min_val, y_max_val, y_min_det, y_max_det = _robust_endpoints(y_cluster, axis="y")

    overlay_numbers = all_numbers
    overlay_points = [det.center for det in overlay_numbers]
    overlay_rects = [_bbox_to_rect(det.bbox) for det in overlay_numbers]

    # De-duplicate by object identity so a detection reused for two of the four
    # axis endpoints (e.g. a single-value axis) is not double-counted in the mean.
    used_dets = list(
        {id(det): det for det in (x_min_det, x_max_det, y_min_det, y_max_det) if det is not None}.values()
    )
    used_confs = [det.conf for det in used_dets if det.conf is not None and det.conf >= 0]
    confidence = (sum(used_confs) / len(used_confs)) if used_confs else None

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
        confidence=confidence,
    )


def _configure_tesseract() -> None:
    if pytesseract is None:
        return
    local_exe = _resolve_tesseract_executable()
    if local_exe is not None:
        pytesseract.pytesseract.tesseract_cmd = str(local_exe)


def _resolve_tesseract_executable() -> Optional[Path]:
    here = Path(__file__).resolve()
    env_cmd = os.environ.get("TESSERACT_CMD", "").strip().strip('"')
    candidates = [
        Path(env_cmd) if env_cmd else None,
        here.parent / "vendor" / "tesseract" / "tesseract.exe",
        here.parent.parent / "vendor" / "tesseract" / "tesseract.exe",
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]
    for path in candidates:
        if path is None:
            continue
        if path.is_file():
            return path
    on_path = shutil.which("tesseract")
    if on_path:
        return Path(on_path)
    return None


def _missing_tesseract_message() -> str:
    return (
        "Tesseract OCR executable not found. Install Tesseract and add it to PATH, "
        "or set TESSERACT_CMD, or place it at vendor/tesseract/tesseract.exe."
    )


def _is_tesseract_not_found(exc: BaseException) -> bool:
    name = type(exc).__name__
    msg = str(exc).lower()
    return name == "TesseractNotFoundError" or "tesseract is not installed" in msg


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
    # Whitelist digits, sign, decimal point, and comma. The comma lets a thousands
    # separator ("1,000") come through as one token instead of being split; it is
    # stripped in _parse_number. (Axis labels are numbers, so no letters are allowed.)
    config = "--psm 6 -c tessedit_char_whitelist=0123456789.-+,"
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
        try:
            conf = float(data.get("conf", [])[i])
        except (IndexError, TypeError, ValueError):
            conf = -1.0
        results.append(
            DetectedNumber(value=value, bbox=(left, top, width, height), center=center, conf=conf)
        )
    return results


def _parse_number(text: str) -> Optional[float]:
    # Drop a thousands separator between digits ("1,000" -> "1000"); accept optional
    # scientific-notation exponent ("1.5e4") if the OCR produced one.
    cleaned = re.sub(r"(?<=\d),(?=\d)", "", text)
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
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
        # Only one label was detected on this axis: report it as the min and
        # leave max unknown. A proper max needs a second calibration point, so
        # we cannot infer it here without broader, riskier changes.
        return cluster[0].value, None, cluster[0], None
    if axis == "x":
        ordered = sorted(cluster, key=lambda n: n.center[0])
        return ordered[0].value, ordered[-1].value, ordered[0], ordered[-1]
    ordered = sorted(cluster, key=lambda n: n.center[1])
    # larger y is lower on screen -> y min at bottom
    return ordered[-1].value, ordered[0].value, ordered[-1], ordered[0]


def _line_inliers(
    positions: List[float],
    values: List[float],
    use_log: bool,
    prefer_sign: int,
) -> Optional[Tuple[int, int, List[int]]]:
    """Find the largest set of ticks whose (pixel position -> read value) lie on one
    straight line (or log line, if use_log), via pair-consensus (RANSAC over every
    pair of ticks). Returns (num_inliers, orientation_matches, inlier_indices) or None.

    `prefer_sign` is the expected sign of value-vs-pixel slope (+1 for X, where value
    grows rightward; -1 for Y, where value grows upward as pixel-y shrinks). It breaks
    ties toward the normal axis orientation, which is what lets a single misread be
    dropped even when only three ticks are present (median-slope fitting can't, because
    one bad tick out of three exceeds its breakdown point).
    """
    import math

    n = len(values)
    if use_log:
        if any(v <= 0 for v in values):
            return None
        ys = [math.log10(v) for v in values]
    else:
        ys = list(values)
    span = max(ys) - min(ys)
    if span <= 0:
        return None
    thr = 0.15 * span  # a misread label lands far outside 15% of the axis span
    best: Optional[Tuple[Tuple[int, int, float], List[int]]] = None
    for i in range(n):
        for j in range(i + 1, n):
            dp = positions[j] - positions[i]
            if abs(dp) < 1e-9:
                continue
            slope = (ys[j] - ys[i]) / dp
            intercept = ys[i] - slope * positions[i]
            resid = [abs(ys[k] - (slope * positions[k] + intercept)) for k in range(n)]
            inliers = [k for k in range(n) if resid[k] <= thr]
            orient = 1 if slope * prefer_sign > 0 else 0
            total_resid = sum(r for r in resid if r <= thr)
            key = (len(inliers), orient, -total_resid)
            if best is None or key > best[0]:
                best = (key, inliers)
    if best is None:
        return None
    (num, orient, _), inliers = best
    return num, orient, inliers


def _robust_endpoints(
    cluster: List[DetectedNumber],
    axis: str,
) -> Tuple[Optional[float], Optional[float], Optional[DetectedNumber], Optional[DetectedNumber]]:
    """Pick the axis min/max from the two extreme *consistent* ticks.

    A tick's value must sit on a straight (or log) line vs its pixel position, so
    a single misread label (e.g. a negative "-5" read as "9", or a decimal misread)
    is rejected as an outlier and the scale is taken from the ticks that agree.
    Falls back to the raw extremes when there are too few ticks to vote (<3) or no
    consistent pair is found. The two returned ticks anchor the calibration scale,
    so dropping a misread endpoint still yields the correct mapping by extrapolation.
    """
    if len(cluster) < 3:
        return _min_max_from_cluster(cluster, axis)
    positions = [(n.center[0] if axis == "x" else n.center[1]) for n in cluster]
    values = [n.value for n in cluster]
    prefer_sign = 1 if axis == "x" else -1
    linear = _line_inliers(positions, values, use_log=False, prefer_sign=prefer_sign)
    logarithmic = _line_inliers(positions, values, use_log=True, prefer_sign=prefer_sign)
    best = linear
    if logarithmic is not None and (best is None or logarithmic[0] > best[0]):
        best = logarithmic
    if best is None:
        return _min_max_from_cluster(cluster, axis)
    inliers = best[2]
    # Only override the raw reading when a clear ~2/3 SUPERMAJORITY of ticks agree on
    # one ladder, so a *minority* of misreads gets dropped. A bare majority is not
    # enough: a few misread labels can coincidentally line up, and when the true
    # values are unrecoverable (e.g. a dropped minus sign, or mostly-misread log
    # labels) there is no trustworthy consensus -- keep the raw extremes rather than
    # risk an even worse guess.
    need = max(3, (2 * len(cluster) + 2) // 3)
    if len(inliers) < need or len(inliers) == len(cluster):
        return _min_max_from_cluster(cluster, axis)
    kept = [cluster[i] for i in sorted(inliers)]
    return _min_max_from_cluster(kept, axis)


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
        conf=det.conf,
    )


def _rotated_to_original(point: Tuple[float, float], original_height: int) -> Tuple[float, float]:
    xr, yr = point
    xo = yr
    yo = original_height - 1 - xr
    return xo, yo


def _bbox_to_rect(bbox: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
    left, top, width, height = bbox
    return float(left), float(top), float(width), float(height)

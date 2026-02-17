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
class OcrBox:
    text: str
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]


def detect_word_masks(image: QtGui.QImage) -> List[Tuple[float, float, float, float]]:
    boxes = _extract_ocr_boxes(image)
    out = []
    for box in boxes:
        if any(ch.isalpha() for ch in box.text):
            out.append(_bbox_to_rect(box.bbox))
    return out


def detect_number_masks(image: QtGui.QImage) -> List[Tuple[float, float, float, float]]:
    boxes = _extract_ocr_boxes(image)
    out = []
    for box in boxes:
        if _is_number(box.text):
            out.append(_bbox_to_rect(box.bbox))
    return out


def detect_legend_mask(image: QtGui.QImage) -> List[Tuple[float, float, float, float]]:
    rect = _find_rect_by_text_cluster(image)
    if rect is None:
        return []
    return [rect]


def _extract_ocr_boxes(image: QtGui.QImage) -> List[OcrBox]:
    if pytesseract is None or Image is None:
        return []
    _configure_tesseract()
    pil_image = _qimage_to_pil(image)
    if pil_image is None:
        return []
    prepped = _preprocess_for_ocr(pil_image)
    config = "--psm 6"
    data = pytesseract.image_to_data(prepped, output_type=Output.DICT, config=config)
    results: List[OcrBox] = []
    count = len(data.get("text", []))
    for i in range(count):
        text = data["text"][i].strip()
        if not text:
            continue
        left = int(data["left"][i])
        top = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])
        if width <= 0 or height <= 0:
            continue
        center = (left + width / 2.0, top + height / 2.0)
        results.append(OcrBox(text=text, bbox=(left, top, width, height), center=center))
    return results


def _find_rect_by_text_cluster(image: QtGui.QImage) -> Optional[Tuple[int, int, int, int]]:
    boxes = _extract_ocr_boxes(image)
    letter_boxes = [b for b in boxes if any(ch.isalpha() for ch in b.text)]
    if not letter_boxes:
        return None
    clusters: List[List[OcrBox]] = []
    max_dist = 60
    for box in letter_boxes:
        placed = False
        for cluster in clusters:
            cx = sum(b.center[0] for b in cluster) / len(cluster)
            cy = sum(b.center[1] for b in cluster) / len(cluster)
            if (box.center[0] - cx) ** 2 + (box.center[1] - cy) ** 2 <= max_dist ** 2:
                cluster.append(box)
                placed = True
                break
        if not placed:
            clusters.append([box])
    if not clusters:
        return None
    best = clusters[0]
    best_count = len(best)
    best_area = _cluster_area(best)
    for cluster in clusters[1:]:
        count = len(cluster)
        area = _cluster_area(cluster)
        if count > best_count or (count == best_count and area < best_area):
            best = cluster
            best_count = count
            best_area = area
    return _cluster_rect(best)


def _cluster_rect(cluster: List[OcrBox], padding: int = 3) -> Tuple[int, int, int, int]:
    lefts = [b.bbox[0] for b in cluster]
    tops = [b.bbox[1] for b in cluster]
    rights = [b.bbox[0] + b.bbox[2] for b in cluster]
    bottoms = [b.bbox[1] + b.bbox[3] for b in cluster]
    left = max(0, min(lefts) - padding)
    top = max(0, min(tops) - padding)
    right = max(rights) + padding
    bottom = max(bottoms) + padding
    return left, top, right - left, bottom - top


def _cluster_area(cluster: List[OcrBox]) -> float:
    left, top, width, height = _cluster_rect(cluster, padding=0)
    return float(width * height)


def _is_number(text: str) -> bool:
    text = text.replace(",", "").replace("?", "-")
    return bool(re.fullmatch(r"[-+]?\d*\.?\d+", text))


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


def _configure_tesseract() -> None:
    if pytesseract is None:
        return
    local_exe = _find_local_tesseract()
    if local_exe is not None:
        pytesseract.pytesseract.tesseract_cmd = str(local_exe)


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


def _bbox_to_rect(bbox: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
    left, top, width, height = bbox
    return float(left), float(top), float(width), float(height)

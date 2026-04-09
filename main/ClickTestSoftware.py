from __future__ import annotations

import csv
import math
import random
import statistics
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from ImageTray import ImageTray


IMAGE_WIDTH = 920
IMAGE_HEIGHT = 620
TILE_WIDTH = 170
TILE_HEIGHT = 208
TILE_COLUMNS = 5
AUTO_ADVANCE_DELAY_MS = 800

POINT_ORDER: list[tuple[str, str]] = [
    ("x_min", "X min"),
    ("x_max", "X max"),
    ("y_min", "Y min"),
    ("y_max", "Y max"),
]
POINT_LABELS = dict(POINT_ORDER)
REQUIRED_ANALYSIS_COLUMNS = {
    "point_name",
    "target_x_px",
    "target_y_px",
    "clicked_x_px",
    "clicked_y_px",
}


@dataclass
class RoundVariation:
    round_index: int
    shift_x_px: int
    shift_y_px: int
    scale_x: float
    scale_y: float
    origin_x: int
    origin_y: int
    x_end_x: int
    y_top_y: int
    targets: dict[str, tuple[int, int]]


@dataclass
class RoundPointRecord:
    point_name: str
    target_x_px: int
    target_y_px: int
    clicked_x_px: int
    clicked_y_px: int
    dx_px: float
    dy_px: float
    distance_px: float


@dataclass
class RoundResult:
    variation: RoundVariation
    points: list[RoundPointRecord]


@dataclass
class AnalysisRow:
    file_path: Path
    file_name: str
    participant: str
    batch_label: str
    session_id: str
    planned_rounds: Optional[int]
    round_index: int
    point_name: str
    target_x_px: float
    target_y_px: float
    clicked_x_px: float
    clicked_y_px: float
    dx_px: float
    dy_px: float
    distance_px: float


@dataclass
class ImportedCsvFile:
    path: Path
    rows: list[AnalysisRow]
    participants: list[str] = field(default_factory=list)
    batches: list[str] = field(default_factory=list)
    sessions: list[str] = field(default_factory=list)

    @property
    def tile_title(self) -> str:
        return self.path.stem

    @property
    def clicks_count(self) -> int:
        return len(self.rows)

    @property
    def rounds_count(self) -> int:
        return _count_unique_rounds(self.rows)

    @property
    def tile_lines(self) -> list[str]:
        lines = []
        if self.participants:
            lines.append("Participant: " + ", ".join(self.participants[:2]))
        if self.batches:
            lines.append("Batch: " + ", ".join(self.batches[:2]))
        if self.sessions:
            session = self.sessions[0]
            if len(session) > 12:
                session = session[:12] + "..."
            lines.append("Session: " + session)
        return lines


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _slugify(text: str) -> str:
    pieces = []
    for char in text.strip():
        if char.isalnum():
            pieces.append(char.lower())
        elif char in {" ", "-", "_"}:
            pieces.append("_")
    slug = "".join(pieces).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "session"


def _parse_float(value: object, field_name: str, path: Path) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{path.name}: missing value for '{field_name}'")
    return float(text)


def _parse_optional_float(value: object) -> Optional[float]:
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _parse_optional_int(value: object) -> Optional[int]:
    text = str(value).strip()
    if not text:
        return None
    return int(float(text))


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * percentile
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return float(ordered[lower])
    fraction = index - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _format_number(value: float) -> str:
    return f"{value:.3f}"


def _count_unique_rounds(rows: list[AnalysisRow]) -> int:
    keys = {(str(row.file_path), row.session_id or row.file_name, row.round_index) for row in rows}
    return len(keys)


def generate_round_variation(session_id: str, round_index: int) -> RoundVariation:
    rng = random.Random(f"{session_id}:{round_index}")

    base_origin_x = 172
    base_origin_y = 516
    base_x_len = 620
    base_y_len = 360
    margin = 54

    raw_shift_x = int(round(rng.uniform(-58.0, 58.0)))
    raw_shift_y = int(round(rng.uniform(-46.0, 46.0)))
    scale_x = round(rng.uniform(0.88, 1.12), 4)
    scale_y = round(rng.uniform(0.88, 1.12), 4)

    x_len = int(round(base_x_len * scale_x))
    y_len = int(round(base_y_len * scale_y))

    min_origin_x = margin
    max_origin_x = IMAGE_WIDTH - margin - x_len
    min_origin_y = margin + y_len
    max_origin_y = IMAGE_HEIGHT - margin

    origin_x = int(round(_clamp(base_origin_x + raw_shift_x, min_origin_x, max_origin_x)))
    origin_y = int(round(_clamp(base_origin_y + raw_shift_y, min_origin_y, max_origin_y)))
    shift_x = origin_x - base_origin_x
    shift_y = origin_y - base_origin_y

    x_end_x = origin_x + x_len
    y_top_y = origin_y - y_len

    targets = {
        "x_min": (origin_x + 28, origin_y),
        "x_max": (x_end_x - 28, origin_y),
        "y_min": (origin_x, origin_y - 34),
        "y_max": (origin_x, y_top_y + 28),
    }
    return RoundVariation(
        round_index=round_index,
        shift_x_px=shift_x,
        shift_y_px=shift_y,
        scale_x=scale_x,
        scale_y=scale_y,
        origin_x=origin_x,
        origin_y=origin_y,
        x_end_x=x_end_x,
        y_top_y=y_top_y,
        targets=targets,
    )


def build_round_image(variation: RoundVariation, active_point: Optional[str]) -> QtGui.QImage:
    image = QtGui.QImage(IMAGE_WIDTH, IMAGE_HEIGHT, QtGui.QImage.Format.Format_ARGB32)
    image.fill(QtGui.QColor("white"))

    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

    axis_pen = QtGui.QPen(QtGui.QColor(0, 0, 0), 3)
    painter.setPen(axis_pen)
    painter.drawLine(variation.origin_x, variation.origin_y, variation.x_end_x, variation.origin_y)
    painter.drawLine(variation.origin_x, variation.origin_y, variation.origin_x, variation.y_top_y)

    axis_tick_pen = QtGui.QPen(QtGui.QColor(0, 0, 0), 2)
    painter.setPen(axis_tick_pen)
    painter.drawLine(variation.origin_x - 8, variation.origin_y, variation.origin_x + 8, variation.origin_y)
    painter.drawLine(variation.origin_x, variation.origin_y - 8, variation.origin_x, variation.origin_y + 8)

    for key, label in POINT_ORDER:
        x, y = variation.targets[key]
        is_active = key == active_point
        cross_color = QtGui.QColor(190, 0, 0) if is_active else QtGui.QColor(0, 0, 0)
        painter.setPen(QtGui.QPen(cross_color, 2))
        painter.drawLine(x - 10, y, x + 10, y)
        painter.drawLine(x, y - 10, x, y + 10)
        if is_active:
            painter.setPen(QtGui.QPen(QtGui.QColor(190, 0, 0), 1))
            painter.setFont(QtGui.QFont("Segoe UI", 10))
            painter.drawText(QtCore.QPointF(x + 12, y - 12), label)

    painter.end()
    return image


def load_click_test_csv(path: Path) -> ImportedCsvFile:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path.name}: CSV has no header row.")
        missing = [field for field in REQUIRED_ANALYSIS_COLUMNS if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path.name}: missing required columns: {', '.join(sorted(missing))}")

        rows: list[AnalysisRow] = []
        participant_values: list[str] = []
        batch_values: list[str] = []
        session_values: list[str] = []

        for raw_row in reader:
            point_name = str(raw_row.get("point_name", "")).strip()
            if not point_name:
                raise ValueError(f"{path.name}: found a row with an empty point_name.")

            target_x = _parse_float(raw_row.get("target_x_px"), "target_x_px", path)
            target_y = _parse_float(raw_row.get("target_y_px"), "target_y_px", path)
            clicked_x = _parse_float(raw_row.get("clicked_x_px"), "clicked_x_px", path)
            clicked_y = _parse_float(raw_row.get("clicked_y_px"), "clicked_y_px", path)

            dx = _parse_optional_float(raw_row.get("dx_px"))
            dy = _parse_optional_float(raw_row.get("dy_px"))
            distance = _parse_optional_float(raw_row.get("distance_px"))
            if dx is None:
                dx = clicked_x - target_x
            if dy is None:
                dy = clicked_y - target_y
            if distance is None:
                distance = math.hypot(dx, dy)

            participant = str(raw_row.get("participant", "")).strip()
            batch_label = str(raw_row.get("batch_label", "")).strip()
            session_id = str(raw_row.get("session_id", "")).strip()
            planned_rounds = _parse_optional_int(raw_row.get("planned_rounds"))
            round_index = _parse_optional_int(raw_row.get("round_index"))
            if round_index is None:
                raise ValueError(f"{path.name}: missing round_index.")

            if participant and participant not in participant_values:
                participant_values.append(participant)
            if batch_label and batch_label not in batch_values:
                batch_values.append(batch_label)
            if session_id and session_id not in session_values:
                session_values.append(session_id)

            rows.append(
                AnalysisRow(
                    file_path=path,
                    file_name=path.name,
                    participant=participant,
                    batch_label=batch_label,
                    session_id=session_id,
                    planned_rounds=planned_rounds,
                    round_index=round_index,
                    point_name=point_name,
                    target_x_px=target_x,
                    target_y_px=target_y,
                    clicked_x_px=clicked_x,
                    clicked_y_px=clicked_y,
                    dx_px=dx,
                    dy_px=dy,
                    distance_px=distance,
                )
            )

    if not rows:
        raise ValueError(f"{path.name}: CSV has no data rows.")

    return ImportedCsvFile(
        path=path,
        rows=rows,
        participants=participant_values,
        batches=batch_values,
        sessions=session_values,
    )


def compute_analysis_metrics(rows: list[AnalysisRow]) -> dict[str, object]:
    distances = [row.distance_px for row in rows]
    dx_values = [row.dx_px for row in rows]
    dy_values = [row.dy_px for row in rows]
    abs_dx = [abs(value) for value in dx_values]
    abs_dy = [abs(value) for value in dy_values]

    point_breakdown: list[dict[str, object]] = []
    for _, label in POINT_ORDER:
        subset = [row for row in rows if row.point_name == label]
        point_breakdown.append(
            {
                "point_name": label,
                "clicks": len(subset),
                "mean_distance": _mean([row.distance_px for row in subset]) if subset else 0.0,
                "median_distance": _median([row.distance_px for row in subset]) if subset else 0.0,
                "p95_distance": _percentile([row.distance_px for row in subset], 0.95) if subset else 0.0,
                "max_distance": max((row.distance_px for row in subset), default=0.0),
            }
        )

    use_participant_groups = any(row.participant for row in rows)
    grouped: dict[str, list[AnalysisRow]] = {}
    batch_map: dict[str, set[str]] = {}
    file_map: dict[str, set[str]] = {}
    for row in rows:
        group_key = row.participant or row.file_name if use_participant_groups else row.file_name
        grouped.setdefault(group_key, []).append(row)
        if row.batch_label:
            batch_map.setdefault(group_key, set()).add(row.batch_label)
        file_map.setdefault(group_key, set()).add(row.file_name)

    group_breakdown: list[dict[str, object]] = []
    for group_key in sorted(grouped):
        subset = grouped[group_key]
        group_distances = [row.distance_px for row in subset]
        group_breakdown.append(
            {
                "group_name": group_key,
                "batches": ", ".join(sorted(batch_map.get(group_key, set()))),
                "files": len(file_map.get(group_key, set())),
                "rounds": _count_unique_rounds(subset),
                "clicks": len(subset),
                "mean_distance": _mean(group_distances),
                "p95_distance": _percentile(group_distances, 0.95),
                "within_5_px": 100.0 * sum(1 for value in group_distances if value <= 5.0) / len(group_distances),
            }
        )

    return {
        "total_rounds": _count_unique_rounds(rows),
        "total_clicks": len(rows),
        "mean_distance": _mean(distances),
        "median_distance": _median(distances),
        "p95_distance": _percentile(distances, 0.95),
        "max_distance": max(distances, default=0.0),
        "mean_abs_dx": _mean(abs_dx),
        "mean_abs_dy": _mean(abs_dy),
        "signed_x_bias": _mean(dx_values),
        "signed_y_bias": _mean(dy_values),
        "within_3_px": 100.0 * sum(1 for value in distances if value <= 3.0) / len(distances),
        "within_5_px": 100.0 * sum(1 for value in distances if value <= 5.0) / len(distances),
        "within_10_px": 100.0 * sum(1 for value in distances if value <= 10.0) / len(distances),
        "point_breakdown": point_breakdown,
        "group_breakdown": group_breakdown,
        "group_mode": "Participant" if use_participant_groups else "File",
    }


class PageTileButton(QtWidgets.QAbstractButton):
    files_dropped = QtCore.pyqtSignal(list)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, accept_drops: bool = False) -> None:
        super().__init__(parent)
        self.setFixedSize(TILE_WIDTH, TILE_HEIGHT)
        self.setAcceptDrops(accept_drops)
        self.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    def _page_rect(self) -> QtCore.QRectF:
        return self.rect().adjusted(12, 10, -12, -10)

    def _page_fold(self) -> float:
        return 28.0

    def _page_title_rect(self, rect: QtCore.QRectF) -> QtCore.QRectF:
        return QtCore.QRectF(rect.left() + 16, rect.top() + 42, rect.width() - 32, 24)

    def _elide_text(self, painter: QtGui.QPainter, text: str, width: float) -> str:
        metrics = painter.fontMetrics()
        return metrics.elidedText(text, QtCore.Qt.TextElideMode.ElideRight, max(10, int(width)))

    def _draw_badge(
        self,
        painter: QtGui.QPainter,
        rect: QtCore.QRectF,
        text: str,
        fill_color: QtGui.QColor,
        text_color: QtGui.QColor,
    ) -> None:
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QBrush(fill_color))
        painter.drawRoundedRect(rect, 7, 7)
        painter.setPen(QtGui.QPen(text_color, 1))
        font = QtGui.QFont("Segoe UI", 8)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(rect, QtCore.Qt.AlignmentFlag.AlignCenter, text)

    def _draw_page_frame(
        self,
        painter: QtGui.QPainter,
        outline_color: QtGui.QColor,
        fill_color: Optional[QtGui.QColor] = None,
        line_width: int = 2,
    ) -> None:
        rect = self._page_rect()
        fold = self._page_fold()
        path = QtGui.QPainterPath()
        path.moveTo(rect.left(), rect.top())
        path.lineTo(rect.right() - fold, rect.top())
        path.lineTo(rect.right(), rect.top() + fold)
        path.lineTo(rect.right(), rect.bottom())
        path.lineTo(rect.left(), rect.bottom())
        path.closeSubpath()

        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        shadow_path = QtGui.QPainterPath(path)
        shadow_transform = QtGui.QTransform()
        shadow_transform.translate(2.0, 3.0)
        shadow_path = shadow_transform.map(shadow_path)
        painter.setPen(QtCore.Qt.PenStyle.NoPen)
        painter.setBrush(QtGui.QColor(0, 0, 0, 18))
        painter.drawPath(shadow_path)

        painter.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255) if fill_color is None else fill_color))
        painter.setPen(QtGui.QPen(outline_color, line_width))
        painter.drawPath(path)

        fold_path = QtGui.QPainterPath()
        fold_path.moveTo(rect.right() - fold, rect.top())
        fold_path.lineTo(rect.right() - fold, rect.top() + fold)
        fold_path.lineTo(rect.right(), rect.top() + fold)
        fold_path.closeSubpath()
        painter.setBrush(QtGui.QBrush(QtGui.QColor(245, 247, 250)))
        painter.drawPath(fold_path)

        painter.setPen(QtGui.QPen(QtGui.QColor(140, 146, 153), 1))
        painter.drawLine(
            QtCore.QPointF(rect.right() - fold, rect.top()),
            QtCore.QPointF(rect.right() - fold, rect.top() + fold),
        )
        painter.drawLine(
            QtCore.QPointF(rect.right() - fold, rect.top() + fold),
            QtCore.QPointF(rect.right(), rect.top() + fold),
        )

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if self.acceptDrops() and self._extract_files(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        if self.acceptDrops() and self._extract_files(event.mimeData()):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        files = self._extract_files(event.mimeData())
        if self.acceptDrops() and files:
            self.files_dropped.emit(files)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def _extract_files(self, mime_data: QtCore.QMimeData) -> list[str]:
        if not mime_data.hasUrls():
            return []
        files = []
        for url in mime_data.urls():
            if url.isLocalFile():
                path = url.toLocalFile()
                if path.lower().endswith(".csv"):
                    files.append(path)
        return files


class ImportTileButton(PageTileButton):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent, accept_drops=True)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        outline = QtGui.QColor(90, 98, 106) if self.underMouse() else QtGui.QColor(124, 132, 141)
        self._draw_page_frame(painter, outline, fill_color=QtGui.QColor(252, 253, 254))

        rect = self._page_rect()
        badge_rect = QtCore.QRectF(rect.left() + 16, rect.top() + 14, 46, 22)
        self._draw_badge(
            painter,
            badge_rect,
            "CSV",
            QtGui.QColor(24, 131, 68),
            QtGui.QColor(255, 255, 255),
        )

        painter.setPen(QtGui.QPen(QtGui.QColor(24, 131, 68), 4))
        cx = rect.center().x()
        cy = rect.center().y() - 2
        painter.drawLine(QtCore.QPointF(cx - 20, cy), QtCore.QPointF(cx + 20, cy))
        painter.drawLine(QtCore.QPointF(cx, cy - 20), QtCore.QPointF(cx, cy + 20))

        painter.setPen(QtGui.QPen(QtGui.QColor(49, 55, 61), 1))
        label_font = QtGui.QFont("Segoe UI", 10)
        label_font.setBold(True)
        painter.setFont(label_font)
        painter.drawText(
            QtCore.QRectF(rect.left() + 12, rect.bottom() - 42, rect.width() - 24, 30),
            QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap,
            "Import CSV",
        )


class CsvFileTile(PageTileButton):
    def __init__(self, imported_file: ImportedCsvFile, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent, accept_drops=False)
        self.setCheckable(True)
        self.setChecked(True)
        self.imported_file = imported_file
        self.setToolTip(str(imported_file.path))

    def update_file(self, imported_file: ImportedCsvFile) -> None:
        self.imported_file = imported_file
        self.setToolTip(str(imported_file.path))
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        checked = self.isChecked()
        fill = QtGui.QColor(255, 255, 255) if checked else QtGui.QColor(248, 249, 250)
        outline = QtGui.QColor(24, 131, 68) if checked else QtGui.QColor(124, 132, 141)
        self._draw_page_frame(painter, outline, fill_color=fill, line_width=2)

        rect = self._page_rect()
        inner = rect.adjusted(14, 12, -14, -12)

        badge_rect = QtCore.QRectF(inner.left(), inner.top(), 46, 22)
        self._draw_badge(
            painter,
            badge_rect,
            "CSV",
            QtGui.QColor(24, 131, 68),
            QtGui.QColor(255, 255, 255),
        )

        state_rect = QtCore.QRectF(inner.right() - 70, inner.top(), 70, 22)
        state_fill = QtGui.QColor(226, 244, 234) if checked else QtGui.QColor(237, 240, 243)
        state_text = "Selected" if checked else "Hidden"
        self._draw_badge(painter, state_rect, state_text, state_fill, QtGui.QColor(49, 55, 61))

        title_rect = QtCore.QRectF(inner.left(), inner.top() + 34, inner.width(), 24)
        title_font = QtGui.QFont("Segoe UI", 10)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QtGui.QPen(QtGui.QColor(18, 20, 23), 1))
        painter.drawText(
            title_rect,
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            self._elide_text(painter, self.imported_file.tile_title, title_rect.width()),
        )

        painter.setFont(QtGui.QFont("Segoe UI", 8))
        painter.setPen(QtGui.QPen(QtGui.QColor(72, 78, 84), 1))
        line_top = title_rect.bottom() + 12
        for line in self.imported_file.tile_lines[:3]:
            painter.drawText(
                QtCore.QRectF(inner.left(), line_top, inner.width(), 22),
                QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
                line,
            )
            line_top += 26

        footer_rect = QtCore.QRectF(inner.left(), rect.bottom() - 40, inner.width(), 28)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(244, 246, 248)))
        painter.setPen(QtGui.QPen(QtGui.QColor(224, 228, 232), 1))
        painter.drawRoundedRect(footer_rect, 6, 6)

        painter.setPen(QtGui.QPen(QtGui.QColor(49, 55, 61), 1))
        footer_font = QtGui.QFont("Segoe UI", 8)
        footer_font.setBold(True)
        painter.setFont(footer_font)
        painter.drawText(
            QtCore.QRectF(footer_rect.left() + 10, footer_rect.top(), footer_rect.width() / 2 - 10, footer_rect.height()),
            QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter,
            f"Clicks {self.imported_file.clicks_count}",
        )
        painter.drawText(
            QtCore.QRectF(footer_rect.center().x(), footer_rect.top(), footer_rect.width() / 2 - 10, footer_rect.height()),
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter,
            f"Rounds {self.imported_file.rounds_count}",
        )


class ClickTestWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Click Test Software")
        self.resize(1180, 860)

        self.session_id = uuid.uuid4().hex[:12]
        self.session_planned_rounds = 10
        self.current_round: Optional[RoundVariation] = None
        self.current_clicks: list[tuple[str, int, int]] = []
        self.completed_rounds: list[RoundResult] = []
        self.awaiting_next_round = False
        self.auto_advance_token = 0

        self.analysis_files: dict[str, ImportedCsvFile] = {}
        self.analysis_tiles: dict[str, CsvFileTile] = {}
        self.analysis_order: list[str] = []

        self._build_ui()
        self.start_new_session()

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        central.setAutoFillBackground(True)
        palette = central.palette()
        palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor(255, 255, 255))
        central.setPalette(palette)
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.setSpacing(6)
        self.export_button = QtWidgets.QPushButton("Export")
        self.export_button.clicked.connect(self.export_results)
        top_bar.addWidget(self.export_button)
        top_bar.addStretch(1)
        root.addLayout(top_bar)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._build_graph_tab(), "Graph Clicking")
        self.tabs.addTab(self._build_analysis_tab(), "Data Analysis")
        root.addWidget(self.tabs, 1)

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        root.addWidget(self.status_label)

    def _build_graph_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(8)
        controls.addWidget(QtWidgets.QLabel("Participant"))
        self.participant_edit = QtWidgets.QLineEdit()
        self.participant_edit.setMaximumWidth(180)
        controls.addWidget(self.participant_edit)

        controls.addWidget(QtWidgets.QLabel("Batch"))
        self.batch_edit = QtWidgets.QLineEdit()
        self.batch_edit.setMaximumWidth(180)
        controls.addWidget(self.batch_edit)

        controls.addWidget(QtWidgets.QLabel("Planned rounds"))
        self.rounds_spin = QtWidgets.QSpinBox()
        self.rounds_spin.setRange(1, 9999)
        self.rounds_spin.setValue(10)
        controls.addWidget(self.rounds_spin)

        self.start_button = QtWidgets.QPushButton("Start New Session")
        self.start_button.clicked.connect(self.start_new_session)
        controls.addWidget(self.start_button)

        self.reset_button = QtWidgets.QPushButton("Reset Current Session")
        self.reset_button.clicked.connect(self.reset_current_session)
        controls.addWidget(self.reset_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.graph_tray = ImageTray()
        self.graph_tray.setMinimumSize(760, 500)
        self.graph_tray.image_clicked.connect(self.on_graph_clicked)
        layout.addWidget(self.graph_tray, 1)

        footer = QtWidgets.QHBoxLayout()
        footer.setSpacing(8)
        self.next_round_button = QtWidgets.QPushButton("Next Round")
        self.next_round_button.setEnabled(False)
        self.next_round_button.clicked.connect(self.advance_to_next_round)
        footer.addWidget(self.next_round_button, 0, QtCore.Qt.AlignmentFlag.AlignLeft)
        footer.addStretch(1)

        self.graph_summary_label = QtWidgets.QLabel("")
        self.graph_summary_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        footer.addWidget(self.graph_summary_label)
        layout.addLayout(footer)
        return tab

    def _build_analysis_tab(self) -> QtWidgets.QWidget:
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        tile_group = QtWidgets.QGroupBox("Imported CSV Files")
        tile_layout = QtWidgets.QVBoxLayout(tile_group)
        tile_layout.setContentsMargins(8, 8, 8, 8)

        self.tile_container = QtWidgets.QWidget()
        self.tile_grid = QtWidgets.QGridLayout(self.tile_container)
        self.tile_grid.setContentsMargins(0, 0, 0, 0)
        self.tile_grid.setHorizontalSpacing(8)
        self.tile_grid.setVerticalSpacing(8)

        self.import_tile = ImportTileButton()
        self.import_tile.clicked.connect(self.browse_analysis_files)
        self.import_tile.files_dropped.connect(self.import_analysis_files)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.tile_container)
        tile_layout.addWidget(scroll)
        layout.addWidget(tile_group, 0)

        metrics_group = QtWidgets.QGroupBox("Aggregate Metrics")
        metrics_layout = QtWidgets.QVBoxLayout(metrics_group)
        metrics_layout.setContentsMargins(8, 8, 8, 8)
        self.summary_text = QtWidgets.QPlainTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setMinimumHeight(170)
        metrics_layout.addWidget(self.summary_text)
        layout.addWidget(metrics_group, 0)

        tables = QtWidgets.QHBoxLayout()
        tables.setSpacing(8)

        point_group = QtWidgets.QGroupBox("By Point")
        point_layout = QtWidgets.QVBoxLayout(point_group)
        point_layout.setContentsMargins(8, 8, 8, 8)
        self.point_table = QtWidgets.QTableWidget(0, 5)
        self.point_table.setHorizontalHeaderLabels(["Point", "Clicks", "Mean Dist", "P95 Dist", "Max Dist"])
        self.point_table.horizontalHeader().setStretchLastSection(True)
        self.point_table.verticalHeader().setVisible(False)
        self.point_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.point_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        point_layout.addWidget(self.point_table)
        tables.addWidget(point_group, 1)

        self.group_group = QtWidgets.QGroupBox("By Participant/File")
        group_layout = QtWidgets.QVBoxLayout(self.group_group)
        group_layout.setContentsMargins(8, 8, 8, 8)
        self.group_table = QtWidgets.QTableWidget(0, 6)
        self.group_table.setHorizontalHeaderLabels(["Group", "Batches", "Files", "Rounds", "Mean Dist", "Within 5px"])
        self.group_table.horizontalHeader().setStretchLastSection(True)
        self.group_table.verticalHeader().setVisible(False)
        self.group_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.group_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        group_layout.addWidget(self.group_table)
        tables.addWidget(self.group_group, 1)

        layout.addLayout(tables, 1)
        self._refresh_analysis_tiles()
        self._refresh_analysis_metrics()
        return tab

    def _set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _current_target_key(self) -> Optional[str]:
        if self.current_round is None:
            return None
        index = len(self.current_clicks)
        if index >= len(POINT_ORDER):
            return None
        return POINT_ORDER[index][0]

    def _current_target_label(self) -> str:
        key = self._current_target_key()
        if key is None:
            return ""
        return POINT_LABELS[key]

    def _requested_planned_rounds(self) -> int:
        return int(self.rounds_spin.value())

    def _planned_rounds(self) -> int:
        return int(self.session_planned_rounds)

    def start_new_session(self) -> None:
        self.session_id = uuid.uuid4().hex[:12]
        self.session_planned_rounds = self._requested_planned_rounds()
        self.completed_rounds = []
        self.current_clicks = []
        self.awaiting_next_round = False
        self.auto_advance_token += 1
        self._start_round(1)
        self._set_status(
            f"Session {self.session_id}: round 1 of {self._planned_rounds()}. Click {self._current_target_label()}."
        )

    def reset_current_session(self) -> None:
        self.session_planned_rounds = self._requested_planned_rounds()
        self.completed_rounds = []
        self.current_clicks = []
        self.awaiting_next_round = False
        self.auto_advance_token += 1
        self._start_round(1)
        self._set_status(
            f"Session {self.session_id} reset. Round 1 of {self._planned_rounds()}. Click {self._current_target_label()}."
        )

    def _start_round(self, round_index: int) -> None:
        self.current_round = generate_round_variation(self.session_id, round_index)
        self.current_clicks = []
        self.awaiting_next_round = False
        self._refresh_graph_view()
        self._refresh_graph_summary()
        self._refresh_next_button()

    def _refresh_graph_view(self) -> None:
        active_point = None if self.awaiting_next_round else self._current_target_key()
        if self.current_round is None:
            image = QtGui.QImage(IMAGE_WIDTH, IMAGE_HEIGHT, QtGui.QImage.Format.Format_ARGB32)
            image.fill(QtGui.QColor("white"))
            self.graph_tray.set_image(image)
            self.graph_tray.set_points([])
            return
        image = build_round_image(self.current_round, active_point)
        self.graph_tray.set_image(image)
        self.graph_tray.set_points([(x, y) for _, x, y in self.current_clicks])

    def _refresh_graph_summary(self) -> None:
        current_round_index = self.current_round.round_index if self.current_round else 0
        completed = len(self.completed_rounds)
        planned = self._planned_rounds()
        click_text = self._current_target_label() if not self.awaiting_next_round else "Round complete"
        self.graph_summary_label.setText(
            f"Session {self.session_id}   Completed {completed} / {planned} planned   Current round {current_round_index}   {click_text}"
        )

    def _refresh_next_button(self) -> None:
        text = "Continue" if len(self.completed_rounds) >= self._planned_rounds() else "Next Round"
        self.next_round_button.setText(text)
        self.next_round_button.setEnabled(self.awaiting_next_round)

    def on_graph_clicked(self, x: int, y: int) -> None:
        if self.current_round is None or self.awaiting_next_round:
            return
        target_key = self._current_target_key()
        if target_key is None:
            return

        self.current_clicks.append((target_key, x, y))
        if len(self.current_clicks) < len(POINT_ORDER):
            self._refresh_graph_view()
            self._refresh_graph_summary()
            self._set_status(
                f"Round {self.current_round.round_index}: recorded {POINT_LABELS[target_key]} at ({x}, {y}). "
                f"Click {self._current_target_label()} next."
            )
            return

        self._finalize_round()

    def _finalize_round(self) -> None:
        if self.current_round is None:
            return

        point_records: list[RoundPointRecord] = []
        for point_key, clicked_x, clicked_y in self.current_clicks:
            target_x, target_y = self.current_round.targets[point_key]
            dx = clicked_x - target_x
            dy = clicked_y - target_y
            point_records.append(
                RoundPointRecord(
                    point_name=POINT_LABELS[point_key],
                    target_x_px=target_x,
                    target_y_px=target_y,
                    clicked_x_px=clicked_x,
                    clicked_y_px=clicked_y,
                    dx_px=dx,
                    dy_px=dy,
                    distance_px=math.hypot(dx, dy),
                )
            )

        self.completed_rounds.append(RoundResult(variation=self.current_round, points=point_records))
        self.awaiting_next_round = True
        self.auto_advance_token += 1
        token = self.auto_advance_token

        completed = len(self.completed_rounds)
        planned = self._planned_rounds()
        self._refresh_graph_view()
        self._refresh_graph_summary()
        self._refresh_next_button()

        if completed < planned:
            self._set_status(
                f"Round {self.current_round.round_index} complete. Auto-advancing to round {completed + 1}. "
                "You can also click Next Round now."
            )
            QtCore.QTimer.singleShot(AUTO_ADVANCE_DELAY_MS, lambda: self._auto_advance_if_current(token))
        else:
            self._set_status(
                f"Round {self.current_round.round_index} complete. Planned rounds reached ({planned}). "
                "Use Continue for more rounds or Export to save the finished rounds."
            )

    def _auto_advance_if_current(self, token: int) -> None:
        if token != self.auto_advance_token:
            return
        if not self.awaiting_next_round:
            return
        if len(self.completed_rounds) >= self._planned_rounds():
            return
        self.advance_to_next_round()

    def advance_to_next_round(self) -> None:
        if not self.awaiting_next_round:
            return
        next_round_index = len(self.completed_rounds) + 1
        self._start_round(next_round_index)
        planned = self._planned_rounds()
        extra = " (manual continuation)" if next_round_index > planned else ""
        self._set_status(f"Round {next_round_index} of {planned}{extra}. Click {self._current_target_label()}.")

    def export_results(self) -> None:
        rows = self._build_export_rows()
        if not rows:
            self._set_status("No completed rounds are available to export yet.")
            return

        participant = _slugify(self.participant_edit.text())
        batch = _slugify(self.batch_edit.text()) if self.batch_edit.text().strip() else "batch"
        default_name = f"click_test_{participant}_{batch}_{self.session_id}.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Click Test CSV",
            default_name,
            "CSV Files (*.csv)",
        )
        if not path:
            return

        file_path = Path(path)
        if file_path.suffix.lower() != ".csv":
            file_path = file_path.with_suffix(".csv")

        headers = [
            "participant",
            "batch_label",
            "session_id",
            "planned_rounds",
            "round_index",
            "point_name",
            "target_x_px",
            "target_y_px",
            "clicked_x_px",
            "clicked_y_px",
            "dx_px",
            "dy_px",
            "distance_px",
            "shift_x_px",
            "shift_y_px",
            "scale_x",
            "scale_y",
        ]
        with file_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

        self._set_status(f"Exported {len(rows)} point rows to {file_path.name}.")

    def _build_export_rows(self) -> list[dict[str, object]]:
        participant = self.participant_edit.text().strip()
        batch_label = self.batch_edit.text().strip()
        planned_rounds = self._planned_rounds()
        export_rows: list[dict[str, object]] = []
        for result in self.completed_rounds:
            for point in result.points:
                export_rows.append(
                    {
                        "participant": participant,
                        "batch_label": batch_label,
                        "session_id": self.session_id,
                        "planned_rounds": planned_rounds,
                        "round_index": result.variation.round_index,
                        "point_name": point.point_name,
                        "target_x_px": point.target_x_px,
                        "target_y_px": point.target_y_px,
                        "clicked_x_px": point.clicked_x_px,
                        "clicked_y_px": point.clicked_y_px,
                        "dx_px": round(point.dx_px, 6),
                        "dy_px": round(point.dy_px, 6),
                        "distance_px": round(point.distance_px, 6),
                        "shift_x_px": result.variation.shift_x_px,
                        "shift_y_px": result.variation.shift_y_px,
                        "scale_x": round(result.variation.scale_x, 6),
                        "scale_y": round(result.variation.scale_y, 6),
                    }
                )
        return export_rows

    def browse_analysis_files(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Import Click Test CSV Files",
            "",
            "CSV Files (*.csv)",
        )
        if paths:
            self.import_analysis_files(paths)

    def import_analysis_files(self, file_paths: list[str]) -> None:
        loaded = 0
        errors: list[str] = []
        for raw_path in file_paths:
            path = Path(raw_path)
            try:
                imported = load_click_test_csv(path)
            except Exception as exc:
                errors.append(str(exc))
                continue

            path_key = str(path.resolve())
            if path_key not in self.analysis_order:
                self.analysis_order.append(path_key)
            self.analysis_files[path_key] = imported

            tile = self.analysis_tiles.get(path_key)
            if tile is None:
                tile = CsvFileTile(imported)
                tile.toggled.connect(self._refresh_analysis_metrics)
                self.analysis_tiles[path_key] = tile
            else:
                keep_checked = tile.isChecked()
                tile.update_file(imported)
                tile.setChecked(keep_checked)
            loaded += 1

        self._refresh_analysis_tiles()
        self._refresh_analysis_metrics()

        if errors:
            QtWidgets.QMessageBox.warning(self, "CSV Import Issues", "\n".join(errors[:8]))
        if loaded:
            self._set_status(f"Imported or refreshed {loaded} CSV file(s) for analysis.")

    def _refresh_analysis_tiles(self) -> None:
        while self.tile_grid.count():
            item = self.tile_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        for index, path_key in enumerate(self.analysis_order):
            tile = self.analysis_tiles.get(path_key)
            if tile is None:
                continue
            row = index // TILE_COLUMNS
            column = index % TILE_COLUMNS
            self.tile_grid.addWidget(tile, row, column)

        plus_index = len(self.analysis_order)
        plus_row = plus_index // TILE_COLUMNS
        plus_column = plus_index % TILE_COLUMNS
        self.tile_grid.addWidget(self.import_tile, plus_row, plus_column)
        self.tile_grid.setRowStretch(plus_row + 1, 1)
        self.tile_grid.setColumnStretch(TILE_COLUMNS, 1)

    def _selected_analysis_rows(self) -> list[AnalysisRow]:
        rows: list[AnalysisRow] = []
        for path_key in self.analysis_order:
            tile = self.analysis_tiles.get(path_key)
            imported = self.analysis_files.get(path_key)
            if tile is None or imported is None or not tile.isChecked():
                continue
            rows.extend(imported.rows)
        return rows

    def _refresh_analysis_metrics(self) -> None:
        selected_rows = self._selected_analysis_rows()
        loaded_files = len(self.analysis_files)
        selected_files = sum(
            1 for key in self.analysis_order if self.analysis_tiles.get(key) and self.analysis_tiles[key].isChecked()
        )

        if not selected_rows:
            self.summary_text.setPlainText(
                f"Loaded files: {loaded_files}\nSelected files: {selected_files}\n\n"
                "Select one or more click-test CSV files to compute accuracy metrics."
            )
            self.point_table.setRowCount(0)
            self.group_table.setRowCount(0)
            self.group_group.setTitle("By Participant/File")
            return

        metrics = compute_analysis_metrics(selected_rows)
        lines = [
            f"Loaded files: {loaded_files}",
            f"Selected files: {selected_files}",
            f"Total rounds: {metrics['total_rounds']}",
            f"Total clicks: {metrics['total_clicks']}",
            "",
            f"Mean Euclidean error: {_format_number(metrics['mean_distance'])} px",
            f"Median Euclidean error: {_format_number(metrics['median_distance'])} px",
            f"P95 Euclidean error: {_format_number(metrics['p95_distance'])} px",
            f"Max Euclidean error: {_format_number(metrics['max_distance'])} px",
            "",
            f"Mean |X error|: {_format_number(metrics['mean_abs_dx'])} px",
            f"Mean |Y error|: {_format_number(metrics['mean_abs_dy'])} px",
            f"Signed X bias: {_format_number(metrics['signed_x_bias'])} px",
            f"Signed Y bias: {_format_number(metrics['signed_y_bias'])} px",
            "",
            f"Within 3 px: {_format_number(metrics['within_3_px'])}%",
            f"Within 5 px: {_format_number(metrics['within_5_px'])}%",
            f"Within 10 px: {_format_number(metrics['within_10_px'])}%",
        ]
        self.summary_text.setPlainText("\n".join(lines))

        point_rows: list[dict[str, object]] = metrics["point_breakdown"]  # type: ignore[assignment]
        self.point_table.setRowCount(len(point_rows))
        for row_index, row in enumerate(point_rows):
            values = [
                row["point_name"],
                str(row["clicks"]),
                _format_number(float(row["mean_distance"])),
                _format_number(float(row["p95_distance"])),
                _format_number(float(row["max_distance"])),
            ]
            for column_index, value in enumerate(values):
                self.point_table.setItem(row_index, column_index, QtWidgets.QTableWidgetItem(value))
        self.point_table.resizeColumnsToContents()

        group_mode = str(metrics["group_mode"])
        self.group_group.setTitle(f"By {group_mode}")
        group_rows: list[dict[str, object]] = metrics["group_breakdown"]  # type: ignore[assignment]
        self.group_table.setRowCount(len(group_rows))
        for row_index, row in enumerate(group_rows):
            values = [
                str(row["group_name"]),
                str(row["batches"]),
                str(row["files"]),
                str(row["rounds"]),
                _format_number(float(row["mean_distance"])),
                _format_number(float(row["within_5_px"])) + "%",
            ]
            for column_index, value in enumerate(values):
                self.group_table.setItem(row_index, column_index, QtWidgets.QTableWidgetItem(value))
        self.group_table.resizeColumnsToContents()


def main() -> None:
    app = QtWidgets.QApplication.instance()
    owns_app = app is None
    if app is None:
        app = QtWidgets.QApplication([])
    window = ClickTestWindow()
    window.show()
    if owns_app:
        app.exec()


if __name__ == "__main__":
    main()

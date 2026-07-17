"""Shared helpers for the batch accuracy pipeline (Part 3).

File matching and pair scoring used by ``accuracy_batch_cli.py`` (a folder of
digitized output vs. a ground-truth folder). All scoring math comes from
``accuracy_core`` — nothing statistical is implemented here.

Matching rule
-------------
Every CSV gets two candidate keys: its *raw* stem (lowercased) and its
*stripped* stem (one known suffix removed, longest match first). The default
suffix list covers this project's conventions::

    _digitized_points  (Data Digitizer CLI/batch output)
    _digitized, _data, _points, _gt, _ground_truth, _groundtruth

A ground-truth file and a digitized file pair up when any of their keys
match, tried in this priority order (first level with a candidate wins):

    1. raw  == raw    (identical stems)
    2. gt raw  == digitized stripped   (e.g. ``sensor_data.csv`` vs.
                                        ``sensor_data_digitized_points.csv`` —
                                        works even when the true stem itself
                                        ends in a strippable word like _data)
    3. gt stripped == digitized raw
    4. stripped == stripped            (both sides carry a suffix)

If one level yields SEVERAL digitized candidates for one ground-truth file,
the pair is reported as ``ambiguous_match`` and not scored — an ambiguous
match is never resolved by guessing. A competitor folder's own suffix
convention is supported by passing extra suffixes (or disabling the defaults)
through ``MatchSettings``. The exact report files the pipelines write
(``batch_report.csv/json``, ``accuracy_report.csv``,
``accuracy_summary.json``) are ignored when indexing a folder — by exact
name, so a legitimate chart called e.g. ``accuracy_curve`` still gets scored —
and every ignored or unmatched file is reported, never silently skipped.

Statuses
--------
ok
    Pair scored; metrics present.
no_method_match
    Ground-truth image with no file in the method folder.
no_ground_truth_match
    Method file whose stem matches no ground-truth image.
ambiguous_match
    Several method files matched one ground-truth image equally well;
    reported with all candidates, never scored by guessing.
parse_error
    A CSV could not be read at all (corrupt, empty, malformed).
invalid_data
    A CSV parsed but yielded no usable numeric series (missing columns, no
    valid rows, bad color slot, or fewer than 2 unique points).
insufficient_overlap
    Both series parsed but share no usable overlapping x-range (or fewer than
    2 comparison points) - reported as failed, never forced to a number.
score_error
    Unexpected failure while computing metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from accuracy_core import comparison_grid, compute_metrics, prepare_series

# Longest suffixes first so "_digitized_points" wins over "_digitized".
DEFAULT_STRIP_SUFFIXES = (
    "_digitized_points",
    "_ground_truth",
    "_groundtruth",
    "_digitized",
    "_points",
    "_data",
    "_gt",
)

# Exact names of the report files the pipelines themselves write; matched by
# exact filename (never by prefix) so real data files are never swallowed.
REPORT_FILE_NAMES = frozenset(
    {"batch_report.csv", "batch_report.json", "accuracy_report.csv", "accuracy_summary.json"}
)

# Column order for metric output; "corr" is exported as "correlation" and the
# four headline stats (rmse, correlation, mse, r2) come first.
METRIC_COLUMNS = (
    ("rmse", "rmse"),
    ("corr", "correlation"),
    ("mse", "mse"),
    ("r2", "r2"),
    ("mae", "mae"),
    ("median_ae", "median_ae"),
    ("p95_ae", "p95_ae"),
    ("max_ae", "max_ae"),
    ("bias", "bias"),
    ("std_residual", "std_residual"),
    ("mape_pct", "mape_pct"),
    ("smape_pct", "smape_pct"),
    ("wape_pct", "wape_pct"),
    ("nrmse_range_pct", "nrmse_range_pct"),
)
METRIC_OUTPUT_NAMES = tuple(out for _, out in METRIC_COLUMNS)


@dataclass(frozen=True)
class MatchSettings:
    strip_suffixes: Tuple[str, ...] = DEFAULT_STRIP_SUFFIXES
    extra_suffixes: Tuple[str, ...] = ()

    def all_suffixes(self) -> Tuple[str, ...]:
        merged = {s.lower() for s in self.strip_suffixes} | {s.lower() for s in self.extra_suffixes}
        return tuple(sorted(merged, key=len, reverse=True))

    def describe(self) -> str:
        suffixes = ", ".join(self.all_suffixes()) or "(none)"
        return (
            "files pair when the ground-truth stem (raw or suffix-stripped, "
            "lowercased) equals the digitized stem (raw or suffix-stripped); "
            "exact raw-stem matches take priority, and a tie between several "
            "candidates is reported as ambiguous_match, never guessed. "
            f"Strippable suffixes (longest first): {suffixes}"
        )


@dataclass(frozen=True)
class ScoreSettings:
    x_col: str = "x"
    y_col: str = "y"
    dup_policy: str = "median"       # GUI default
    grid_mode: str = "original_x"    # GUI default: score at ground-truth x-values
    grid_points: int = 1000          # GUI default (used by common_uniform)
    color_slot: Optional[str] = None  # None = first slot (numeric order), if any


@dataclass(frozen=True)
class FileEntry:
    path: Path
    raw: str    # lowercased raw stem
    norm: str   # raw with one known suffix stripped (== raw when none applies)


@dataclass
class FolderIndex:
    folder: Path
    entries: List[FileEntry] = field(default_factory=list)
    ignored: List[str] = field(default_factory=list)  # pipeline report files


@dataclass
class MatchResult:
    pairs: List[Tuple[str, Path, Path]] = field(default_factory=list)       # (image id, gt, method)
    ambiguous: List[Tuple[str, Path, List[Path]]] = field(default_factory=list)
    gt_unmatched: List[Tuple[str, Path]] = field(default_factory=list)
    method_unmatched: List[Tuple[str, Path]] = field(default_factory=list)


@dataclass
class PairScore:
    status: str
    reason: str = ""
    metrics: Optional[Dict[str, float]] = None
    points_compared: Optional[int] = None
    overlap_start: Optional[float] = None
    overlap_end: Optional[float] = None
    notes: str = ""


def normalize_stem(stem: str, suffixes: Tuple[str, ...]) -> str:
    lowered = stem.lower()
    for suffix in suffixes:
        if lowered.endswith(suffix) and len(lowered) > len(suffix):
            return lowered[: -len(suffix)]
    return lowered


def index_folder(folder: Path, settings: MatchSettings) -> FolderIndex:
    """List every data CSV in ``folder`` with its raw and stripped stems."""
    index = FolderIndex(folder=folder)
    suffixes = settings.all_suffixes()
    for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file() or path.suffix.lower() != ".csv":
            continue
        if path.name.lower() in REPORT_FILE_NAMES:
            index.ignored.append(path.name)
            continue
        raw = path.stem.lower()
        index.entries.append(FileEntry(path=path, raw=raw, norm=normalize_stem(path.stem, suffixes)))
    return index


def match_pairs(gt_index: FolderIndex, method_index: FolderIndex) -> MatchResult:
    """Pair ground-truth and method files per the documented key cascade.

    Deterministic: ground-truth files are visited in sorted order and each
    method file can be claimed once. The image identifier reported for a pair
    is the ground-truth file's raw stem.
    """
    by_raw: Dict[str, List[FileEntry]] = {}
    by_norm: Dict[str, List[FileEntry]] = {}
    for entry in method_index.entries:
        by_raw.setdefault(entry.raw, []).append(entry)
        by_norm.setdefault(entry.norm, []).append(entry)

    result = MatchResult()
    claimed: set = set()

    for gt in sorted(gt_index.entries, key=lambda e: e.raw):
        candidates: List[FileEntry] = []
        for level in (
            by_raw.get(gt.raw, ()),      # 1. raw == raw
            by_norm.get(gt.raw, ()),     # 2. gt raw == method stripped
            by_raw.get(gt.norm, ()),     # 3. gt stripped == method raw
            by_norm.get(gt.norm, ()),    # 4. stripped == stripped
        ):
            candidates = [e for e in level if e.path not in claimed]
            if candidates:
                break
        if not candidates:
            result.gt_unmatched.append((gt.raw, gt.path))
        elif len(candidates) > 1:
            result.ambiguous.append((gt.raw, gt.path, [e.path for e in candidates]))
        else:
            claimed.add(candidates[0].path)
            result.pairs.append((gt.raw, gt.path, candidates[0].path))

    for entry in method_index.entries:
        if entry.path not in claimed:
            result.method_unmatched.append((entry.raw, entry.path))
    return result


class _ParseFailure(Exception):
    """A CSV could not be read at all (pd.read_csv failed)."""


def load_series_frame(path: Path, settings: ScoreSettings) -> Tuple[pd.DataFrame, str]:
    """Read one CSV; if it has a ``color_slot`` column, keep a single slot.

    Digitizer exports can hold several color series in one file. The GUI makes
    the user pick a slot; the batch equivalent picks ``settings.color_slot``
    or, by default, the first slot in numeric order (falling back to string
    order for non-numeric labels) - and says so in the returned note so runs
    are auditable. Raises ``_ParseFailure`` when the file cannot be read at
    all and ``ValueError`` for slot-selection problems.
    """
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # ParserError/EmptyDataError are ValueError subclasses
        raise _ParseFailure(f"{type(exc).__name__}: {exc}") from exc
    note = ""
    if "color_slot" in df.columns:
        slots = sorted(df["color_slot"].dropna().unique(), key=_slot_sort_key)
        if not slots:
            raise ValueError("color_slot column present but has no values")
        if settings.color_slot is not None:
            wanted = [s for s in slots if str(s) == str(settings.color_slot)]
            if not wanted:
                raise ValueError(
                    f"requested color slot {settings.color_slot!r} not in file "
                    f"(available: {', '.join(map(str, slots))})"
                )
            slot = wanted[0]
        else:
            slot = slots[0]
        df = df[df["color_slot"] == slot]
        if len(slots) > 1:
            note = f"file has {len(slots)} color slots; used slot {slot}"
    return df, note


def _slot_sort_key(value) -> Tuple[int, float, str]:
    """Numeric-aware ordering so slot 2 comes before slot 10."""
    try:
        return (0, float(value), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(value))


def score_pair(gt_path: Path, method_path: Path, settings: ScoreSettings) -> PairScore:
    """Score one (ground truth, method output) file pair via accuracy_core."""
    notes: List[str] = []

    def read(path: Path, label: str) -> Optional[pd.DataFrame]:
        try:
            frame, note = load_series_frame(path, settings)
        except _ParseFailure as exc:
            raise _StageFailure("parse_error", f"{label} {path.name}: {exc}") from exc
        except ValueError as exc:  # slot selection problems
            raise _StageFailure("invalid_data", f"{label} {path.name}: {exc}") from exc
        except Exception as exc:
            raise _StageFailure("parse_error", f"{label} {path.name}: {type(exc).__name__}: {exc}") from exc
        if note:
            notes.append(f"{label}: {note}")
        return frame

    try:
        gt_df = read(gt_path, "ground truth")
        method_df = read(method_path, "method output")

        try:
            gt_series = prepare_series(gt_df, settings.x_col, settings.y_col, settings.dup_policy)
            method_series = prepare_series(method_df, settings.x_col, settings.y_col, settings.dup_policy)
        except ValueError as exc:
            raise _StageFailure("invalid_data", str(exc)) from exc

        # accuracy_core's comparison needs at least 2 unique points per side
        # (same floor the GUI enforces before comparing).
        for label, series in (("ground truth", gt_series), ("method output", method_series)):
            if series.x.size < 2:
                raise _StageFailure(
                    "invalid_data",
                    f"{label} has fewer than 2 valid unique points after cleanup",
                )

        try:
            x_cmp, y_ref, y_cmp, meta = comparison_grid(
                gt_series.x,
                gt_series.y,
                method_series.x,
                method_series.y,
                settings.grid_mode,
                settings.grid_points,
            )
        except ValueError as exc:
            raise _StageFailure("insufficient_overlap", str(exc)) from exc

        try:
            metrics = compute_metrics(y_ref, y_cmp)
        except Exception as exc:
            raise _StageFailure("score_error", f"{type(exc).__name__}: {exc}") from exc
    except _StageFailure as failure:
        return PairScore(status=failure.status, reason=failure.reason, notes="; ".join(notes))

    return PairScore(
        status="ok",
        metrics=metrics,
        points_compared=int(x_cmp.size),
        overlap_start=float(meta["overlap_start"]),
        overlap_end=float(meta["overlap_end"]),
        notes="; ".join(notes),
    )


class _StageFailure(Exception):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


def metrics_row(score: PairScore) -> Dict[str, object]:
    """Flatten a PairScore's metrics into output-named columns (empty if failed)."""
    row: Dict[str, object] = {}
    for internal, output in METRIC_COLUMNS:
        value = score.metrics.get(internal) if score.metrics else None
        row[output] = _clean_float(value)
    row["points_compared"] = score.points_compared if score.points_compared is not None else ""
    row["overlap_start"] = _clean_float(score.overlap_start)
    row["overlap_end"] = _clean_float(score.overlap_end)
    return row


def _clean_float(value: Optional[float]) -> object:
    if value is None:
        return ""
    value = float(value)
    if np.isnan(value):
        return "nan"
    return value

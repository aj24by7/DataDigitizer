"""Batch accuracy comparison: score a folder of digitized CSVs against ground truth.

Pairs files between the two folders by filename stem (raw or with one known
suffix such as ``_digitized_points`` / ``_data`` / ``_gt`` stripped - see
``batch_scoring.MatchSettings``; a competitor's own suffix convention can be
added with ``--extra-suffix``), scores every matched pair with the exact
Accuracy Tester math from ``accuracy_core``, and writes:

  accuracy_report.csv    one row per image (matched or not): identifier,
                         every statistic the Accuracy Tester computes
                         (rmse, correlation, mse, r2, mae, ...), points
                         compared, overlap range, status, reason, and the two
                         source filenames. Failed/unmatched/ambiguous rows
                         keep their status + reason and empty statistics -
                         missing means missing, never estimated.
  accuracy_summary.json  totals and the failure rate: images scored ok,
                         matched-but-failed, ambiguous, unmatched on either
                         side, ignored report files, and
                         failure_rate_pct = (1 - ok / total rows) * 100.

Usage:
  py accuracy_batch_cli.py --digitized-dir <folder> --ground-truth-dir <folder>
                           [--output-dir <folder>] [--x-col x --y-col y]
                           [--grid-mode original_x|digitized_x|common_uniform]
                           [--grid-points 1000] [--dup-policy median|mean|first]
                           [--color-slot N] [--extra-suffix _mysuffix ...]
                           [--no-default-suffixes]

Defaults mirror the Accuracy Tester GUI: grid mode ``original_x`` (score at
the ground-truth x-values), 1000 grid points, duplicate-x policy ``median``,
columns ``x`` / ``y``.

Exit codes: 0 = run completed (even if some images failed - see the report),
2 = bad input (missing folder / no CSVs / bad option), 1 = unexpected fatal error.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from batch_scoring import (
    METRIC_OUTPUT_NAMES,
    MatchSettings,
    PairScore,
    ScoreSettings,
    index_folder,
    match_pairs,
    metrics_row,
    score_pair,
)

DETAIL_COLUMNS = (
    ["image"]
    + list(METRIC_OUTPUT_NAMES)
    + ["points_compared", "overlap_start", "overlap_end", "status", "reason",
       "ground_truth_file", "digitized_file", "notes"]
)


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Filenames can contain characters the Windows ANSI console codepage
    # cannot encode; never let a progress print kill the run or its reports.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    try:
        digitized_dir = _existing_dir(args.digitized_dir, "--digitized-dir")
        ground_truth_dir = _existing_dir(args.ground_truth_dir, "--ground-truth-dir")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    match_settings = MatchSettings(
        strip_suffixes=() if args.no_default_suffixes else MatchSettings().strip_suffixes,
        extra_suffixes=tuple(args.extra_suffix or ()),
    )
    score_settings = ScoreSettings(
        x_col=args.x_col,
        y_col=args.y_col,
        dup_policy=args.dup_policy,
        grid_mode=args.grid_mode,
        grid_points=args.grid_points,
        color_slot=args.color_slot,
    )

    gt_index = index_folder(ground_truth_dir, match_settings)
    dig_index = index_folder(digitized_dir, match_settings)
    if not gt_index.entries:
        print(f"error: no data CSVs found in ground-truth folder {ground_truth_dir}", file=sys.stderr)
        return 2
    if not dig_index.entries:
        print(f"error: no data CSVs found in digitized folder {digitized_dir}", file=sys.stderr)
        return 2

    output_dir = (
        Path(str(args.output_dir)).expanduser().resolve()
        if args.output_dir
        else digitized_dir / "accuracy_report"
    )
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"error: cannot create output folder {output_dir}: {exc}", file=sys.stderr)
        return 2

    print(f"Ground truth : {ground_truth_dir} ({len(gt_index.entries)} data CSVs)")
    print(f"Digitized    : {digitized_dir} ({len(dig_index.entries)} data CSVs)")
    print(f"Matching rule: {match_settings.describe()}")
    for idx in (gt_index, dig_index):
        for name in idx.ignored:
            print(f"  note: ignored pipeline report file {idx.folder.name}\\{name}")

    matches = match_pairs(gt_index, dig_index)
    rows: list[dict] = []
    counts = {"ok": 0, "matched_failed": 0, "ambiguous": 0,
              "gt_unmatched": 0, "digitized_unmatched": 0}

    for image_id, gt_path, dig_path in matches.pairs:
        score = score_pair(gt_path, dig_path, score_settings)
        if score.status == "ok":
            counts["ok"] += 1
        else:
            counts["matched_failed"] += 1
            print(f"  failed: {image_id} [{score.status}] {score.reason}")
        rows.append(_row(image_id, score, gt_path, dig_path,
                         status=score.status, reason=score.reason))

    for image_id, gt_path, candidates in matches.ambiguous:
        counts["ambiguous"] += 1
        names = ", ".join(p.name for p in candidates)
        print(f"  ambiguous: {gt_path.name} matches several digitized files: {names}")
        rows.append(_row(image_id, None, gt_path, None, status="ambiguous_match",
                         reason=f"several digitized files match this ground truth: {names}"))

    for image_id, gt_path in matches.gt_unmatched:
        counts["gt_unmatched"] += 1
        print(f"  unmatched ground truth: {gt_path.name} (no digitized file)")
        rows.append(_row(image_id, None, gt_path, None, status="no_method_match",
                         reason="no digitized file matches this ground-truth stem"))

    for image_id, dig_path in matches.method_unmatched:
        counts["digitized_unmatched"] += 1
        print(f"  unmatched digitized file: {dig_path.name} (no ground truth)")
        rows.append(_row(image_id, None, None, dig_path, status="no_ground_truth_match",
                         reason="no ground-truth file matches this stem"))

    rows.sort(key=lambda r: (r["image"], r["status"]))
    total_images = len(rows)
    failure_rate = (1.0 - counts["ok"] / total_images) * 100.0 if total_images else 0.0

    report_csv = output_dir / "accuracy_report.csv"
    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "ground_truth_dir": str(ground_truth_dir),
        "digitized_dir": str(digitized_dir),
        "matching_rule": match_settings.describe(),
        "score_settings": {
            "x_col": score_settings.x_col,
            "y_col": score_settings.y_col,
            "dup_policy": score_settings.dup_policy,
            "grid_mode": score_settings.grid_mode,
            "grid_points": score_settings.grid_points,
            "color_slot": score_settings.color_slot,
        },
        "total_images": total_images,
        "scored_ok": counts["ok"],
        "matched_but_failed": counts["matched_failed"],
        "ambiguous_matches": counts["ambiguous"],
        "ground_truth_unmatched": counts["gt_unmatched"],
        "digitized_unmatched": counts["digitized_unmatched"],
        "ignored_report_files": {
            "ground_truth_dir": gt_index.ignored,
            "digitized_dir": dig_index.ignored,
        },
        "failure_rate_pct": round(failure_rate, 2),
        "failure_rate_definition": (
            "percentage of all report rows (matched pairs + ambiguous + "
            "unmatched on either side) that did not produce an ok score"
        ),
    }
    summary_json = output_dir / "accuracy_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"Scored {counts['ok']}/{total_images} images ok; "
          f"{counts['matched_failed']} failed, {counts['ambiguous']} ambiguous, "
          f"{counts['gt_unmatched'] + counts['digitized_unmatched']} unmatched "
          f"(failure rate {failure_rate:.1f}%).")
    print(f"Report  -> {report_csv}")
    print(f"Summary -> {summary_json}")
    return 0


def _row(image_id, score, gt_path, dig_path, status: str, reason: str) -> dict:
    row = {"image": image_id, "status": status, "reason": reason,
           "ground_truth_file": gt_path.name if gt_path else "",
           "digitized_file": dig_path.name if dig_path else "",
           "notes": score.notes if score else ""}
    row.update(metrics_row(score if score is not None else PairScore(status=status)))
    return row


def _existing_dir(value: Optional[str], flag: str) -> Path:
    if not value:
        raise ValueError(f"{flag} is required")
    path = Path(str(value)).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"{flag} folder not found: {path}")
    if not path.is_dir():
        raise ValueError(f"{flag} must be a folder: {path}")
    return path


def _grid_points_type(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}")
    if value < 2:
        raise argparse.ArgumentTypeError("grid points must be >= 2 (matching the GUI's validation)")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="accuracy-batch",
        description=(
            "Score every digitized CSV in a folder against its ground-truth CSV, "
            "using the Accuracy Tester's exact statistics (accuracy_core). Files "
            "pair up by filename stem (raw, or with one known suffix stripped); "
            "unmatched or ambiguous files are reported, never silently skipped."
        ),
        epilog=(
            "Examples:\n"
            "  py accuracy_batch_cli.py --digitized-dir C:\\out --ground-truth-dir C:\\truth\n"
            "  py accuracy_batch_cli.py --digitized-dir C:\\out --ground-truth-dir C:\\truth "
            "--grid-mode common_uniform --grid-points 500\n"
            "accuracy_report.csv holds one row per image with every statistic and a "
            "status column; accuracy_summary.json holds the failure rate."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--digitized-dir", required=True,
                        help="Folder of digitized CSVs (e.g. the batch digitizer's output).")
    parser.add_argument("--ground-truth-dir", required=True,
                        help="Folder of ground-truth CSVs.")
    parser.add_argument("--output-dir", "--out", "-o", dest="output_dir",
                        help="Report folder (created if absent). Default: <digitized-dir>\\accuracy_report.")
    parser.add_argument("--x-col", default="x", help="X column name in both CSVs (default: x).")
    parser.add_argument("--y-col", default="y", help="Y column name in both CSVs (default: y).")
    parser.add_argument("--grid-mode", default="original_x",
                        choices=("original_x", "digitized_x", "common_uniform"),
                        help="Comparison grid (default original_x = score at ground-truth x-values).")
    parser.add_argument("--grid-points", type=_grid_points_type, default=1000,
                        help="Grid size for common_uniform mode, minimum 2 (default 1000).")
    parser.add_argument("--dup-policy", default="median", choices=("median", "mean", "first"),
                        help="How duplicate x-values collapse (default median).")
    parser.add_argument("--color-slot", default=None,
                        help="Color slot to use when a CSV holds several series "
                             "(default: first slot in numeric order, noted in the report).")
    parser.add_argument("--extra-suffix", action="append",
                        help="Additional filename suffix to strip when matching (repeatable).")
    parser.add_argument("--no-default-suffixes", action="store_true",
                        help="Disable the built-in suffix list; match on exact stems "
                             "(plus any --extra-suffix values).")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

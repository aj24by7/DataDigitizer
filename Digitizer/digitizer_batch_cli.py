"""Batch front-end for the Data Digitizer: digitize every chart image in a folder.

Calls ``digitize_image`` from ``digitizer_api`` once per image inside one process
(the QApplication is created once and reused). By default the outputs land in TWO
subfolders created inside the input folder, so it stays organised and each output
is named after its source image:

    <input-folder>/
        digitized_csvs/      <stem>_digitized_points.csv   (one per image)
        digitized_overlays/  <stem>_digitized_overlay.png  (one per image)
        batch_report.csv     per-image status + failure reasons
        batch_report.json    totals + failure rate + the same per-image records

One bad file never stops the run: a corrupt image, a non-plot photo, an OCR
failure, or a file that is not an image at all is recorded and the batch moves on.

Usage (also reachable as `py digitizer.py <folder>` or `py digitizer.py batch <folder>`):
  py digitizer_batch_cli.py C:\\charts
  py digitizer_batch_cli.py --input-dir C:\\charts --output-dir D:\\results

Exit codes: 0 = batch ran (check the report for per-file failures),
2 = bad input (missing/empty folder, bad option), 1 = unexpected fatal error.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

from digitizer_api import DigitizerCliError, digitize_image
from digitizer_cli import parse_numbers, parse_points, parse_rgb

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
CSV_SUBFOLDER = "digitized_csvs"
OVERLAY_SUBFOLDER = "digitized_overlays"


@dataclass
class ImageRecord:
    image: str
    status: str  # "ok" | "failed" | "skipped"
    reason: str  # empty when ok
    points: Optional[int]
    csv: str
    overlay: str
    elapsed_seconds: float


def main(argv: Optional[Sequence[str]] = None) -> int:
    # Filenames can contain characters the Windows ANSI console codepage cannot
    # encode; never let a progress print kill the run or its reports.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    parser = build_parser()
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    input_dir_arg = args.input_dir_option or args.input_dir
    if not input_dir_arg:
        parser.error("provide an image folder as a positional argument or with --input-dir")

    try:
        input_dir = Path(str(input_dir_arg)).expanduser().resolve()
        if not input_dir.exists():
            raise DigitizerCliError(f"Input folder not found: {input_dir}")
        if not input_dir.is_dir():
            raise DigitizerCliError(f"input must be a folder, not a file: {input_dir}")

        # Where the two output subfolders live: inside the input folder by default,
        # or inside --output-dir if given.
        output_parent = (
            Path(str(args.output_dir)).expanduser().resolve()
            if args.output_dir
            else input_dir
        )
        csv_dir = output_parent / CSV_SUBFOLDER
        overlay_dir = output_parent / OVERLAY_SUBFOLDER

        # Every non-directory file in the folder (top level only). Supported image
        # types are digitized; anything else is reported as skipped, never silently
        # ignored. The output subfolders themselves are directories, so they are
        # not picked up here.
        all_files = sorted(
            (p for p in input_dir.iterdir() if p.is_file()),
            key=lambda p: p.name.lower(),
        )
        images = [p for p in all_files if p.suffix.lower() in SUPPORTED_EXTENSIONS]
        others = [p for p in all_files if p.suffix.lower() not in SUPPORTED_EXTENSIONS
                  and p.name.lower() not in {"batch_report.csv", "batch_report.json"}]
        if not images:
            raise DigitizerCliError(
                f"No supported images found in {input_dir} "
                f"(looked for: {', '.join(SUPPORTED_EXTENSIONS)})."
            )

        try:
            csv_dir.mkdir(parents=True, exist_ok=True)
            overlay_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise DigitizerCliError(f"Cannot create output folders under {output_parent}: {exc}") from exc

        color = parse_rgb(args.color)
        ticks = parse_points(args.ticks)
        axis_values = parse_numbers(args.axis_values, expected=4, name="axis-values")
    except DigitizerCliError as exc:
        print(f"digitizer error: {exc}", file=sys.stderr)
        return 2

    verbose = int(args.verbose or 0)
    records = run_batch(
        images,
        csv_dir=csv_dir,
        overlay_dir=overlay_dir,
        log_dir=output_parent,
        color=color,
        ticks=ticks,
        axis_values=axis_values,
        log_x=args.log_x,
        log_y=args.log_y,
        normalize_y=args.normalize_y,
        limit_to_calibration=args.limit_to_calibration,
        verbose=verbose,
    )
    for other in others:
        records.append(ImageRecord(other.name, "skipped", "not a supported image type",
                                   None, "", "", 0.0))

    report_csv, report_json, summary = write_reports(records, output_parent)
    print()
    print(f"Batch complete: {summary['succeeded']}/{summary['attempted']} images digitized, "
          f"{summary['failed']} failed"
          + (f", {summary['skipped']} non-image files skipped" if summary['skipped'] else "")
          + f" (failure rate {summary['failure_rate_pct']:.1f}%).")
    print(f"CSVs     -> {csv_dir}")
    print(f"Overlays -> {overlay_dir}")
    print(f"Report   -> {report_csv.name}, {report_json.name}  (in {output_parent})")
    return 0


def run_batch(
    images: Sequence[Path],
    *,
    csv_dir: Path,
    overlay_dir: Path,
    log_dir: Path,
    color,
    ticks,
    axis_values,
    log_x: bool,
    log_y: bool,
    normalize_y: bool,
    limit_to_calibration: bool,
    verbose: int,
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    seen_stems: dict[str, str] = {}
    total = len(images)

    for index, image_path in enumerate(images, start=1):
        label = image_path.name
        print(f"[{index}/{total}] {label} ...", end=" ", flush=True)

        # digitize_image names outputs by stem, so two images with the same stem
        # (chart.png + chart.jpg, or Chart.png + chart.jpg on a case-insensitive
        # filesystem) would collide. Guard case-insensitively.
        stem_key = image_path.stem.lower()
        prior = seen_stems.get(stem_key)
        if prior is not None:
            reason = (f"duplicate filename stem '{image_path.stem}' (already produced by {prior}); "
                      "rename the file to digitize it")
            print("FAILED (duplicate stem)")
            records.append(ImageRecord(label, "failed", reason, None, "", "", 0.0))
            continue

        # digitize_image writes the CSV and the overlay into one folder; we point it
        # at the CSV folder, then move the overlay (and any log) into place. Snapshot
        # pre-existing outputs so a mid-write failure only removes what this attempt made.
        csv_out = csv_dir / f"{image_path.stem}_digitized_points.csv"
        overlay_tmp = csv_dir / f"{image_path.stem}_digitized_overlay.png"
        log_tmp = csv_dir / f"{image_path.stem}_log.txt"
        preexisting = {p for p in (csv_out, overlay_tmp, log_tmp) if p.exists()}

        start = time.perf_counter()
        try:
            result = digitize_image(
                pic_dir=image_path,
                color_rgb=color,
                tick_points=ticks,
                axis_values=axis_values,
                output_dir=csv_dir,
                log_x=log_x,
                log_y=log_y,
                normalize_y=normalize_y,
                limit_to_calibration=limit_to_calibration,
                verbose=verbose,
            )
        except DigitizerCliError as exc:
            elapsed = round(time.perf_counter() - start, 3)
            _remove_partial(( csv_out, overlay_tmp, log_tmp), preexisting)
            print(f"FAILED ({exc})")
            records.append(ImageRecord(label, "failed", str(exc), None, "", "", elapsed))
            continue
        except Exception as exc:  # a corrupt/non-plot file must not kill the batch
            elapsed = round(time.perf_counter() - start, 3)
            _remove_partial((csv_out, overlay_tmp, log_tmp), preexisting)
            reason = f"{type(exc).__name__}: {exc}"
            print(f"FAILED ({reason})")
            records.append(ImageRecord(label, "failed", reason, None, "", "", elapsed))
            continue

        # Move the overlay into the overlays subfolder (and the log to the parent),
        # so the two subfolders stay pure (CSVs / overlay images).
        final_overlay = overlay_dir / overlay_tmp.name
        _move(overlay_tmp, final_overlay)
        if log_tmp.exists():
            _move(log_tmp, log_dir / log_tmp.name)

        seen_stems[stem_key] = label
        print(f"ok ({result.point_count} points, {result.elapsed_seconds:.1f}s)")
        records.append(
            ImageRecord(label, "ok", "", result.point_count,
                        str(csv_out), str(final_overlay), result.elapsed_seconds)
        )

    return records


def _remove_partial(paths: Sequence[Path], preexisting: set) -> None:
    for path in paths:
        if path in preexisting:
            continue
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass  # best-effort; the report row already says failed


def _move(src: Path, dst: Path) -> None:
    try:
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
    except OSError:
        pass  # non-fatal; the CSV is the primary artifact


def write_reports(records: Sequence[ImageRecord], output_parent: Path):
    attempted = sum(1 for r in records if r.status in ("ok", "failed"))
    succeeded = sum(1 for r in records if r.status == "ok")
    failed = sum(1 for r in records if r.status == "failed")
    skipped = sum(1 for r in records if r.status == "skipped")
    summary = {
        "attempted": attempted,
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "failure_rate_pct": round(failed / attempted * 100.0, 2) if attempted else 0.0,
    }

    report_csv = output_parent / "batch_report.csv"
    with report_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image", "status", "points", "csv", "overlay", "elapsed_seconds", "reason"])
        for r in records:
            writer.writerow([r.image, r.status, r.points if r.points is not None else "",
                             r.csv, r.overlay, r.elapsed_seconds, r.reason])

    report_json = output_parent / "batch_report.json"
    report_json.write_text(
        json.dumps({**summary, "images": [asdict(r) for r in records]}, indent=2),
        encoding="utf-8",
    )
    return report_csv, report_json, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="digitizer-batch",
        description=(
            "Digitize every chart image in a folder. Outputs go into two subfolders "
            "inside the input folder -- digitized_csvs/ and digitized_overlays/ -- each "
            "file named after its source image. Per-file failures (corrupt files, "
            "non-plot images, OCR failures) are recorded in the batch report and never "
            "stop the run; non-image files are reported as skipped."
        ),
        epilog=(
            "Examples:\n"
            "  py digitizer_batch_cli.py C:\\charts\n"
            "  py digitizer_batch_cli.py --input-dir C:\\charts --output-dir D:\\results\n"
            "  py digitizer.py C:\\charts          (same thing via the main CLI)\n"
            f"Supported image types: {', '.join(SUPPORTED_EXTENSIONS)}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input_dir", nargs="?", help="Folder containing the chart images.")
    parser.add_argument("--input-dir", dest="input_dir_option", help="Folder containing the chart images.")
    parser.add_argument(
        "--output-dir", "--out", "-o", dest="output_dir",
        help="Parent folder for the two output subfolders (created if absent). "
             "Default: the input folder itself.",
    )
    parser.add_argument(
        "--color",
        help="RGB color 'r,g,b' applied to EVERY image. Default: per-image auto color pick.",
    )
    parser.add_argument(
        "--ticks", "--tick-setting", "--tick-coordinates", dest="ticks",
        help="Four pixel points x_min,x_max,y_min,y_max applied to EVERY image "
             "(only sensible when all images share a layout). Default: per-image OCR.",
    )
    parser.add_argument(
        "--axis-values", "--axis", dest="axis_values",
        help="Axis values 'xmin,xmax,ymin,ymax' applied to EVERY image. Default: per-image OCR.",
    )
    parser.add_argument("--log-x", dest="log_x", action="store_true",
                        help="Treat the X axis as base-10 logarithmic on every image.")
    parser.add_argument("--log-y", dest="log_y", action="store_true",
                        help="Treat the Y axis as base-10 logarithmic on every image.")
    parser.add_argument(
        "-v", "--verbose", dest="verbose", nargs="?", type=int, const=1, default=0,
        help="Verbosity level. At 1, each image also gets a <stem>_log.txt run record.",
    )
    parser.add_argument("--normalize-y", action="store_true", help="(Optional) Add a y_norm column to each CSV.")
    parser.add_argument(
        "--limit-to-calibration", dest="limit_to_calibration", action="store_true", default=False,
        help="(Optional) Only export points INSIDE the calibration window. Off by default: a tick "
             "misread slightly short would otherwise clip real data off the ends of the curve.",
    )
    parser.add_argument(
        "--no-limit-to-calibration", dest="limit_to_calibration", action="store_false",
        help="(Optional) Keep points outside the calibration window. This is already the default; "
             "the flag is kept so existing scripts keep working.",
    )
    return parser


if __name__ == "__main__":
    raise SystemExit(main())

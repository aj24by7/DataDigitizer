from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

from digitizer_api import DigitizerCliError, digitize_image


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    pic_path = args.pic_dir_option or args.pic_path
    if not pic_path:
        parser.error("provide an image path as a positional argument or with --pic-dir")

    try:
        result = digitize_image(
            pic_dir=pic_path,
            color_rgb=parse_rgb(args.color),
            tick_points=parse_points(args.ticks),
            axis_values=parse_numbers(args.axis_values, expected=4, name="axis-values"),
            output_dir=args.output_dir,
            normalize_y=args.normalize_y,
            limit_to_calibration=args.limit_to_calibration,
        )
    except DigitizerCliError as exc:
        print(f"digitizer error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(result.to_json())
    else:
        print(f"CSV: {result.csv_path}")
        print(f"Overlay: {result.overlay_path}")
        print(f"Points: {result.point_count}")
        print(f"Color RGB: {result.color_rgb}")
        print(f"Used OCR: {result.used_ocr}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="DataDigitizer-2.11 cli",
        description="Digitize an image from the terminal using the existing Data Digitizer algorithms.",
    )
    parser.add_argument("pic_path", nargs="?", help="Path to the image file.")
    parser.add_argument("--pic-dir", dest="pic_dir_option", help="Path to the image file.")
    parser.add_argument(
        "--color",
        help="RGB color as 'r,g,b' or '[r,g,b]'. If omitted, the auto color selector is used.",
    )
    parser.add_argument(
        "--ticks",
        help="Four pixel points in x_min,x_max,y_min,y_max order, e.g. '[10,200],[500,200],[10,200],[10,20]'.",
    )
    parser.add_argument(
        "--axis-values",
        help="Axis values as 'xmin,xmax,ymin,ymax'. If omitted, OCR axis detection is used.",
    )
    parser.add_argument("--output-dir", help="Output directory. Defaults to the image directory.")
    parser.add_argument("--normalize-y", action="store_true", help="Add a y_norm column to the CSV.")
    parser.add_argument(
        "--limit-to-calibration",
        dest="limit_to_calibration",
        action="store_true",
        default=True,
        help="Only export points inside the calibration window. This is the CLI default.",
    )
    parser.add_argument(
        "--no-limit-to-calibration",
        dest="limit_to_calibration",
        action="store_false",
        help="Match the GUI default and export points outside the calibration window too.",
    )
    parser.add_argument("--json", action="store_true", help="Print result metadata as JSON.")
    return parser


def parse_rgb(value: Optional[str]) -> Optional[tuple[int, int, int]]:
    if _is_null(value):
        return None
    numbers = parse_numbers(value, expected=3, name="color")
    rgb = tuple(int(round(v)) for v in numbers)
    if any(v < 0 or v > 255 for v in rgb):
        raise DigitizerCliError("color RGB values must be between 0 and 255.")
    return rgb  # type: ignore[return-value]


def parse_points(value: Optional[str]) -> Optional[list[tuple[float, float]]]:
    if _is_null(value):
        return None
    numbers = parse_numbers(value, expected=8, name="ticks")
    return [
        (numbers[0], numbers[1]),
        (numbers[2], numbers[3]),
        (numbers[4], numbers[5]),
        (numbers[6], numbers[7]),
    ]


def parse_numbers(value: Optional[str], expected: int, name: str) -> Optional[list[float]]:
    if _is_null(value):
        return None
    assert value is not None
    text = value.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        flattened = _flatten(parsed)
        numbers = [float(item) for item in flattened]
    else:
        numbers = [float(item) for item in re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", text)]

    if len(numbers) != expected:
        raise DigitizerCliError(f"{name} expects {expected} numeric values, got {len(numbers)}.")
    return numbers


def _flatten(value: list[object]) -> list[object]:
    out: list[object] = []
    for item in value:
        if isinstance(item, list):
            out.extend(_flatten(item))
        else:
            out.append(item)
    return out


def _is_null(value: Optional[str]) -> bool:
    return value is None or value.strip().lower() in {"", "none", "null"}


if __name__ == "__main__":
    raise SystemExit(main())

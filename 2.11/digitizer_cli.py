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


def interactive_main() -> int:
    print()
    print("Data Digitizer 2.11 Interactive CLI")
    print("Leave optional fields blank and press Enter to use auto/default behavior.")
    print()

    while True:
        try:
            values = _prompt_values()
        except (EOFError, KeyboardInterrupt):
            print("\nCanceled.")
            return 130

        if values is None:
            print("Canceled.")
            return 0

        action = input("Ready. Press Enter to run, type 'edit' to re-enter, or 'q' to quit: ").strip().lower()
        if action in {"q", "quit", "cancel"}:
            print("Canceled.")
            return 0
        if action == "edit":
            print()
            continue

        try:
            result = digitize_image(
                pic_dir=values["pic_dir"],
                color_rgb=parse_rgb(values["color"]),
                tick_points=parse_points(values["ticks"]),
                axis_values=parse_numbers(values["axis_values"], expected=4, name="axis-values"),
                output_dir=values["output_dir"],
                normalize_y=values["normalize_y"],
                limit_to_calibration=values["limit_to_calibration"],
            )
        except DigitizerCliError as exc:
            print(f"digitizer error: {exc}", file=sys.stderr)
            retry = input("Type 'edit' to try again, or press Enter to exit: ").strip().lower()
            if retry == "edit":
                print()
                continue
            return 2
        except Exception as exc:
            print(f"unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

        print()
        print("Done.")
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


def _prompt_values() -> Optional[dict[str, object]]:
    while True:
        pic_dir = _clean_input(input("Plot location: "))
        if pic_dir.lower() in {"q", "quit", "cancel"}:
            return None
        if pic_dir:
            break
        print("Plot location is required.")

    while True:
        color = _clean_input(input("Color RGB [blank/null = auto] (example 255,0,0): "))
        try:
            parse_rgb(color)
            break
        except DigitizerCliError as exc:
            print(f"Invalid color: {exc}")

    while True:
        ticks = _clean_input(
            input("Tick coordinates [blank/null = OCR] as [x,y],[x,y],[x,y],[x,y]: ")
        )
        try:
            parse_points(ticks)
            break
        except DigitizerCliError as exc:
            print(f"Invalid tick coordinates: {exc}")

    while True:
        axis_values = _clean_input(input("Xmin Xmax Ymin Ymax [blank/null = OCR] (example 0,10,0,100): "))
        try:
            parse_numbers(axis_values, expected=4, name="axis-values")
            break
        except DigitizerCliError as exc:
            print(f"Invalid axis values: {exc}")

    output_dir = _clean_input(input("Output directory [blank = image folder]: "))
    normalize_y = _prompt_yes_no("Add normalized Y column? [y/N]: ", default=False)
    limit_to_calibration = _prompt_yes_no("Limit points to calibration window? [Y/n]: ", default=True)

    print()
    print("Summary")
    print(f"  plot location: {pic_dir}")
    print(f"  color: {color or 'auto'}")
    print(f"  tick coordinates: {ticks or 'OCR auto'}")
    print(f"  axis values: {axis_values or 'OCR auto'}")
    print(f"  output directory: {output_dir or 'image folder'}")
    print(f"  normalize y: {'yes' if normalize_y else 'no'}")
    print(f"  limit to calibration: {'yes' if limit_to_calibration else 'no'}")
    print()

    return {
        "pic_dir": pic_dir,
        "color": color,
        "ticks": ticks,
        "axis_values": axis_values,
        "output_dir": output_dir or None,
        "normalize_y": normalize_y,
        "limit_to_calibration": limit_to_calibration,
    }


def _prompt_yes_no(prompt: str, default: bool) -> bool:
    while True:
        value = input(prompt).strip().lower()
        if not value:
            return default
        if value in {"y", "yes"}:
            return True
        if value in {"n", "no"}:
            return False
        print("Please answer y or n.")


def _clean_input(value: str) -> str:
    return value.strip().strip('"').strip("'")


if __name__ == "__main__":
    raise SystemExit(main())

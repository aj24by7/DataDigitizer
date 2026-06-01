from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from digitizer_api import DigitizerCliError, DigitizerOutputs, digitize_image


CALL_NAME = "digitizer_cli"
CALL_SLOT_NAMES = ("pic_dir", "color", "tick_setting", "axis_values", "output_dir")


def digitizer_cli(
    pic_dir: str | Path,
    color: object = None,
    tick_setting: object = None,
    axis_values: object = None,
    output_dir: str | Path | None = None,
    normalize_y: bool = False,
    limit_to_calibration: bool = True,
) -> DigitizerOutputs:
    """Function-style wrapper for one-line CLI/API use."""

    if _is_null_value(pic_dir):
        raise DigitizerCliError("pic_dir is required.")

    return digitize_image(
        pic_dir=pic_dir,
        color_rgb=_coerce_rgb(color),
        tick_points=_coerce_points(tick_setting),
        axis_values=_coerce_axis_values(axis_values),
        output_dir=None if _is_null_value(output_dir) else output_dir,
        normalize_y=bool(normalize_y),
        limit_to_calibration=bool(limit_to_calibration),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if is_function_call_syntax(raw_args):
        return _run_function_call(raw_args)

    parser = build_parser()
    args = parser.parse_args(raw_args)

    pic_path = args.pic_dir_option or args.pic_path
    if not pic_path:
        parser.error("provide an image path as a positional argument or with --pic-dir")

    try:
        result = digitizer_cli(
            pic_dir=pic_path,
            color=args.color,
            tick_setting=args.ticks,
            axis_values=args.axis_values,
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
        _print_standard_result(result)
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
            result = digitizer_cli(
                pic_dir=values["pic_dir"],
                color=values["color"],
                tick_setting=values["ticks"],
                axis_values=values["axis_values"],
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
        _print_standard_result(result)
        return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="DataDigitizer-2.11 cli",
        description="Digitize an image from the terminal using the existing Data Digitizer algorithms.",
        epilog=(
            "Function-call mode:\n"
            "  py 2.11.py \"digitizer_cli(pic_dir='C:/plots/example.png', output_dir='C:/plots/out')\"\n"
            "  py 2.11.py \"digitizer_cli(pic_dir='C:/plots/example.png', , ([10,200],[500,200],[10,200],[10,20]), (0,10,0,100), output_dir='C:/plots/out')\""
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("pic_path", nargs="?", help="Path to the image file.")
    parser.add_argument("--pic-dir", dest="pic_dir_option", help="Path to the image file.")
    parser.add_argument(
        "--color",
        help="RGB color as 'r,g,b' or '[r,g,b]'. If omitted, the auto color selector is used.",
    )
    parser.add_argument(
        "--ticks",
        "--tick-setting",
        "--tick-coordinates",
        dest="ticks",
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


def print_function_call_usage() -> None:
    print(
        "\n".join(
            [
                "Data Digitizer 2.11 CLI",
                "",
                "Use one quoted function-call line:",
                "  py 2.11.py \"digitizer_cli(pic_dir='C:/plots/example.png', output_dir='C:/plots/out')\"",
                "",
                "Full manual inputs:",
                "  py 2.11.py \"digitizer_cli(pic_dir='C:/plots/example.png', color=(255,0,0), tick_setting=([10,200],[500,200],[10,200],[10,20]), axis_values=(0,10,0,100), output_dir='C:/plots/out')\"",
                "",
                "Blank color, ticks, axis values, or output_dir use auto/default behavior:",
                "  py 2.11.py \"digitizer_cli(pic_dir='C:/plots/example.png', , , , output_dir='')\"",
            ]
        )
    )


def is_function_call_syntax(argv: Sequence[str]) -> bool:
    return bool(re.match(rf"^\s*{CALL_NAME}\s*\(", " ".join(argv).strip()))


def _run_function_call(argv: Sequence[str]) -> int:
    try:
        kwargs = parse_function_call(" ".join(argv))
        result = digitizer_cli(**kwargs)
    except DigitizerCliError as exc:
        print(f"digitizer error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"Pipeline done. Output data to {result.csv_path}. Points: {result.point_count}.")
    print(f"Overlapping plot: {result.overlay_path}")
    return 0


def parse_function_call(call_text: str) -> dict[str, Any]:
    text = call_text.strip()
    match = re.match(rf"^{CALL_NAME}\s*\((.*)\)\s*$", text, re.DOTALL)
    if not match:
        raise DigitizerCliError(f"expected {CALL_NAME}(...)")

    values: dict[str, Any] = {}
    for index, raw_part in enumerate(_split_top_level(match.group(1))):
        part = raw_part.strip()
        if not part:
            continue

        name, raw_value = _split_keyword_argument(part)
        if name is None:
            if index >= len(CALL_SLOT_NAMES):
                raise DigitizerCliError(f"too many positional values in {CALL_NAME}(...).")
            values[CALL_SLOT_NAMES[index]] = _parse_call_value(part)
            continue

        canonical_name = _canonical_call_name(name)
        if canonical_name is None:
            raise DigitizerCliError(f"unknown {CALL_NAME}(...) input: {name}")
        values[canonical_name] = _parse_call_value(raw_value)

    pic_dir = values.get("pic_dir")
    if _is_null_value(pic_dir):
        raise DigitizerCliError("pic_dir is required.")

    return {
        "pic_dir": pic_dir,
        "color": values.get("color"),
        "tick_setting": values.get("tick_setting"),
        "axis_values": values.get("axis_values"),
        "output_dir": values.get("output_dir"),
        "normalize_y": _coerce_bool(values.get("normalize_y"), default=False),
        "limit_to_calibration": _coerce_bool(values.get("limit_to_calibration"), default=True),
    }


def _split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    escape = False

    for index, char in enumerate(text):
        if quote is not None:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue

        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                raise DigitizerCliError("unbalanced closing bracket in function-call input.")
        elif char == "," and depth == 0:
            parts.append(text[start:index])
            start = index + 1

    if quote is not None:
        raise DigitizerCliError("unterminated quote in function-call input.")
    if depth != 0:
        raise DigitizerCliError("unbalanced brackets in function-call input.")

    parts.append(text[start:])
    return parts


def _split_keyword_argument(part: str) -> tuple[str | None, str]:
    depth = 0
    quote: str | None = None
    escape = False

    for index, char in enumerate(part):
        if quote is not None:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                quote = None
            continue

        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "=" and depth == 0:
            name = part[:index].strip()
            if not re.match(r"^[A-Za-z_]\w*$", name):
                return None, part
            return name, part[index + 1 :].strip()

    return None, part


def _canonical_call_name(name: str) -> str | None:
    normalized = name.strip().lower()
    aliases = {
        "pic_dir": "pic_dir",
        "pic_path": "pic_dir",
        "plot_location": "pic_dir",
        "image": "pic_dir",
        "image_path": "pic_dir",
        "color": "color",
        "rgb": "color",
        "color_rgb": "color",
        "tick_setting": "tick_setting",
        "ticks": "tick_setting",
        "tick_coordinates": "tick_setting",
        "tick_coordinate": "tick_setting",
        "axis_values": "axis_values",
        "axis": "axis_values",
        "bounds": "axis_values",
        "output_dir": "output_dir",
        "out_dir": "output_dir",
        "normalize_y": "normalize_y",
        "limit_to_calibration": "limit_to_calibration",
    }
    return aliases.get(normalized)


def _parse_call_value(raw_value: str) -> Any:
    value = raw_value.strip()
    if _is_null(value):
        return None
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _print_standard_result(result: DigitizerOutputs) -> None:
    print(f"CSV: {result.csv_path}")
    print(f"Overlay: {result.overlay_path}")
    print(f"Points: {result.point_count}")
    print(f"Color RGB: {result.color_rgb}")
    print(f"Used OCR: {result.used_ocr}")


def _coerce_rgb(value: object) -> Optional[tuple[int, int, int]]:
    if _is_null_value(value):
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != 3:
            raise DigitizerCliError("color expects 3 RGB values.")
        rgb = tuple(int(round(float(item))) for item in value)
        if any(item < 0 or item > 255 for item in rgb):
            raise DigitizerCliError("color RGB values must be between 0 and 255.")
        return rgb  # type: ignore[return-value]
    return parse_rgb(str(value))


def _coerce_points(value: object) -> Optional[list[tuple[float, float]]]:
    if _is_null_value(value):
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 4 and all(isinstance(item, (list, tuple)) for item in value):
            points: list[tuple[float, float]] = []
            for item in value:
                if len(item) != 2:
                    raise DigitizerCliError("each tick point must contain x,y pixel coordinates.")
                points.append((float(item[0]), float(item[1])))
            return points
        if len(value) == 8:
            numbers = [float(item) for item in value]
            return [
                (numbers[0], numbers[1]),
                (numbers[2], numbers[3]),
                (numbers[4], numbers[5]),
                (numbers[6], numbers[7]),
            ]
        raise DigitizerCliError("tick_setting expects [x,y],[x,y],[x,y],[x,y].")
    return parse_points(str(value))


def _coerce_axis_values(value: object) -> Optional[list[float]]:
    if _is_null_value(value):
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != 4:
            raise DigitizerCliError("axis_values expects xmin,xmax,ymin,ymax.")
        return [float(item) for item in value]
    return parse_numbers(str(value), expected=4, name="axis-values")


def _coerce_bool(value: object, default: bool) -> bool:
    if _is_null_value(value):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    raise DigitizerCliError(f"expected a boolean value, got {value!r}.")


def _is_null_value(value: object) -> bool:
    return value is None or (isinstance(value, str) and _is_null(value))


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

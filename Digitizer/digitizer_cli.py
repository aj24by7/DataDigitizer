from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from app_version import APP_TITLE
from digitizer_api import DigitizerCliError, DigitizerOutputs, digitize_image


CALL_NAME = "digitizer_cli"
CALL_SLOT_NAMES = ("pic_dir", "color", "tick_setting", "axis_values", "output_dir")


def digitizer_cli(
    pic_dir: str | Path,
    color: object = None,
    tick_setting: object = None,
    axis_values: object = None,
    output_dir: str | Path | None = None,
    log_x: object = False,
    log_y: object = False,
    normalize_y: bool = False,
    limit_to_calibration: bool = False,
    verbose: object = 0,
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
        log_x=_coerce_bool(log_x, default=False),
        log_y=_coerce_bool(log_y, default=False),
        normalize_y=bool(normalize_y),
        limit_to_calibration=bool(limit_to_calibration),
        verbose=_coerce_verbose(verbose),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if is_function_call_syntax(raw_args):
        return _run_function_call(raw_args)

    parser = build_parser()
    args = parser.parse_args(_detach_nonnumeric_verbose(raw_args))

    pic_path = args.pic_dir_option or args.pic_path
    if not pic_path:
        parser.error("provide an image path as a positional argument or with --pic-dir")

    verbose = _coerce_verbose(args.verbose)
    try:
        result = digitizer_cli(
            pic_dir=pic_path,
            color=args.color,
            tick_setting=args.ticks,
            axis_values=args.axis_values,
            output_dir=args.output_dir,
            log_x=args.log_x,
            log_y=args.log_y,
            normalize_y=args.normalize_y,
            limit_to_calibration=args.limit_to_calibration,
            verbose=verbose,
        )
    except DigitizerCliError as exc:
        print(f"digitizer error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        _report_unexpected(exc, verbose)
        return 1

    _emit_result(result, as_json=args.json, verbose=verbose)
    return 0


def interactive_main() -> int:
    print()
    print(f"{APP_TITLE} Interactive CLI")
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
                log_x=values["log_x"],
                log_y=values["log_y"],
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
        prog="digitizer",
        description=(
            "Digitize a plot image into a CSV (+ overlay). With just an image path it "
            "auto-detects the curve color and the axes (OCR) and saves to your Downloads folder."
        ),
        epilog=(
            "Examples:\n"
            "  py digitizer.py plot2.png\n"
            "  py digitizer.py plot2.png --verbose 1\n"
            "  py digitizer.py plot2.png --color 255,0,0 --axis 0,10,0,100\n"
            "  py digitizer.py plot2.png --ticks \"[10,200],[500,200],[10,200],[10,20]\" --out C:\\out\n"
            "A bare filename is also looked up in your Downloads folder."
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
        "--axis",
        dest="axis_values",
        help="Axis values as 'xmin,xmax,ymin,ymax'. If omitted, OCR axis detection is used.",
    )
    parser.add_argument(
        "--log-x",
        dest="log_x",
        action="store_true",
        help="Treat the X axis as base-10 logarithmic (X min and X max must be positive). Mirrors the GUI's log toggle.",
    )
    parser.add_argument(
        "--log-y",
        dest="log_y",
        action="store_true",
        help="Treat the Y axis as base-10 logarithmic (Y min and Y max must be positive). Mirrors the GUI's log toggle.",
    )
    parser.add_argument(
        "--output-dir",
        "--out",
        "-o",
        dest="output_dir",
        help="Output folder. Default: your Downloads folder.",
    )
    parser.add_argument("--json", action="store_true", help="Print result metadata as JSON.")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        nargs="?",
        type=_verbose_value,
        const=1,
        default=0,
        help=(
            "Verbosity level. 0 (default) prints only success + output folder. "
            "1 (or just -v / --verbose) prints color, pixel coords, tick->OCR values, "
            "point count, and OCR confidence, and writes a <image>_log.txt."
        ),
    )
    # --- Rarely needed extras (kept for power users; safe to ignore) -------------
    parser.add_argument("--normalize-y", action="store_true", help="(Optional) Add a y_norm column to the CSV.")
    parser.add_argument(
        "--limit-to-calibration",
        dest="limit_to_calibration",
        action="store_true",
        default=False,
        help="(Optional) Only export points INSIDE the calibration window. Off by default: a tick "
             "misread slightly short would otherwise clip real data off the ends of the curve.",
    )
    parser.add_argument(
        "--no-limit-to-calibration",
        dest="limit_to_calibration",
        action="store_false",
        help="(Optional) Keep points outside the calibration window. This is already the default; "
             "the flag is kept so existing scripts keep working.",
    )
    return parser


def _verbose_value(token: str) -> object:
    """Type for -v/--verbose: parse a numeric level, else leave the token alone.

    A non-numeric token (e.g. an image path) is normally detached before parsing by
    _detach_nonnumeric_verbose; if one still reaches here we return the implicit level
    1 rather than raising an int-parse error so 'digitizer.py --verbose image.png' is safe.
    """
    text = str(token).strip()
    try:
        return int(text)
    except ValueError:
        return 1


def _detach_nonnumeric_verbose(argv: Sequence[str]) -> list[str]:
    """Insert the implicit level when -v/--verbose is followed by a non-numeric token.

    With nargs='?', argparse would otherwise consume the next token (e.g. the image
    path) as the verbose value. By only letting a numeric token attach to the flag,
    'digitizer.py --verbose image.png' keeps image.png as the positional argument while
    'digitizer.py --verbose 1 image.png' still reads the level.
    """
    args = list(argv)
    out: list[str] = []
    flags = {"-v", "--verbose"}
    for index, token in enumerate(args):
        out.append(token)
        if token in flags:
            following = args[index + 1] if index + 1 < len(args) else None
            if following is None or not _looks_like_verbose_level(following):
                out.append("1")
    return out


def _looks_like_verbose_level(token: str) -> bool:
    try:
        int(str(token).strip())
    except ValueError:
        return False
    return True


def print_function_call_usage() -> None:
    print(
        "\n".join(
            [
                f"{APP_TITLE} CLI",
                "",
                "Use one quoted function-call line:",
                "  py digitizer.py 'digitizer_cli(pic_dir=\"C:\\Users\\User\\Pictures\\Screenshots\\Example 2.png\", output_dir=\"C:\\Users\\User\\Downloads\\testcli\")'",
                "",
                "Full manual inputs:",
                "  py digitizer.py 'digitizer_cli(pic_dir=\"C:\\Users\\User\\Pictures\\Screenshots\\Example 2.png\", color=(255,0,0), tick_setting=([10,200],[500,200],[10,200],[10,20]), axis_values=(0,10,0,100), output_dir=\"C:\\Users\\User\\Downloads\\testcli\")'",
                "",
                "Blank color, ticks, axis values, or output_dir use auto/default behavior:",
                "  py digitizer.py 'digitizer_cli(pic_dir=\"C:\\Users\\User\\Pictures\\Screenshots\\Example 2.png\", , , , output_dir=\"C:\\Users\\User\\Downloads\\testcli\")'",
            ]
        )
    )


def print_template() -> None:
    """Print a ready-to-copy digitizer_cli(...) line that users can edit."""

    print(
        "\n".join(
            [
                "Function-call template - copy a line below, then change the values.",
                "Keep the single quotes around the whole digitizer_cli(...) call.",
                "",
                "Minimal (auto-detect the color, OCR the axes, save to Downloads):",
                "  py digitizer.py 'digitizer_cli(pic_dir=\"plot2.png\")'",
                "",
                "Full template (every option - delete the parts you don't need):",
                "  py digitizer.py 'digitizer_cli(pic_dir=\"plot2.png\", color=(255,0,0), "
                "axis_values=(0,10,0,100), tick_setting=([10,200],[500,200],[10,200],[10,20]), "
                "log_x=False, log_y=False, output_dir=\"C:/Users/User/Downloads/out\", "
                "verbose=1, json=False, normalize_y=False, limit_to_calibration=False)'",
                "",
                "What each value means:",
                "  pic_dir=\"...\"               required - image file (a bare name is looked up in Downloads)",
                "  color=(R,G,B)               curve color, each 0-255; omit to auto-detect",
                "  axis_values=(x0,x1,y0,y1)   axis numbers (xmin,xmax,ymin,ymax); omit to read by OCR",
                "  tick_setting=([x,y] x4)     tick pixel points x_min,x_max,y_min,y_max; omit for OCR",
                "  log_x=True                  read the X axis in base-10 log space (X min/max must be positive)",
                "  log_y=True                  read the Y axis in base-10 log space (Y min/max must be positive)",
                "  output_dir=\"...\"            folder to save into; omit for Downloads",
                "  verbose=1                   show full detail + write a <image>_log.txt (0 = quiet)",
                "  json=True                   print the full result details as JSON",
                "  -- the two below are optional extras; leave them off for normal use --",
                "  normalize_y=True            (optional) add a 0-1 normalized Y column to the CSV",
                "  limit_to_calibration=True   (optional) drop points outside the calibration box (off by default)",
            ]
        )
    )


def is_function_call_syntax(argv: Sequence[str]) -> bool:
    return bool(re.match(rf"^\s*{CALL_NAME}\s*\(", " ".join(argv).strip()))


def _run_function_call(argv: Sequence[str]) -> int:
    verbose = 0
    try:
        kwargs = parse_function_call(" ".join(argv))
        as_json = bool(kwargs.pop("_json", False))
        verbose = _coerce_verbose(kwargs.get("verbose", 0))
        result = digitizer_cli(**kwargs)
    except DigitizerCliError as exc:
        print(f"digitizer error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        _report_unexpected(exc, verbose)
        return 1

    _emit_result(result, as_json=as_json, verbose=verbose)
    return 0


def parse_function_call(call_text: str) -> dict[str, Any]:
    text = call_text.strip()
    match = re.match(rf"^{CALL_NAME}\s*\((.*)\)\s*$", text, re.DOTALL)
    if not match:
        raise DigitizerCliError(f"expected {CALL_NAME}(...)")

    values: dict[str, Any] = {}
    # Track the positional slot separately so it only advances for UNNAMED parts;
    # an empty slot (",,") still consumes a positional slot but contributes no value.
    positional_index = 0
    seen_keyword = False
    for raw_part in _split_top_level(match.group(1)):
        part = raw_part.strip()

        name, raw_value = _split_keyword_argument(part)
        if name is None:
            if seen_keyword and part:
                raise DigitizerCliError("positional value after keyword argument")
            if positional_index >= len(CALL_SLOT_NAMES):
                raise DigitizerCliError(f"too many positional values in {CALL_NAME}(...).")
            slot = CALL_SLOT_NAMES[positional_index]
            positional_index += 1
            if not part:
                continue
            values[slot] = _parse_call_value(part)
            continue

        seen_keyword = True
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
        "log_x": _coerce_bool(values.get("log_x"), default=False),
        "log_y": _coerce_bool(values.get("log_y"), default=False),
        "normalize_y": _coerce_bool(values.get("normalize_y"), default=False),
        "limit_to_calibration": _coerce_bool(values.get("limit_to_calibration"), default=False),
        "verbose": _coerce_verbose(values.get("verbose")),
        # Not a digitizer_cli argument; _run_function_call pops it to pick the output style.
        "_json": _coerce_bool(values.get("json"), default=False),
    }


def _split_top_level(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None

    for index, char in enumerate(text):
        if quote is not None:
            # Value semantics are literal (no unescaping in _parse_call_value), so a
            # backslash never escapes here: only a matching quote char closes the string.
            # This lets quoted Windows paths ending in a backslash parse correctly.
            if char == quote:
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

    for index, char in enumerate(part):
        if quote is not None:
            # Literal value semantics: a backslash never escapes here, only a matching
            # quote char closes the string, so trailing-backslash Windows paths parse.
            if char == quote:
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
        "out": "output_dir",
        "log_x": "log_x",
        "logx": "log_x",
        "x_log": "log_x",
        "log_y": "log_y",
        "logy": "log_y",
        "y_log": "log_y",
        "normalize_y": "normalize_y",
        "normalize": "normalize_y",
        "limit_to_calibration": "limit_to_calibration",
        "limit": "limit_to_calibration",
        "verbose": "verbose",
        "v": "verbose",
        "json": "json",
        "as_json": "json",
        "print_json": "json",
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


def _coerce_verbose(value: object) -> int:
    """Normalize a verbose value (flag, bool, int, or string) to an int level >= 0."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, int):
        return max(0, value)
    text = str(value).strip().lower()
    if text in {"", "none", "null", "false", "no", "off"}:
        return 0
    if text in {"true", "yes", "on"}:
        return 1
    try:
        return max(0, int(float(text)))
    except ValueError:
        raise DigitizerCliError(f"verbose expects a number (0, 1, ...), got {value!r}.")


def _output_paths(result: DigitizerOutputs) -> list[str]:
    paths = [result.csv_path, result.overlay_path]
    if result.log_path:
        paths.append(result.log_path)
    return paths


def _emit_result(result: DigitizerOutputs, as_json: bool, verbose: int) -> None:
    if as_json:
        print(result.to_json())
    elif verbose >= 1:
        _print_verbose(result)
    else:
        _print_minimal(result)


def _print_minimal(result: DigitizerOutputs) -> None:
    """verbose=0: success line + output folder + the file names written there."""
    out_dir = Path(result.csv_path).parent
    print(f"Success. Output -> {out_dir}")
    for path_str in _output_paths(result):
        print(f"    {Path(path_str).name}")


def _print_verbose(result: DigitizerOutputs) -> None:
    """verbose>=1: full detail (color, pixel coords, tick->OCR, points, OCR confidence)."""
    out_dir = Path(result.csv_path).parent
    x_min_pt, x_max_pt, y_min_pt, y_max_pt = result.tick_points
    x_min, x_max, y_min, y_max = result.axis_values
    color = result.color_rgb
    axis_source = "OCR" if result.used_ocr else "provided manually"
    conf = (
        f"{result.ocr_confidence:.1f}%"
        if result.ocr_confidence is not None
        else "n/a (axes not read by OCR)"
    )
    print("Digitized successfully.")
    print(f"    color (r,g,b)  : {color[0]}, {color[1]}, {color[2]}")
    print(f"    pixel coords   : x_min={x_min_pt} x_max={x_max_pt} y_min={y_min_pt} y_max={y_max_pt}")
    print(f"    tick -> values : x_min={x_min} x_max={x_max} y_min={y_min} y_max={y_max}  ({axis_source})")
    print(f"    OCR confidence : {conf}")
    print(f"    num points     : {result.point_count}")
    print(f"    elapsed (s)    : {result.elapsed_seconds:.2f}")
    print(f"    output dir     : {out_dir}")
    for path_str in _output_paths(result):
        print(f"        {Path(path_str).name}")


def _report_unexpected(exc: Exception, verbose: int) -> None:
    """Quiet mode shows a one-line error; verbose mode shows the full traceback."""
    if verbose >= 1:
        import traceback

        traceback.print_exc()
    else:
        print(f"unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)


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

    log_x = _prompt_yes_no("Log scale on the X axis? [y/N]: ", default=False)
    log_y = _prompt_yes_no("Log scale on the Y axis? [y/N]: ", default=False)
    output_dir = _clean_input(input("Output directory [blank = image folder]: "))
    normalize_y = _prompt_yes_no("Add normalized Y column? (optional) [y/N]: ", default=False)
    limit_to_calibration = _prompt_yes_no("Limit points to calibration window? (optional) [y/N]: ", default=False)

    print()
    print("Summary")
    print(f"  plot location: {pic_dir}")
    print(f"  color: {color or 'auto'}")
    print(f"  tick coordinates: {ticks or 'OCR auto'}")
    print(f"  axis values: {axis_values or 'OCR auto'}")
    print(f"  log scale: X={'log' if log_x else 'linear'}, Y={'log' if log_y else 'linear'}")
    print(f"  output directory: {output_dir or 'image folder'}")
    print(f"  normalize y: {'yes' if normalize_y else 'no'}")
    print(f"  limit to calibration: {'yes' if limit_to_calibration else 'no'}")
    print()

    return {
        "pic_dir": pic_dir,
        "color": color,
        "ticks": ticks,
        "axis_values": axis_values,
        "log_x": log_x,
        "log_y": log_y,
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

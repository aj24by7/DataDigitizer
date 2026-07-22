"""Locks in the project-wide default: do NOT limit exported points to the calibration box.

Why this is the default rather than an option you remember to pass: the calibration box is
built from the detected tick positions, so if OCR reads an extreme tick even slightly short,
the box clips real data off the ends of the curve. Measured on
Pilot/Pilot Images/example_3_corundum_raman.png, limiting kept 81 of 930 points -- it threw
away 91% of the spectrum, silently.

The GUI has always defaulted to off; the CLI, batch CLI and API each defaulted to ON, so the
same image gave different answers depending on how you ran it. These tests keep all four in
agreement.
"""

import argparse
import inspect

import digitizer_api
import digitizer_batch_cli
import digitizer_cli


def _parsed_default(parser: argparse.ArgumentParser, args):
    return parser.parse_args(args).limit_to_calibration


def test_api_default_is_off():
    sig = inspect.signature(digitizer_api.digitize_image)
    assert sig.parameters["limit_to_calibration"].default is False


def test_cli_wrapper_default_is_off():
    sig = inspect.signature(digitizer_cli.digitizer_cli)
    assert sig.parameters["limit_to_calibration"].default is False


def test_cli_parser_default_is_off():
    parser = digitizer_cli.build_parser()
    assert _parsed_default(parser, ["image.png"]) is False


def test_batch_parser_default_is_off():
    parser = digitizer_batch_cli.build_parser()
    assert _parsed_default(parser, ["some_folder"]) is False


def test_flags_still_work_in_both_directions():
    """The flags remain available: --limit-to-calibration opts in, --no-... stays a no-op."""
    parser = digitizer_cli.build_parser()
    assert _parsed_default(parser, ["image.png", "--limit-to-calibration"]) is True
    assert _parsed_default(parser, ["image.png", "--no-limit-to-calibration"]) is False

    batch = digitizer_batch_cli.build_parser()
    assert _parsed_default(batch, ["folder", "--limit-to-calibration"]) is True
    assert _parsed_default(batch, ["folder", "--no-limit-to-calibration"]) is False


def test_function_call_template_default_is_off():
    """The `digitizer_cli(...)` string form must agree with the flag form."""
    values = digitizer_cli.parse_function_call('digitizer_cli(pic_dir="image.png")')
    assert values["limit_to_calibration"] is False


def test_gui_default_is_off():
    """The GUI was already correct; assert it so the four entry points cannot drift apart."""
    from UI import ColorExtractionState

    assert ColorExtractionState().limit_points_to_calib is False

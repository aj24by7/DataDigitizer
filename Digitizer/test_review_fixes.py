"""Regression tests for the 2026-08-31 readiness-review fixes.

Each test pins one fixed bug; the docstrings name the failure it guards against.
Run:  py -m pytest test_review_fixes.py -v   (from the Digitizer folder)
"""

from __future__ import annotations

import pytest

from digitizer import _first_positional
from digitizer_cli import (
    DigitizerCliError,
    _attach_negative_values,
    parse_numbers,
)
from PointPlacer import _pick_run_mid, _split_runs


# --- silent axis rescale (parse_numbers regex fallback) ---------------------------

def test_axis_typo_rejected_not_silently_rescaled():
    # "10O" (letter O) used to parse as 10, silently shrinking every y by 10x.
    with pytest.raises(DigitizerCliError, match="unrecognized text"):
        parse_numbers("0,10,0,10O", expected=4, name="axis-values")


def test_axis_good_inputs_still_parse():
    assert parse_numbers("0,10,0,100", 4, "axis") == [0.0, 10.0, 0.0, 100.0]
    assert parse_numbers("-5,5,0,100", 4, "axis") == [-5.0, 5.0, 0.0, 100.0]
    assert parse_numbers("1e2,1E+3,0,1", 4, "axis") == [100.0, 1000.0, 0.0, 1.0]
    assert parse_numbers("[10,200],[500,200],[10,200],[10,20]", 8, "ticks") is not None
    assert parse_numbers("none", 4, "axis") is None


# --- negative --axis values (argparse "expected one argument") --------------------

def test_negative_axis_value_attaches_to_flag():
    out = _attach_negative_values(["img.png", "--axis", "-5,5,0,100"])
    assert out == ["img.png", "--axis=-5,5,0,100"]


def test_attach_leaves_flags_and_positives_alone():
    args = ["img.png", "--axis", "0,10,0,100", "--color", "255,0,0", "-v"]
    assert _attach_negative_values(args) == args


# --- -v before a folder must not swallow the positional ---------------------------

def test_first_positional_not_swallowed_by_bare_verbose():
    assert _first_positional(["-v", "C:/charts"]) == "C:/charts"
    assert _first_positional(["--verbose", "C:/charts"]) == "C:/charts"


def test_first_positional_numeric_verbose_level_still_attaches():
    assert _first_positional(["-v", "1", "C:/charts"]) == "C:/charts"
    assert _first_positional(["--out", "D:/x", "C:/charts"]) == "C:/charts"


# --- PointPlacer run selection (legend line sample vs curve) ----------------------

def test_single_run_matches_old_midpoint_behaviour():
    runs = _split_runs([10, 11, 12, 13])
    assert runs == [(10, 13)]
    assert _pick_run_mid(runs, None) == 11.5


def test_small_gaps_bridge_into_one_run():
    # anti-aliasing / tolerance-fringe gaps (<= 32px) stay one stroke
    runs = _split_runs([10, 11, 40, 41])
    assert runs == [(10, 41)]
    assert _pick_run_mid(runs, None) == 25.5


def test_distant_object_no_longer_drags_midpoint():
    # curve at 100-104, legend line sample at 500-501: the old midpoint landed at
    # ~300 in empty space; now the larger (curve) run wins.
    runs = _split_runs([100, 101, 102, 103, 104, 500, 501])
    assert runs == [(100, 104), (500, 501)]
    assert _pick_run_mid(runs, None) == 102.0


def test_tie_broken_toward_previous_column():
    runs = _split_runs([100, 101, 500, 501])
    assert _pick_run_mid(runs, prev_y=498.0) == 500.5
    assert _pick_run_mid(runs, prev_y=99.0) == 100.5


# --- calibration border separation (thick single spine != two borders) ------------

def test_thick_spine_does_not_collapse_calibration_box():
    pytest.importorskip("PyQt6")
    from PyQt6 import QtGui

    from Calibration import _find_black_border_box

    # White canvas with ONE 2px-thick bottom spine and ONE 2px-thick left spine
    # (matplotlib's default open-spine style). The old code took rows {y, y+1} of
    # the single bottom spine as "top and bottom borders" -> a ~1px box.
    w = h = 200
    image = QtGui.QImage(w, h, QtGui.QImage.Format.Format_RGB888)
    image.fill(QtGui.QColor(255, 255, 255))
    black = QtGui.QColor(0, 0, 0)
    for x in range(10, 190):
        image.setPixelColor(x, 180, black)
        image.setPixelColor(x, 181, black)
    for y in range(10, 190):
        image.setPixelColor(10, y, black)
        image.setPixelColor(11, y, black)

    box = _find_black_border_box(image, threshold=60)
    assert box is not None
    xs = [pt[0] for pt in box]
    ys = [pt[1] for pt in box]
    # Whatever box is returned (the fallback bounding box of all black pixels is
    # fine), it must span a real area - never a ~1px sliver.
    assert max(xs) - min(xs) > 50
    assert max(ys) - min(ys) > 50

"""Guards against a degenerate OCR axis producing silently wrong data.

Axis values passed by hand go through `_normalize_axis_values`, which rejects x_min == x_max.
Values read by OCR skipped that check. When OCR misreads BOTH ticks on an axis as the same
number -- which happens on real charts -- the scale collapses to zero and every exported
value on that axis becomes identical. The run still reported success: a CSV where every x
was "8", 766 points, exit code 0, no warning.

Silently wrong numbers are the worst failure mode for a measurement tool, so this must be a
loud, actionable error instead.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

import digitizer_api
from digitizer_api import DigitizerCliError


class _FakeAxisResult:
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max


class _FakeWindow:
    def __init__(self, result):
        self._axis_result = result


def _detect(x_min, x_max, y_min, y_max):
    window = _FakeWindow(_FakeAxisResult(x_min, x_max, y_min, y_max))
    return digitizer_api._axis_values_from_detection(window)


def test_healthy_axis_passes_through():
    assert _detect(400.0, 700.0, 0.0, 20000.0) == (400.0, 700.0, 0.0, 20000.0)


def test_degenerate_x_is_rejected():
    """The exact failure seen in the wild: both x ticks misread as 8.0."""
    with pytest.raises(DigitizerCliError) as excinfo:
        _detect(8.0, 8.0, 0.0, 20000.0)
    message = str(excinfo.value)
    assert "same X value" in message
    assert "--axis" in message, "the error must tell the user how to recover"


def test_degenerate_y_is_rejected():
    with pytest.raises(DigitizerCliError) as excinfo:
        _detect(400.0, 700.0, 5.0, 5.0)
    assert "same Y value" in str(excinfo.value)
    assert "--axis" in str(excinfo.value)


def test_missing_value_still_rejected():
    with pytest.raises(DigitizerCliError):
        _detect(400.0, None, 0.0, 20000.0)


def test_hand_supplied_values_are_still_validated():
    """The pre-existing check on user-supplied --axis must remain."""
    with pytest.raises(DigitizerCliError):
        digitizer_api._normalize_axis_values([8.0, 8.0, 0.0, 100.0])
    with pytest.raises(DigitizerCliError):
        digitizer_api._normalize_axis_values([0.0, 10.0, 5.0, 5.0])
    assert digitizer_api._normalize_axis_values([0, 10, 0, 100]) == (0.0, 10.0, 0.0, 100.0)

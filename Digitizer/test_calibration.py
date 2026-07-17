"""Tests for the axis-pair repair in coordinate_mediated_calibration.

The bug these pin down: exclude_rects hides every OCR text bbox, and an oversized one can
cover the real axis under a tick label, so snap_up sails past it to the opposite spine. The
pair then spans a diagonal and the affine mapper skews (example_3_corundum_raman.png).

The repair must be surgical -- it may only fire on a pair that is tilted far past any real
scan, and must leave every already-working case byte-identical.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6 import QtGui, QtWidgets

from Calibration import coordinate_mediated_calibration

W, H = 800, 600
BOTTOM_SPINE = 550  # drawn frame: bottom edge lands here, top edge at y=20


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _canvas():
    image = QtGui.QImage(W, H, QtGui.QImage.Format.Format_RGB888)
    image.fill(QtGui.QColor(255, 255, 255))
    return image


def _draw(image, lines, pen_width=3):
    painter = QtGui.QPainter(image)
    painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), pen_width))
    for x1, y1, x2, y2 in lines:
        painter.drawLine(x1, y1, x2, y2)
    painter.end()
    return image


def _framed(gridlines=0):
    """Full four-sided frame, optionally with black gridlines across the plot area."""
    lines = [
        (50, 20, W - 50, 20), (50, BOTTOM_SPINE, W - 50, BOTTOM_SPINE),
        (50, 20, 50, BOTTOM_SPINE), (W - 50, 20, W - 50, BOTTOM_SPINE),
    ]
    for i in range(gridlines):
        y = 20 + (i + 1) * (BOTTOM_SPINE - 20) // (gridlines + 1)
        lines.append((50, y, W - 50, y))
    return _draw(_canvas(), lines)


def _calibrate(image, x_min, x_max, rects=None):
    return coordinate_mediated_calibration(
        image, x_min, x_max, (40, 300), (40, 60), exclude_rects=rects or []
    )


# A rect over the "x_min" label that also swallows the axis line beneath it.
HIDES_AXIS = [(150, 500, 120, 60)]


def test_bogus_rect_no_longer_produces_a_diagonal(app):
    result = _calibrate(_framed(), (200, 570), (700, 570), rects=HIDES_AXIS)
    assert result.x_min_point[1] == result.x_max_point[1]
    assert result.x_min_point == (200, BOTTOM_SPINE + 1)


def test_clean_image_is_untouched(app):
    result = _calibrate(_framed(), (200, 570), (700, 570))
    assert result.x_min_point == (200, BOTTOM_SPINE + 1)
    assert result.x_max_point == (700, BOTTOM_SPINE + 1)


def test_pair_that_agrees_is_left_alone_even_on_the_wrong_line(app):
    """Both snaps overshoot to the top spine together. Parallel is all the mapper needs --
    a constant offset of the axis line cancels out of the u/v solve -- so do not touch it."""
    both_hidden = [(150, 500, 600, 60)]
    result = _calibrate(_framed(), (200, 570), (700, 570), rects=both_hidden)
    assert result.x_min_point[1] == result.x_max_point[1] == 21


def test_real_tilt_is_preserved(app):
    """A rotated/scanned axis is the affine mapper's job; flattening it would regress."""
    image = _draw(_canvas(), [(50, 500, 750, 561), (50, 20, 50, 561)])
    result = _calibrate(image, (200, 560), (700, 590))
    assert result.x_min_point[1] != result.x_max_point[1]
    assert abs(result.x_min_point[1] - result.x_max_point[1]) > 20


def test_gridlines_defeat_line_evidence_but_travel_still_repairs(app):
    """Black gridlines make the projection untrustworthy, so arbitration falls through to
    snap travel: the point that ran furthest from its own label is the wrong one."""
    result = _calibrate(_framed(gridlines=8), (200, 570), (700, 570), rects=HIDES_AXIS)
    assert result.x_min_point[1] == result.x_max_point[1] == BOTTOM_SPINE + 1


def test_partial_frame_bottom_and_left_only(app):
    image = _draw(_canvas(), [(50, 550, 750, 550), (50, 20, 50, 550)])
    result = _calibrate(image, (200, 570), (700, 570))
    assert result.x_min_point[1] == result.x_max_point[1]


def test_missing_tick_is_not_invented(app):
    result = _calibrate(_framed(), None, (700, 570))
    assert result.x_min_point is None
    assert result.x_max_point == (700, BOTTOM_SPINE + 1)


def test_blank_image_finds_nothing(app):
    result = _calibrate(_canvas(), (200, 570), (700, 570))
    assert result.x_min_point is None
    assert result.x_max_point is None

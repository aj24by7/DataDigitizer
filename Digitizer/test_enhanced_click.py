"""Tests for EnhancedClick.snap.

The important properties are not "does it move the point" but:
  * it lands on the true feature centre, sub-pixel, for thin AND thick strokes;
  * it corrects only the axis it has evidence for;
  * it does NOT move the point when there is no feature, or when the nearest feature is
    implausibly far away.

The last group matters most: a snap that silently relocates a calibration point to the
wrong tick is exactly the silent-error class this project exists to avoid.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6 import QtGui, QtWidgets

from EnhancedClick import snap

W = H = 61
BG = QtGui.QColor(255, 255, 255)
INK = QtGui.QColor(0, 0, 0)


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _canvas():
    im = QtGui.QImage(W, H, QtGui.QImage.Format.Format_RGB888)
    im.fill(BG)
    return im


def _draw(im, lines, width=1, colour=INK, antialias=False):
    p = QtGui.QPainter(im)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, antialias)
    p.setPen(QtGui.QPen(colour, width))
    for x1, y1, x2, y2 in lines:
        p.drawLine(x1, y1, x2, y2)
    p.end()
    return im


def _cross(width=1, cx=30, cy=30, colour=INK, antialias=False):
    return _draw(_canvas(), [(cx, cy - 12, cx, cy + 12), (cx - 12, cy, cx + 12, cy)],
                 width, colour, antialias)


# --------------------------------------------------------------- finds the centre
@pytest.mark.parametrize("dx,dy", [(0, 0), (1, 1), (-2, 1), (2, -2), (-1, -2)])
def test_thin_cross_recovers_centre(app, dx, dy):
    """A click offset by a plausible human error lands back on the true centre."""
    im = _cross(width=1)
    r = snap(im, 30 + dx, 30 + dy)
    assert r.kind == "cross", r.message
    assert abs(r.x - 30) < 0.6, r
    assert abs(r.y - 30) < 0.6, r


@pytest.mark.parametrize("width", [1, 2, 3, 5])
def test_thick_strokes_find_their_own_centre(app, width):
    """Thickness needs no special case: the centroid of the stroke is its middle."""
    im = _cross(width=width)
    r = snap(im, 32, 28)
    assert r.kind == "cross", r.message
    # Qt centres even-width pens half a pixel off; allow for that rather than pretend
    # sub-pixel accuracy we cannot have about where Qt put the ink.
    assert abs(r.x - 30) <= 1.0, r
    assert abs(r.y - 30) <= 1.0, r


def test_antialiased_cross(app):
    im = _cross(width=1, antialias=True)
    r = snap(im, 31, 29)
    assert r.kind == "cross"
    assert abs(r.x - 30) < 1.0 and abs(r.y - 30) < 1.0


def test_coloured_axis(app):
    """Ink is measured against the window's own background, not against black."""
    im = _cross(width=2, colour=QtGui.QColor(30, 90, 200))
    r = snap(im, 31, 31)
    assert r.kind == "cross", r.message
    assert abs(r.x - 30) <= 1.0 and abs(r.y - 30) <= 1.0


def test_corner_not_a_full_cross(app):
    """An L corner - axis meets tick - is still a two-stroke feature."""
    im = _draw(_canvas(), [(30, 30, 30, 44), (30, 30, 44, 30)], 2)
    r = snap(im, 31, 29)
    assert r.kind == "cross", r.message


# --------------------------------------------------------------- partial snapping
def test_vertical_line_only_corrects_x(app):
    """With one vertical stroke, x is knowable and y is not. y must be left alone."""
    im = _draw(_canvas(), [(30, 5, 30, 55)], 1)
    r = snap(im, 32, 27)
    assert r.kind == "line-x", r.message
    assert abs(r.x - 30) < 0.6
    assert r.y == 27, "y was invented from an image that does not constrain it"


def test_horizontal_line_only_corrects_y(app):
    im = _draw(_canvas(), [(5, 30, 55, 30)], 1)
    r = snap(im, 27, 32)
    assert r.kind == "line-y", r.message
    assert abs(r.y - 30) < 0.6
    assert r.x == 27


# --------------------------------------------------------------- refuses to guess
def test_blank_image_does_not_move_the_point(app):
    r = snap(_canvas(), 30, 30)
    assert r.kind == "none"
    assert (r.x, r.y) == (30, 30)
    assert not r.changed


def test_uniform_ink_does_not_move_the_point(app):
    """Clicked inside a filled block: there is no stroke, only area."""
    im = _canvas()
    p = QtGui.QPainter(im)
    p.fillRect(15, 15, 30, 30, INK)
    p.end()
    r = snap(im, 30, 30)
    assert r.kind == "none", r.message
    assert (r.x, r.y) == (30, 30)


def test_distant_feature_is_rejected_not_snapped(app):
    """The guard against locking onto a neighbouring gridline or the axis label."""
    im = _cross(width=1, cx=30, cy=30)
    r = snap(im, 40, 40, search_radius=14, max_move=3.0)
    assert r.kind == "rejected", r.message
    assert (r.x, r.y) == (40, 40)
    assert "limit" in r.message


def test_max_move_is_honoured(app):
    im = _cross(width=1)
    far = snap(im, 33, 33, search_radius=8, max_move=0.5)
    assert far.kind == "rejected"
    near = snap(im, 33, 33, search_radius=8, max_move=6.0)
    assert near.kind == "cross"


def test_result_is_subpixel(app):
    """The whole point of the feature: the answer is not restricted to whole pixels."""
    im = _cross(width=2)
    r = snap(im, 31, 31)
    assert isinstance(r.x, float) and isinstance(r.y, float)
    assert (r.x % 1 != 0) or (r.y % 1 != 0), "no sub-pixel component at all"


def test_never_raises_on_edge_clicks(app):
    im = _cross()
    for pt in [(0, 0), (W - 1, H - 1), (0, H - 1), (W - 1, 0)]:
        r = snap(im, *pt)
        assert r is not None and isinstance(r.x, float)


# ------------------------------------------------- regression: the axis-line pedestal
# Found in real use: an x-axis tick snapped 1.76 px off, i.e. it barely moved at all, while
# the y coordinate was exact. A tick sits ON its axis, so the axis line crossing the window
# adds ink to EVERY entry of the perpendicular profile. With a pedestal t under a peak of
# (h + t), the half-height threshold (h + t)/2 falls below the pedestal whenever t >= h, the
# support runs the full width of the window, and the centroid collapses back onto the click.
# The profile must therefore be pedestal-subtracted before the peak is measured.

def _axis_with_tick(horizontal=True, tick_len=7, axis_width=3, inward=False):
    im = QtGui.QImage(200, 140, QtGui.QImage.Format.Format_RGB888)
    im.fill(BG)
    p = QtGui.QPainter(im)
    p.setPen(QtGui.QPen(INK, axis_width))
    if horizontal:
        p.drawLine(10, 70, 190, 70)
    else:
        p.drawLine(70, 10, 70, 130)
    p.setPen(QtGui.QPen(INK, 2))
    d = -tick_len if inward else tick_len
    if horizontal:
        p.drawLine(100, 70, 100, 70 + d)
    else:
        p.drawLine(70, 70, 70 - d, 70)
    p.end()
    return im


@pytest.mark.parametrize("tick_len,axis_width,inward", [
    (7, 3, False), (7, 3, True), (4, 3, False), (7, 1, False), (12, 3, False),
])
def test_x_axis_tick_recovers_x(app, tick_len, axis_width, inward):
    """The coordinate the TICK determines must be recovered, not just the one the axis does."""
    im = _axis_with_tick(True, tick_len, axis_width, inward)
    r = snap(im, 99.5 + 1.2, 69.5 - 0.9, search_radius=5.0, max_move=3.0)
    assert r.kind == "cross", r.message
    assert abs(r.x - 99.5) < 0.5, f"x collapsed back toward the click: {r.x}"


@pytest.mark.parametrize("tick_len,axis_width,inward", [
    (7, 3, False), (7, 3, True), (4, 3, False), (7, 1, False),
])
def test_y_axis_tick_recovers_y(app, tick_len, axis_width, inward):
    im = _axis_with_tick(False, tick_len, axis_width, inward)
    r = snap(im, 69.5 + 1.2, 69.5 - 0.9, search_radius=5.0, max_move=3.0)
    assert r.kind == "cross", r.message
    assert abs(r.y - 69.5) < 0.5, f"y collapsed back toward the click: {r.y}"

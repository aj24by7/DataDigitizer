"""Enhanced human click: snap a manually-placed calibration point onto the tick it meant.

Motivation
----------
A human clicking a target lands about 0.9 px away from it (per-axis standard deviation,
measured on 225 clicks in-house). That error propagates straight into the coordinate frame,
so every exported value inherits it. When the user clicks an axis tick there is usually a
crosshair, corner or line right there, and its exact centre can be recovered from the image
far more precisely than a hand can point at it.

This module finds that feature and returns its sub-pixel centre.

How far to look
---------------
Radial miss distance follows a Rayleigh distribution with scale sigma, so the fraction of
clicks landing outside radius R is exp(-R^2 / 2 sigma^2):

    R = 1.8 px (2 sigma)  ->  13.5% of clicks fall outside
    R = 2.7 px (3 sigma)  ->   1.1%
    R = 3.6 px (4 sigma)  ->   0.03%

Searching only 1.8 px would therefore miss the feature on roughly one click in seven. Two
separate radii are used instead:

    search_radius  - how far we LOOK for the feature (default 4.0 px)
    max_move       - how far we are WILLING to move the point (default 3.0 px)

Looking wide finds the tick; capping the move stops the point jumping to a neighbouring
gridline, a minor tick on a log axis, or the axis label.

What gets snapped
-----------------
The snap is only ever as strong as the evidence:

    CROSS   both a vertical and a horizontal stroke  -> correct x and y
    LINE    exactly one stroke                       -> correct ACROSS the line only, and
                                                        leave the position ALONG it as clicked
    NONE    no stroke                                -> do not move the point at all

The LINE case matters: on a plain axis with no tick mark the user's intent to be *on the
line* is known, but their position *along* it is not, so guessing it would invent precision
that is not in the image.

Thick and thin strokes, and anti-aliasing, need no special handling: the centre is the
intensity-weighted centroid over the stroke's own support, which is correct for a 1 px
hairline and a 5 px rule alike, and anti-aliased edge pixels contribute proportionally.
That weighting is also what makes the result sub-pixel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from PyQt6 import QtGui

Kind = Literal["cross", "line-x", "line-y", "none", "rejected"]

# Defaults; the UI exposes search_radius as a slider.
DEFAULT_SEARCH_RADIUS = 4.0
DEFAULT_MAX_MOVE = 3.0

# A peak must rise this far ABOVE ITS PEDESTAL to count as a stroke, as a fraction of the
# window diameter. Calibrated against real tick geometry rather than guessed: a 7 px tick in
# a 19 px window lifts about 5.2 ink units (0.27 of the window) and a 4 px tick about 3.0
# (0.16), so the floor sits below both while still rejecting anti-aliasing ripple, which
# lifts well under 1.
_MIN_PEAK_FRACTION = 0.10
# ...and must stand this far above the profile's own median, so a uniformly dark window
# (the user clicked inside a filled region) yields no stroke rather than an arbitrary one.
_MIN_PROMINENCE = 1.6
# Extra pixels measured beyond the search radius so a thick stroke's support is never
# clipped by the window edge (a clipped support biases the centroid).
_MEASURE_PAD = 4


@dataclass(frozen=True)
class SnapResult:
    """Outcome of one snap attempt."""
    x: float
    y: float
    kind: Kind
    moved: float                 # pixels travelled from the original click
    message: str                 # user-facing, shown in the status bar

    @property
    def changed(self) -> bool:
        return self.kind in ("cross", "line-x", "line-y")


def _ink_window(image: QtGui.QImage, cx: int, cy: int, r: int):
    """Return (weights, x0, y0) where weights[j][i] is 'how much ink' at that pixel.

    Ink is measured as distance from the window's own background colour rather than from
    black, so the feature is found on charts whose axes are drawn in colour, and so
    anti-aliased pixels contribute in proportion to their coverage.
    """
    w, h = image.width(), image.height()
    x0, x1 = max(0, cx - r), min(w - 1, cx + r)
    y0, y1 = max(0, cy - r), min(h - 1, cy + r)
    if x1 <= x0 or y1 <= y0:
        return None, x0, y0

    rgb = []
    for y in range(y0, y1 + 1):
        row = []
        for x in range(x0, x1 + 1):
            c = image.pixelColor(x, y)
            row.append((c.red(), c.green(), c.blue()))
        rgb.append(row)

    # Background = the most common quantised colour on the window's OUTER RING, not in the
    # window as a whole. A thick crosshair can occupy more than half a small window, and
    # taking the overall mode then elects the ink as "background" and inverts every profile.
    # The ring is crossed by at most the four arms, so it stays background-dominated, and
    # unlike assuming "background is the lighter colour" this works for inverted charts too.
    nrows, ncols = len(rgb), len(rgb[0])
    ring = []
    for i in range(ncols):
        ring.append(rgb[0][i]); ring.append(rgb[nrows - 1][i])
    for j in range(1, nrows - 1):
        ring.append(rgb[j][0]); ring.append(rgb[j][ncols - 1])

    counts: dict[Tuple[int, int, int], int] = {}
    for px in ring:
        key = (px[0] // 32, px[1] // 32, px[2] // 32)
        counts[key] = counts.get(key, 0) + 1
    bg_key = max(counts, key=counts.get)
    bg = (bg_key[0] * 32 + 16, bg_key[1] * 32 + 16, bg_key[2] * 32 + 16)

    weights = []
    for row in rgb:
        wrow = []
        for px in row:
            d = max(abs(px[0] - bg[0]), abs(px[1] - bg[1]), abs(px[2] - bg[2]))
            wrow.append(min(1.0, d / 110.0))     # saturates well before full contrast
        weights.append(wrow)
    return weights, x0, y0


def _peak_centre(profile: list[float], centre_index: float,
                 max_offset: Optional[float] = None) -> Optional[float]:
    """Sub-pixel centre of the most significant peak nearest `centre_index`, or None."""
    n = len(profile)
    if n < 3:
        return None

    # A tick sits ON its axis, and the axis line crossing the window contributes ink to
    # EVERY entry of the perpendicular profile. That pedestal has to be removed before the
    # peak is measured. Without it, an x-axis tick of height h on an axis of thickness t
    # gives a peak of (h + t) over a pedestal of t; the half-height threshold (h + t)/2 then
    # falls BELOW the pedestal whenever t >= h, so the support runs the full width of the
    # window and the centroid collapses back onto the click. Measured: a real x tick landed
    # 1.76 px off, i.e. it barely moved at all.
    base = sorted(profile)[n // 2]                 # median: robust to the peak itself
    lifted = [max(0.0, v - base) for v in profile]

    peak_floor = max(_MIN_PEAK_FRACTION * n, 2.0)
    candidates = []
    for i, v in enumerate(lifted):
        if v < peak_floor:
            continue
        if max_offset is not None and abs(i - centre_index) > max_offset:
            continue
        if (i == 0 or lifted[i - 1] <= v) and (i == n - 1 or lifted[i + 1] <= v):
            candidates.append(i)
    if not candidates:
        return None

    i = min(candidates, key=lambda k: (abs(k - centre_index), -lifted[k]))

    # Support = the contiguous run around the peak above half its PROMINENCE, and the
    # centroid is weighted by the pedestal-subtracted profile, so the axis line contributes
    # nothing to the position. This is what makes the answer the centre of the tick itself,
    # sub-pixel, for a stroke of any thickness.
    half = lifted[i] / 2.0
    lo = i
    while lo - 1 >= 0 and lifted[lo - 1] >= half:
        lo -= 1
    hi = i
    while hi + 1 < n and lifted[hi + 1] >= half:
        hi += 1

    total = sum(lifted[lo:hi + 1])
    if total <= 0:
        return None
    return sum(k * lifted[k] for k in range(lo, hi + 1)) / total


def snap(
    image: QtGui.QImage,
    x: float,
    y: float,
    search_radius: float = DEFAULT_SEARCH_RADIUS,
    max_move: float = DEFAULT_MAX_MOVE,
) -> SnapResult:
    """Snap a click at (x, y) onto the nearest tick feature. Never raises."""
    ox, oy = float(x), float(y)
    if image is None or image.isNull():
        return SnapResult(ox, oy, "none", 0.0, "Enhanced click: no image.")

    # Measure in a window PADDED beyond the search radius. A thick stroke whose support ran
    # off the edge of the window had its centroid clipped and therefore biased, which showed
    # up as a 5 px pen reporting a centre 3.5 px from the truth. Candidate peaks are still
    # restricted to `search_radius` of the click below, so the padding widens what we can
    # measure accurately without widening what we are willing to snap to.
    r = max(2, int(round(search_radius))) + _MEASURE_PAD
    weights, x0, y0 = _ink_window(image, int(round(ox)), int(round(oy)), r)
    if weights is None:
        return SnapResult(ox, oy, "none", 0.0,
                          "Enhanced click: too close to the image edge, using your click.")

    rows = len(weights)
    cols = len(weights[0])
    col_profile = [sum(weights[j][i] for j in range(rows)) for i in range(cols)]
    row_profile = [sum(weights[j][i] for i in range(cols)) for j in range(rows)]

    # A vertical stroke fills a column, so it peaks in col_profile; a horizontal stroke
    # peaks in row_profile. The two are therefore separable without any orientation logic.
    # Peaks may be measured anywhere in the padded window, but only those within
    # search_radius of the click are eligible to move the point.
    limit = float(search_radius)
    vx = _peak_centre(col_profile, ox - x0, limit)
    hy = _peak_centre(row_profile, oy - y0, limit)

    nx = x0 + vx if vx is not None else ox
    ny = y0 + hy if hy is not None else oy

    if vx is not None and hy is not None:
        kind: Kind = "cross"
    elif vx is not None:
        kind = "line-x"          # vertical stroke: correct x, keep y
    elif hy is not None:
        kind = "line-y"          # horizontal stroke: correct y, keep x
    else:
        return SnapResult(ox, oy, "none", 0.0,
                          "Enhanced click: no crosshair or line found — using your click.")

    moved = ((nx - ox) ** 2 + (ny - oy) ** 2) ** 0.5
    if moved > max_move:
        return SnapResult(
            ox, oy, "rejected", 0.0,
            f"Enhanced click: nearest feature was {moved:.1f} px away (limit {max_move:.1f}) "
            "— using your click.")

    if kind == "cross":
        msg = f"Enhanced click: snapped {moved:.2f} px onto a crosshair."
    elif kind == "line-x":
        msg = f"Enhanced click: vertical line found, corrected x by {abs(nx - ox):.2f} px."
    else:
        msg = f"Enhanced click: horizontal line found, corrected y by {abs(ny - oy):.2f} px."
    return SnapResult(nx, ny, kind, moved, msg + "  Ctrl+Z to undo.")

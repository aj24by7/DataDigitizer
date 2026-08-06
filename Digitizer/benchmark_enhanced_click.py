"""Does Enhanced Click actually beat a raw human click? Monte Carlo over simulated clicks.

Method
------
Render tick features of several kinds, then for each trial displace the click by
N(0, sigma^2) per axis with sigma = 0.9 px (the in-house measured value), snap, and record
the residual distance from the true feature centre.

This is a SIMULATED human, and must be reported as such. It measures what the snapping
algorithm can recover given Gaussian click noise; it does not re-measure human behaviour.
The real 225-click study stands on its own and should be reported alongside, not replaced.

Run:  py benchmark_enhanced_click.py
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PyQt6 import QtGui, QtWidgets

from EnhancedClick import snap

SIGMA = 0.9          # px, per axis, measured in-house over 225 clicks
TRIALS = 2000
TRUE = 60.0          # feature centre, in a 121x121 canvas


def canvas():
    im = QtGui.QImage(121, 121, QtGui.QImage.Format.Format_RGB888)
    im.fill(QtGui.QColor(255, 255, 255))
    return im


def draw(lines, width=1, colour=QtGui.QColor(0, 0, 0), aa=False):
    im = canvas()
    p = QtGui.QPainter(im)
    p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, aa)
    p.setPen(QtGui.QPen(colour, width))
    for ln in lines:
        p.drawLine(*ln)
    p.end()
    return im


C = int(TRUE)
FEATURES = {
    "thin cross":        draw([(C, C - 20, C, C + 20), (C - 20, C, C + 20, C)], 1),
    "thick cross (3px)": draw([(C, C - 20, C, C + 20), (C - 20, C, C + 20, C)], 3),
    "antialiased cross": draw([(C, C - 20, C, C + 20), (C - 20, C, C + 20, C)], 1, aa=True),
    "corner (L)":        draw([(C, C, C, C + 20), (C, C, C + 20, C)], 2),
    "axis line only":    draw([(C, C - 25, C, C + 25)], 1),
    "no feature":        canvas(),
}


def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    rng = np.random.default_rng(7)

    print(f"Simulated human click, sigma = {SIGMA} px per axis, {TRIALS} trials per feature")
    print("Residual = distance from the true feature centre after snapping.\n")
    print(f"{'feature':20s}{'raw RMS':>10s}{'snapped RMS':>13s}{'improvement':>13s}{'snap rate':>11s}")
    print("-" * 68)

    for name, im in FEATURES.items():
        dx = rng.normal(0, SIGMA, TRIALS)
        dy = rng.normal(0, SIGMA, TRIALS)
        raw, out, fired = [], [], 0
        for i in range(TRIALS):
            cx, cy = TRUE + dx[i], TRUE + dy[i]
            raw.append(np.hypot(cx - TRUE, cy - TRUE))
            r = snap(im, cx, cy)
            fired += r.changed
            if name == "axis line only":
                # Only x is knowable from a vertical line; y is deliberately untouched, so
                # scoring the full 2-D distance would penalise correct behaviour.
                out.append(abs(r.x - TRUE))
            else:
                out.append(np.hypot(r.x - TRUE, r.y - TRUE))
        raw_rms = float(np.sqrt(np.mean(np.square(raw))))
        out_rms = float(np.sqrt(np.mean(np.square(out))))
        gain = raw_rms / out_rms if out_rms > 1e-9 else float("inf")
        note = "  (x only)" if name == "axis line only" else ""
        print(f"{name:20s}{raw_rms:9.3f}px{out_rms:12.3f}px{gain:12.1f}x{fired / TRIALS * 100:10.0f}%{note}")

    print("\nNotes")
    print("  · 'no feature' must show ~1.0x and a 0% snap rate: the point is left alone.")
    print("  · 'axis line only' is scored on x alone, because y is intentionally not moved.")
    print("  · Raw RMS ~1.27 px is the Rayleigh mean for sigma=0.9 (sigma*sqrt(2)), a check")
    print("    that the simulated noise is what we think it is.")


if __name__ == "__main__":
    main()

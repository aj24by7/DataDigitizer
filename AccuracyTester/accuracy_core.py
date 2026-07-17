"""Core scoring logic shared by Accuracy Tester Pro and the batch pipelines.

These functions are the exact logic previously implemented as instance methods on
the `AccuracyTesterPro` Tkinter class (`_prepare_series`, `_comparison_grid`,
`_compute_metrics`), extracted so they can be used without a running GUI. The
formulas, defaults, and edge-case behavior are intentionally unchanged.

Comparison workflow (matching the GUI):

1. `prepare_series` cleans each CSV-derived table into a sorted, deduplicated
   numeric series.
2. `comparison_grid` resamples the two series onto one comparison grid via
   linear interpolation (`np.interp`), clipped to the overlapping x-range.
3. `compute_metrics` scores reference-vs-comparison y-values on that grid.

Edge-case behavior (all inherited from the GUI implementation):

- Non-numeric, NaN, or +/-inf rows are dropped by `prepare_series`; if nothing
  numeric remains it raises ``ValueError``.
- Duplicate x-values are collapsed by the chosen policy (``median``, ``mean``,
  or ``first``); non-monotonic x is sorted (stable mergesort).
- Non-overlapping x-ranges make `comparison_grid` raise ``ValueError`` — a pair
  with no overlap is a failure, never a number.
- `compute_metrics` returns ``nan`` for R^2 when the reference has zero
  variance, for correlation when either side has zero variance (or fewer than
  2 points), and for MAPE/WAPE when the reference is (near-)zero everywhere.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class SeriesData:
    x: np.ndarray
    y: np.ndarray
    raw_rows: int
    valid_rows: int
    duplicate_rows: int
    unique_rows: int


def prepare_series(df: pd.DataFrame, x_col: str, y_col: str, dup_policy: str) -> SeriesData:
    """Coerce two columns of ``df`` to a clean numeric series sorted by x.

    Raises ``ValueError`` if the columns are missing, if no valid numeric rows
    remain after dropping NaN/inf, or if ``dup_policy`` is not one of
    ``median`` / ``mean`` / ``first``.
    """
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(
            f"Columns '{x_col}' and/or '{y_col}' not found. Available columns: {', '.join(map(str, df.columns))}"
        )

    temp = pd.DataFrame(
        {
            "x": pd.to_numeric(df[x_col], errors="coerce"),
            "y": pd.to_numeric(df[y_col], errors="coerce"),
        }
    )
    raw_rows = len(temp)
    temp = temp.replace([np.inf, -np.inf], np.nan).dropna(subset=["x", "y"])
    valid_rows = len(temp)
    if temp.empty:
        raise ValueError("No valid numeric rows found after converting selected X/Y columns.")

    temp = temp.sort_values("x", kind="mergesort")
    duplicate_rows = int(valid_rows - temp["x"].nunique())

    if dup_policy == "median":
        grouped = temp.groupby("x", as_index=False, sort=True)["y"].median()
    elif dup_policy == "mean":
        grouped = temp.groupby("x", as_index=False, sort=True)["y"].mean()
    elif dup_policy == "first":
        grouped = temp.drop_duplicates(subset=["x"], keep="first")[["x", "y"]].sort_values("x", kind="mergesort")
    else:
        raise ValueError("Duplicate X mode must be one of: median, mean, first")

    return SeriesData(
        x=grouped["x"].to_numpy(dtype=float),
        y=grouped["y"].to_numpy(dtype=float),
        raw_rows=raw_rows,
        valid_rows=valid_rows,
        duplicate_rows=duplicate_rows,
        unique_rows=len(grouped),
    )


def comparison_grid(
    orig_x: np.ndarray,
    orig_y: np.ndarray,
    dig_x: np.ndarray,
    dig_y: np.ndarray,
    mode: str,
    grid_points: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Resample both series onto one comparison grid over the overlapping x-range.

    ``mode`` is one of ``original_x`` (compare at the original series' x-values),
    ``digitized_x`` (at the digitized series' x-values), or ``common_uniform``
    (``grid_points`` evenly spaced x-values). Interpolation is linear
    (``np.interp``), clipped to the overlap. Returns
    ``(x_cmp, y_ref, y_cmp, {"overlap_start": ..., "overlap_end": ...})``.

    Raises ``ValueError`` when there is no overlapping x-range or the grid ends
    up with fewer than 2 points. Inputs must already be sorted by x
    (as produced by `prepare_series`).
    """
    overlap_start = max(float(orig_x[0]), float(dig_x[0]))
    overlap_end = min(float(orig_x[-1]), float(dig_x[-1]))
    if overlap_end <= overlap_start:
        raise ValueError("No overlapping X range between datasets after cleanup.")

    if mode == "original_x":
        mask = (orig_x >= overlap_start) & (orig_x <= overlap_end)
        x_cmp = orig_x[mask]
        y_ref = orig_y[mask]
        y_cmp = np.interp(x_cmp, dig_x, dig_y)
    elif mode == "digitized_x":
        mask = (dig_x >= overlap_start) & (dig_x <= overlap_end)
        x_cmp = dig_x[mask]
        y_ref = np.interp(x_cmp, orig_x, orig_y)
        y_cmp = dig_y[mask]
    elif mode == "common_uniform":
        x_cmp = np.linspace(overlap_start, overlap_end, max(2, int(grid_points)), dtype=float)
        y_ref = np.interp(x_cmp, orig_x, orig_y)
        y_cmp = np.interp(x_cmp, dig_x, dig_y)
    else:
        raise ValueError("Unsupported grid mode.")

    if x_cmp.size < 2:
        raise ValueError("Overlap exists but comparison produced fewer than 2 points.")

    return x_cmp, y_ref, y_cmp, {"overlap_start": overlap_start, "overlap_end": overlap_end}


def compute_metrics(y_ref: np.ndarray, y_cmp: np.ndarray) -> Dict[str, float]:
    """Score comparison-grid y-values against the reference y-values.

    Returns every statistic the Accuracy Tester GUI reports: mse, rmse, mae,
    median_ae, p95_ae, max_ae, bias, std_residual, r2, corr, mape_pct
    (near-zero reference values masked), smape_pct, wape_pct, nrmse_range_pct.
    Undefined statistics come back as ``nan`` (see module docstring).
    """
    residual = y_ref - y_cmp
    abs_err = np.abs(residual)

    mse = float(np.mean(residual ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(abs_err))
    medae = float(np.median(abs_err))
    p95 = float(np.percentile(abs_err, 95))
    maxae = float(np.max(abs_err))
    bias = float(np.mean(residual))
    std = float(np.std(residual))

    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y_ref - np.mean(y_ref)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    if y_ref.size >= 2 and np.std(y_ref) > 0 and np.std(y_cmp) > 0:
        corr = float(np.corrcoef(y_ref, y_cmp)[0, 1])
    else:
        corr = np.nan

    scale = float(np.max(np.abs(y_ref))) if y_ref.size else 0.0
    eps = max(1e-12, scale * 1e-9)
    mape_mask = np.abs(y_ref) > eps
    mape = float(np.mean(np.abs(residual[mape_mask] / y_ref[mape_mask])) * 100.0) if np.any(mape_mask) else np.nan
    smape = float(np.mean(200.0 * abs_err / (np.abs(y_ref) + np.abs(y_cmp) + eps)))
    wape_denom = float(np.sum(np.abs(y_ref)))
    wape = float(np.sum(abs_err) / wape_denom * 100.0) if wape_denom > eps else np.nan

    y_range = float(np.max(y_ref) - np.min(y_ref)) if y_ref.size else 0.0
    nrmse_range = float(rmse / y_range * 100.0) if y_range > 0 else np.nan

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "median_ae": medae,
        "p95_ae": p95,
        "max_ae": maxae,
        "bias": bias,
        "std_residual": std,
        "r2": r2,
        "corr": corr,
        "mape_pct": mape,
        "smape_pct": smape,
        "wape_pct": wape,
        "nrmse_range_pct": nrmse_range,
    }

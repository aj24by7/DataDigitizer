import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except ImportError:  # optional
    DND_FILES = None
    TkinterDnD = None
    HAS_DND = False


@dataclass
class SeriesData:
    x: np.ndarray
    y: np.ndarray
    raw_rows: int
    valid_rows: int
    duplicate_rows: int
    unique_rows: int


class AccuracyTesterPro:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Accuracy Tester Pro - Robust CSV Comparison")
        self.root.geometry("1180x840")

        self.original_df: Optional[pd.DataFrame] = None
        self.digitized_df: Optional[pd.DataFrame] = None
        self.result: Optional[Dict[str, object]] = None
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.current_outlier_index = 0
        self.outlier_label_var = tk.StringVar(value="Outlier zoom: n/a")

        self._build_ui()

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.rowconfigure(4, weight=1)

        ttk.Label(
            main,
            text="Accuracy Tester Pro (Original vs Digitized/Synthesized)",
            font=("Arial", 15, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        self.status_var = tk.StringVar(
            value="Load two CSV files. Drag and drop is {}.".format(
                "enabled" if HAS_DND else "not available (install tkinterdnd2)"
            )
        )
        ttk.Label(main, textvariable=self.status_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 8))

        file_box = ttk.LabelFrame(main, text="Data Input", padding=10)
        file_box.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        file_box.columnconfigure(0, weight=1)
        file_box.columnconfigure(1, weight=1)

        self.orig_drop = tk.Frame(file_box, bg="#d8ecff", relief="solid", bd=2)
        self.orig_drop.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self.orig_drop.columnconfigure(0, weight=1)
        self.orig_label = tk.Label(
            self.orig_drop,
            text="Drop Original CSV Here\n(or click to browse)",
            bg="#d8ecff",
            font=("Arial", 12),
            justify="center",
        )
        self.orig_label.grid(row=0, column=0, padx=16, pady=28)
        self.orig_drop.bind("<Button-1>", lambda _e: self.browse_file("original"))
        self.orig_label.bind("<Button-1>", lambda _e: self.browse_file("original"))

        self.dig_drop = tk.Frame(file_box, bg="#d9f7d9", relief="solid", bd=2)
        self.dig_drop.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.dig_drop.columnconfigure(0, weight=1)
        self.dig_label = tk.Label(
            self.dig_drop,
            text="Drop Digitized/Synthesized CSV Here\n(or click to browse)",
            bg="#d9f7d9",
            font=("Arial", 12),
            justify="center",
        )
        self.dig_label.grid(row=0, column=0, padx=16, pady=28)
        self.dig_drop.bind("<Button-1>", lambda _e: self.browse_file("digitized"))
        self.dig_label.bind("<Button-1>", lambda _e: self.browse_file("digitized"))

        if HAS_DND:
            self.orig_drop.drop_target_register(DND_FILES)
            self.orig_drop.dnd_bind("<<Drop>>", lambda e: self.drop_file(e, "original"))
            self.dig_drop.drop_target_register(DND_FILES)
            self.dig_drop.dnd_bind("<<Drop>>", lambda e: self.drop_file(e, "digitized"))

        opts = ttk.LabelFrame(main, text="Settings", padding=10)
        opts.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        ttk.Label(opts, text="X column").grid(row=0, column=0, sticky="w")
        self.x_col_var = tk.StringVar(value="x")
        ttk.Entry(opts, textvariable=self.x_col_var, width=12).grid(row=0, column=1, padx=(4, 12))

        ttk.Label(opts, text="Y column").grid(row=0, column=2, sticky="w")
        self.y_col_var = tk.StringVar(value="y")
        ttk.Entry(opts, textvariable=self.y_col_var, width=12).grid(row=0, column=3, padx=(4, 12))

        ttk.Label(opts, text="Grid mode").grid(row=0, column=4, sticky="w")
        self.grid_mode_var = tk.StringVar(value="original_x")
        ttk.Combobox(
            opts,
            textvariable=self.grid_mode_var,
            values=("original_x", "digitized_x", "common_uniform"),
            state="readonly",
            width=18,
        ).grid(row=0, column=5, padx=(4, 12))

        ttk.Label(opts, text="Grid points").grid(row=0, column=6, sticky="w")
        self.grid_points_var = tk.StringVar(value="1000")
        ttk.Entry(opts, textvariable=self.grid_points_var, width=8).grid(row=0, column=7, padx=(4, 12))

        ttk.Label(opts, text="Duplicate X").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.dup_policy_var = tk.StringVar(value="median")
        ttk.Combobox(
            opts,
            textvariable=self.dup_policy_var,
            values=("median", "mean", "first"),
            state="readonly",
            width=10,
        ).grid(row=1, column=1, padx=(4, 12), pady=(8, 0))

        self.optimize_shift_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            opts,
            text="Optimize constant X shift",
            variable=self.optimize_shift_var,
        ).grid(row=1, column=2, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Label(opts, text="Max |X shift| (blank=auto)").grid(row=1, column=4, sticky="w", pady=(8, 0))
        self.max_shift_var = tk.StringVar(value="")
        ttk.Entry(opts, textvariable=self.max_shift_var, width=14).grid(row=1, column=5, padx=(4, 12), pady=(8, 0))

        ttk.Label(opts, text="Search steps").grid(row=1, column=6, sticky="w", pady=(8, 0))
        self.shift_steps_var = tk.StringVar(value="121")
        ttk.Entry(opts, textvariable=self.shift_steps_var, width=8).grid(row=1, column=7, padx=(4, 12), pady=(8, 0))

        actions = ttk.Frame(opts)
        actions.grid(row=2, column=0, columnspan=8, sticky="ew", pady=(10, 0))
        actions.columnconfigure(5, weight=1)

        self.process_btn = ttk.Button(actions, text="Process & Compare", command=self.process_data, state="disabled")
        self.process_btn.grid(row=0, column=0, padx=(0, 6))

        self.export_btn = ttk.Button(actions, text="Export Comparison CSV", command=self.export_results_csv, state="disabled")
        self.export_btn.grid(row=0, column=1, padx=(0, 6))

        self.prev_outlier_btn = ttk.Button(
            actions,
            text="Prev Outlier",
            command=lambda: self._change_outlier(-1),
            state="disabled",
        )
        self.prev_outlier_btn.grid(row=0, column=2, padx=(0, 6))

        self.next_outlier_btn = ttk.Button(
            actions,
            text="Next Outlier",
            command=lambda: self._change_outlier(1),
            state="disabled",
        )
        self.next_outlier_btn.grid(row=0, column=3, padx=(0, 6))

        self.outlier_label = ttk.Label(actions, textvariable=self.outlier_label_var)
        self.outlier_label.grid(row=0, column=4, sticky="w", padx=(0, 10))

        self.info_label = ttk.Label(actions, text="")
        self.info_label.grid(row=0, column=5, sticky="w")

        pane = ttk.PanedWindow(main, orient=tk.VERTICAL)
        pane.grid(row=4, column=0, columnspan=2, sticky="nsew")

        results_frame = ttk.LabelFrame(pane, text="Results", padding=8)
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        self.results_text = tk.Text(results_frame, height=14, wrap=tk.WORD)
        self.results_text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.results_text.configure(yscrollcommand=scroll.set)
        pane.add(results_frame, weight=1)

        self.plot_frame = ttk.LabelFrame(pane, text="Plots", padding=8)
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)
        pane.add(self.plot_frame, weight=3)

    def browse_file(self, file_type: str) -> None:
        path = filedialog.askopenfilename(
            title=f"Select {file_type} CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.load_file(path, file_type)

    def drop_file(self, event, file_type: str) -> None:
        files = self.root.tk.splitlist(event.data)
        if files:
            self.load_file(files[0], file_type)

    def load_file(self, path: str, file_type: str) -> None:
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            messagebox.showerror("Load Error", f"Failed to load CSV:\n{path}\n\n{exc}")
            return

        cols = ", ".join(map(str, df.columns))
        summary = f"{len(df)} rows\nColumns: {cols}"
        if file_type == "original":
            self.original_df = df
            self.orig_label.config(text=f"Original CSV Loaded\n{summary}")
        else:
            self.digitized_df = df
            self.dig_label.config(text=f"Digitized/Synth CSV Loaded\n{summary}")

        if self.original_df is not None and self.digitized_df is not None:
            self.process_btn.config(state="normal")
            self.status_var.set("Ready to compare.")

    def _parse_int(self, text: str, default: int, min_value: int = 1) -> int:
        t = str(text).strip()
        if not t:
            return default
        value = int(t)
        if value < min_value:
            raise ValueError(f"Value must be >= {min_value}: {text}")
        return value

    def _parse_float(self, text: str) -> Optional[float]:
        t = str(text).strip()
        if not t:
            return None
        return float(t)

    def _change_outlier(self, delta: int) -> None:
        if not self.result or not self.result.get("outliers"):
            return
        count = len(self.result["outliers"])
        self.current_outlier_index = max(0, min(count - 1, self.current_outlier_index + int(delta)))
        self._update_outlier_controls()
        self._draw_plots()

    def _update_outlier_controls(self) -> None:
        outliers = self.result.get("outliers", []) if self.result else []
        if not outliers:
            self.current_outlier_index = 0
            self.prev_outlier_btn.config(state="disabled")
            self.next_outlier_btn.config(state="disabled")
            self.outlier_label_var.set("Outlier zoom: n/a")
            return

        count = len(outliers)
        self.current_outlier_index = max(0, min(count - 1, self.current_outlier_index))
        x0, residual0, abs_err0 = outliers[self.current_outlier_index]
        self.prev_outlier_btn.config(state="normal" if self.current_outlier_index > 0 else "disabled")
        self.next_outlier_btn.config(state="normal" if self.current_outlier_index < count - 1 else "disabled")
        self.outlier_label_var.set(
            f"Outlier {self.current_outlier_index + 1}/{count}  x={x0:.3f}  resid={residual0:.3f}  |e|={abs_err0:.3f}"
        )

    def _prepare_series(self, df: pd.DataFrame, x_col: str, y_col: str, dup_policy: str) -> SeriesData:
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

    def _comparison_grid(
        self,
        orig_x: np.ndarray,
        orig_y: np.ndarray,
        dig_x: np.ndarray,
        dig_y: np.ndarray,
        mode: str,
        grid_points: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
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

    def _shift_score(
        self,
        orig_x: np.ndarray,
        orig_y: np.ndarray,
        dig_x: np.ndarray,
        dig_y: np.ndarray,
        shift: float,
    ) -> float:
        shifted = dig_x + shift
        start = max(float(orig_x[0]), float(shifted[0]))
        end = min(float(orig_x[-1]), float(shifted[-1]))
        if end <= start:
            return float("inf")
        x = np.linspace(start, end, 800, dtype=float)
        y1 = np.interp(x, orig_x, orig_y)
        y2 = np.interp(x, shifted, dig_y)
        return float(np.sqrt(np.mean((y1 - y2) ** 2)))

    def _optimize_x_shift(
        self,
        orig_x: np.ndarray,
        orig_y: np.ndarray,
        dig_x: np.ndarray,
        dig_y: np.ndarray,
        max_abs_shift: Optional[float],
        steps: int,
    ) -> Tuple[float, Dict[str, object]]:
        x_span = max(float(orig_x[-1] - orig_x[0]), float(dig_x[-1] - dig_x[0]))
        if max_abs_shift is None:
            max_abs_shift = 0.05 * x_span
        max_abs_shift = abs(float(max_abs_shift))
        if max_abs_shift == 0:
            return 0.0, {"reason": "zero_search_window", "best_rmse": np.nan}

        steps = max(5, int(steps))
        if steps % 2 == 0:
            steps += 1

        coarse = np.linspace(-max_abs_shift, max_abs_shift, steps)
        coarse_scores = np.array([self._shift_score(orig_x, orig_y, dig_x, dig_y, s) for s in coarse], dtype=float)
        if not np.isfinite(coarse_scores).any():
            return 0.0, {"reason": "no_overlap_in_search", "best_rmse": np.nan}

        idx = int(np.nanargmin(coarse_scores))
        best_shift = float(coarse[idx])
        step = abs(float(coarse[1] - coarse[0])) if coarse.size > 1 else max_abs_shift

        fine_left = max(-max_abs_shift, best_shift - step)
        fine_right = min(max_abs_shift, best_shift + step)
        fine = np.linspace(fine_left, fine_right, 81)
        fine_scores = np.array([self._shift_score(orig_x, orig_y, dig_x, dig_y, s) for s in fine], dtype=float)
        if np.isfinite(fine_scores).any():
            fidx = int(np.nanargmin(fine_scores))
            best_shift = float(fine[fidx])
            best_rmse = float(fine_scores[fidx])
        else:
            best_rmse = float(coarse_scores[idx])

        return best_shift, {
            "reason": "optimized",
            "search_max_abs_shift": max_abs_shift,
            "best_rmse": best_rmse,
            "steps": steps,
        }

    def _compute_metrics(self, y_ref: np.ndarray, y_cmp: np.ndarray) -> Dict[str, float]:
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

    def process_data(self) -> None:
        if self.original_df is None or self.digitized_df is None:
            messagebox.showerror("Missing Data", "Load both CSV files first.")
            return

        try:
            x_col = self.x_col_var.get().strip()
            y_col = self.y_col_var.get().strip()
            if not x_col or not y_col:
                raise ValueError("X and Y column names are required.")

            grid_mode = self.grid_mode_var.get().strip()
            grid_points = self._parse_int(self.grid_points_var.get(), default=1000, min_value=2)
            dup_policy = self.dup_policy_var.get().strip()
            shift_steps = self._parse_int(self.shift_steps_var.get(), default=121, min_value=5)
            max_abs_shift = self._parse_float(self.max_shift_var.get())

            orig = self._prepare_series(self.original_df, x_col, y_col, dup_policy)
            dig = self._prepare_series(self.digitized_df, x_col, y_col, dup_policy)
            if orig.x.size < 2 or dig.x.size < 2:
                raise ValueError("Each dataset must contain at least 2 valid unique points after cleanup.")

            x_shift = 0.0
            shift_meta: Dict[str, object] = {"reason": "disabled", "best_rmse": np.nan}
            if bool(self.optimize_shift_var.get()):
                x_shift, shift_meta = self._optimize_x_shift(
                    orig.x, orig.y, dig.x, dig.y, max_abs_shift=max_abs_shift, steps=shift_steps
                )

            dig_shifted_x = dig.x + x_shift
            x_cmp, y_ref, y_cmp, cmp_meta = self._comparison_grid(
                orig.x, orig.y, dig_shifted_x, dig.y, grid_mode, grid_points
            )
            residual = y_ref - y_cmp
            abs_err = np.abs(residual)
            metrics = self._compute_metrics(y_ref, y_cmp)

            overlap_span = float(cmp_meta["overlap_end"] - cmp_meta["overlap_start"])
            orig_span = float(orig.x[-1] - orig.x[0]) if orig.x.size > 1 else 0.0
            dig_span = float(dig_shifted_x[-1] - dig_shifted_x[0]) if dig_shifted_x.size > 1 else 0.0
            orig_cov = float(overlap_span / orig_span * 100.0) if orig_span > 0 else 0.0
            dig_cov = float(overlap_span / dig_span * 100.0) if dig_span > 0 else 0.0

            top_idx = np.argsort(abs_err)[-10:][::-1]
            outliers = [
                (float(x_cmp[i]), float(residual[i]), float(abs_err[i])) for i in top_idx
            ]

            self.result = {
                "x": x_cmp,
                "y_ref": y_ref,
                "y_cmp": y_cmp,
                "residual": residual,
                "abs_err": abs_err,
                "orig": orig,
                "dig": dig,
                "dig_shifted_x": dig_shifted_x,
                "metrics": metrics,
                "grid_mode": grid_mode,
                "grid_points": int(grid_points),
                "overlap_start": float(cmp_meta["overlap_start"]),
                "overlap_end": float(cmp_meta["overlap_end"]),
                "overlap_span": overlap_span,
                "orig_cov_pct": orig_cov,
                "dig_cov_pct": dig_cov,
                "x_shift": float(x_shift),
                "shift_meta": shift_meta,
                "outliers": outliers,
            }

            self.current_outlier_index = 0
            self.export_btn.config(state="normal")
            self._update_outlier_controls()
            self._show_results()
            self._draw_plots()
            self.status_var.set("Comparison complete.")
            self.info_label.config(text=f"Compared {len(x_cmp)} points over overlap [{cmp_meta['overlap_start']:.4g}, {cmp_meta['overlap_end']:.4g}]")
        except Exception as exc:
            self.status_var.set("Comparison failed.")
            messagebox.showerror("Processing Error", str(exc))

    def export_results_csv(self) -> None:
        if not self.result:
            return
        path = filedialog.asksaveasfilename(
            title="Export comparison CSV",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        pd.DataFrame(
            {
                "x": self.result["x"],
                "y_original": self.result["y_ref"],
                "y_digitized_interp": self.result["y_cmp"],
                "residual": self.result["residual"],
                "abs_error": self.result["abs_err"],
            }
        ).to_csv(path, index=False)
        self.status_var.set(f"Exported comparison CSV: {path}")

    def _fmt(self, value: object, suffix: str = "") -> str:
        try:
            f = float(value)
        except Exception:
            return "n/a"
        if not np.isfinite(f):
            return "n/a"
        return f"{f:.6f}{suffix}"

    def _show_results(self) -> None:
        if not self.result:
            return
        r = self.result
        m = r["metrics"]
        orig: SeriesData = r["orig"]
        dig: SeriesData = r["dig"]
        shift_meta = r.get("shift_meta", {})

        lines = []
        lines.append("ACCURACY METRICS")
        lines.append("=" * 72)
        lines.append(f"Compared points                 : {len(r['x'])}")
        lines.append(f"MAE                            : {self._fmt(m['mae'])}")
        lines.append(f"Median Abs Error               : {self._fmt(m['median_ae'])}")
        lines.append(f"P95 Abs Error                  : {self._fmt(m['p95_ae'])}")
        lines.append(f"Max Abs Error                  : {self._fmt(m['max_ae'])}")
        lines.append(f"RMSE                           : {self._fmt(m['rmse'])}")
        lines.append(f"NRMSE (% of Y range)           : {self._fmt(m['nrmse_range_pct'], '%')}")
        lines.append(f"Bias (mean residual)           : {self._fmt(m['bias'])}")
        lines.append(f"Residual std                   : {self._fmt(m['std_residual'])}")
        lines.append(f"R-squared                      : {self._fmt(m['r2'])}")
        lines.append(f"Correlation                    : {self._fmt(m['corr'])}")
        lines.append(f"MAPE                           : {self._fmt(m['mape_pct'], '%')}")
        lines.append(f"sMAPE                          : {self._fmt(m['smape_pct'], '%')}")
        lines.append(f"WAPE                           : {self._fmt(m['wape_pct'], '%')}")
        lines.append("")
        lines.append("COMPARISON SETUP")
        lines.append("=" * 72)
        lines.append(f"Grid mode                       : {r['grid_mode']}")
        lines.append(f"Overlap X range                 : [{r['overlap_start']:.6f}, {r['overlap_end']:.6f}]")
        lines.append(f"Overlap span                    : {r['overlap_span']:.6f}")
        lines.append(f"Original overlap coverage       : {r['orig_cov_pct']:.2f}%")
        lines.append(f"Digitized overlap coverage      : {r['dig_cov_pct']:.2f}%")
        lines.append(f"X shift applied to digitized X  : {r['x_shift']:.6f}")
        lines.append(f"X shift search status           : {shift_meta.get('reason', 'n/a')}")
        lines.append(f"X shift search best RMSE        : {self._fmt(shift_meta.get('best_rmse'))}")
        lines.append("")
        lines.append("DATA CLEANUP SUMMARY")
        lines.append("=" * 72)
        lines.append(
            f"Original raw/valid/unique       : {orig.raw_rows} / {orig.valid_rows} / {orig.unique_rows} "
            f"(duplicate X rows collapsed: {orig.duplicate_rows})"
        )
        lines.append(
            f"Digitized raw/valid/unique      : {dig.raw_rows} / {dig.valid_rows} / {dig.unique_rows} "
            f"(duplicate X rows collapsed: {dig.duplicate_rows})"
        )
        lines.append("")
        lines.append("TOP ABSOLUTE-ERROR POINTS (x, residual, abs_error)")
        lines.append("=" * 72)
        for x, res, ae in r["outliers"]:
            lines.append(f"{x:.6f}, {res:.6f}, {ae:.6f}")
        if r["orig_cov_pct"] < 95.0 or r["dig_cov_pct"] < 95.0:
            lines.append("")
            lines.append("NOTE: Limited X overlap can create large edge residuals.")

        self.results_text.delete("1.0", tk.END)
        self.results_text.insert("1.0", "\n".join(lines))

    def _draw_plots(self) -> None:
        if not self.result:
            return
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None

        r = self.result
        fig, axes = plt.subplots(3, 2, figsize=(12.5, 11.0))
        ax_curve, ax_resid, ax_abs, ax_hist, ax_zoom_curve, ax_zoom_err = axes.flatten()

        orig: SeriesData = r["orig"]
        dig: SeriesData = r["dig"]
        dig_shifted_x = r["dig_shifted_x"]
        x = r["x"]
        y_ref = r["y_ref"]
        y_cmp = r["y_cmp"]
        residual = r["residual"]
        abs_err = r["abs_err"]
        outliers = r.get("outliers", [])
        selected_outlier = None
        if outliers:
            idx = max(0, min(len(outliers) - 1, self.current_outlier_index))
            selected_outlier = outliers[idx]
            x_focus = float(selected_outlier[0])
        else:
            x_focus = None

        ax_curve.plot(orig.x, orig.y, color="#1565c0", linewidth=1.5, label="Original")
        ax_curve.plot(dig_shifted_x, dig.y, color="#2e7d32", linewidth=1.3, alpha=0.9, label="Digitized (shifted)")
        step = max(1, len(x) // 1500)
        ax_curve.scatter(x[::step], y_ref[::step], s=8, alpha=0.45, color="#0d47a1", label="Compare grid (orig)")
        ax_curve.scatter(x[::step], y_cmp[::step], s=8, alpha=0.45, color="#c62828", label="Compare grid (interp)")
        ax_curve.set_title("Curve Overlay")
        ax_curve.set_xlabel("X")
        ax_curve.set_ylabel("Y")
        ax_curve.grid(True, alpha=0.25)
        ax_curve.legend(fontsize=8)
        if x_focus is not None:
            ax_curve.axvline(x_focus, color="#ff8f00", linestyle="--", linewidth=1, alpha=0.9)

        rmse = float(r["metrics"]["rmse"])
        ax_resid.scatter(x, residual, s=10, alpha=0.55, color="#8e24aa")
        ax_resid.axhline(0.0, color="black", linewidth=1)
        if np.isfinite(rmse):
            ax_resid.axhline(rmse, color="#ef6c00", linestyle="--", linewidth=1, alpha=0.8)
            ax_resid.axhline(-rmse, color="#ef6c00", linestyle="--", linewidth=1, alpha=0.8)
        ax_resid.set_title("Residuals vs X (Original - Interpolated)")
        ax_resid.set_xlabel("X")
        ax_resid.set_ylabel("Residual")
        ax_resid.grid(True, alpha=0.25)
        if x_focus is not None:
            ax_resid.axvline(x_focus, color="#ff8f00", linestyle="--", linewidth=1, alpha=0.9)

        p95 = float(r["metrics"]["p95_ae"])
        ax_abs.scatter(x, abs_err, s=10, alpha=0.55, color="#d32f2f")
        if np.isfinite(p95):
            ax_abs.axhline(p95, color="#1e88e5", linestyle="--", linewidth=1, alpha=0.8, label=f"P95={p95:.3g}")
        ax_abs.set_title("Absolute Error vs X")
        ax_abs.set_xlabel("X")
        ax_abs.set_ylabel("|Residual|")
        ax_abs.grid(True, alpha=0.25)
        ax_abs.legend(fontsize=8)
        if x_focus is not None:
            ax_abs.axvline(x_focus, color="#ff8f00", linestyle="--", linewidth=1, alpha=0.9)

        bins = min(80, max(20, len(residual) // 20))
        ax_hist.hist(residual, bins=bins, color="#546e7a", alpha=0.85, edgecolor="white")
        ax_hist.axvline(0.0, color="black", linewidth=1)
        ax_hist.axvline(float(np.mean(residual)), color="#d32f2f", linestyle="--", linewidth=1)
        ax_hist.set_title("Residual Histogram")
        ax_hist.set_xlabel("Residual")
        ax_hist.set_ylabel("Count")
        ax_hist.grid(True, alpha=0.2, axis="y")

        if selected_outlier is None:
            ax_zoom_curve.set_axis_off()
            ax_zoom_err.set_axis_off()
        else:
            x0, resid0, ae0 = selected_outlier
            x = np.asarray(x)
            if len(x) > 2:
                dx = np.diff(x)
                dx = dx[np.isfinite(dx) & (dx > 0)]
                median_dx = float(np.median(dx)) if dx.size else 1.0
            else:
                median_dx = 1.0
            overlap_span = float(r.get("overlap_span", 0.0))
            half_window = max(overlap_span * 0.03, median_dx * 40.0)
            if overlap_span > 0:
                half_window = min(half_window, overlap_span * 0.25)
            if half_window <= 0:
                half_window = max(median_dx * 40.0, 1.0)

            for _ in range(5):
                mask_zoom = (x >= x0 - half_window) & (x <= x0 + half_window)
                if int(np.count_nonzero(mask_zoom)) >= 20 or half_window >= max(overlap_span, 1.0):
                    break
                half_window *= 1.75
            mask_zoom = (x >= x0 - half_window) & (x <= x0 + half_window)

            xz = x[mask_zoom]
            y_ref_z = y_ref[mask_zoom]
            y_cmp_z = y_cmp[mask_zoom]
            resid_z = residual[mask_zoom]
            abs_err_z = abs_err[mask_zoom]

            raw_pad = half_window * 1.2
            orig_raw_mask = (orig.x >= x0 - raw_pad) & (orig.x <= x0 + raw_pad)
            dig_raw_mask = (dig_shifted_x >= x0 - raw_pad) & (dig_shifted_x <= x0 + raw_pad)

            ax_zoom_curve.plot(orig.x[orig_raw_mask], orig.y[orig_raw_mask], color="#1565c0", linewidth=1.5, label="Original")
            ax_zoom_curve.plot(dig_shifted_x[dig_raw_mask], dig.y[dig_raw_mask], color="#2e7d32", linewidth=1.3, alpha=0.9, label="Digitized")
            if xz.size:
                step_z = max(1, len(xz) // 500)
                ax_zoom_curve.scatter(xz[::step_z], y_ref_z[::step_z], s=10, alpha=0.55, color="#0d47a1")
                ax_zoom_curve.scatter(xz[::step_z], y_cmp_z[::step_z], s=10, alpha=0.55, color="#c62828")
            ax_zoom_curve.axvline(x0, color="#ff8f00", linestyle="--", linewidth=1)
            ax_zoom_curve.set_title(
                f"Outlier Zoom (rank {self.current_outlier_index + 1}) at x={x0:.3f}, |e|={ae0:.3f}"
            )
            ax_zoom_curve.set_xlabel("X")
            ax_zoom_curve.set_ylabel("Y")
            ax_zoom_curve.grid(True, alpha=0.25)
            ax_zoom_curve.legend(fontsize=8)

            if xz.size:
                ax_zoom_err.scatter(xz, np.abs(resid_z), s=12, alpha=0.6, color="#d32f2f", label="|Residual|")
                ax_zoom_err.plot(xz, resid_z, color="#8e24aa", linewidth=1.0, alpha=0.75, label="Residual")
            ax_zoom_err.axhline(0.0, color="black", linewidth=1)
            ax_zoom_err.axvline(x0, color="#ff8f00", linestyle="--", linewidth=1)
            ax_zoom_err.set_title("Outlier Zoom Error View")
            ax_zoom_err.set_xlabel("X")
            ax_zoom_err.set_ylabel("Error")
            ax_zoom_err.grid(True, alpha=0.25)
            ax_zoom_err.legend(fontsize=8)

        fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")


def main() -> None:
    root = TkinterDnD.Tk() if HAS_DND else tk.Tk()
    AccuracyTesterPro(root)
    root.mainloop()


if __name__ == "__main__":
    main()

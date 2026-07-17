"""Unit tests for accuracy_core.

Expected values are hand-computed (worked out independently on paper / with
plain scalar arithmetic written inline), never by running the function under
test and asserting it equals itself. Reference-library cross-checks compare
RMSE / R^2 / correlation / MAE against sklearn and scipy implementations on
the same synthetic data.

Run:  py -m pytest test_accuracy_core.py -v   (from the AccuracyTester folder)
"""

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from accuracy_core import comparison_grid, compute_metrics, prepare_series

TOL = 1e-9  # absolute tolerance for hand-computed comparisons
REF_TOL = 1e-10  # tolerance vs reference library implementations


# ---------------------------------------------------------------------------
# prepare_series
# ---------------------------------------------------------------------------

class TestPrepareSeries:
    def test_missing_columns_raise(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="not found"):
            prepare_series(df, "x", "y", "median")

    def test_empty_dataframe_raises(self):
        df = pd.DataFrame({"x": [], "y": []})
        with pytest.raises(ValueError, match="No valid numeric rows"):
            prepare_series(df, "x", "y", "median")

    def test_all_invalid_rows_raise(self):
        df = pd.DataFrame({"x": ["a", np.nan, np.inf], "y": ["b", 1.0, 2.0]})
        with pytest.raises(ValueError, match="No valid numeric rows"):
            prepare_series(df, "x", "y", "median")

    def test_nan_and_inf_rows_dropped(self):
        df = pd.DataFrame(
            {"x": [1.0, 2.0, np.nan, 4.0, 5.0], "y": [1.0, np.inf, 3.0, -np.inf, 5.0]}
        )
        s = prepare_series(df, "x", "y", "median")
        # rows 2 (inf y), 3 (nan x), 4 (-inf y) all drop -> x=[1,5]
        assert s.raw_rows == 5
        assert s.valid_rows == 2
        assert s.duplicate_rows == 0
        assert s.unique_rows == 2
        np.testing.assert_allclose(s.x, [1.0, 5.0], atol=TOL)
        np.testing.assert_allclose(s.y, [1.0, 5.0], atol=TOL)

    def test_non_monotonic_x_is_sorted(self):
        df = pd.DataFrame({"x": [3.0, 1.0, 2.0], "y": [30.0, 10.0, 20.0]})
        s = prepare_series(df, "x", "y", "median")
        np.testing.assert_allclose(s.x, [1.0, 2.0, 3.0], atol=TOL)
        np.testing.assert_allclose(s.y, [10.0, 20.0, 30.0], atol=TOL)

    # x=1 appears 3x with y in file order 1, 9, 2 -> median 2, mean 4, first 1
    DUP = pd.DataFrame({"x": [1.0, 2.0, 1.0, 1.0], "y": [1.0, 7.0, 9.0, 2.0]})

    def test_duplicate_policy_median(self):
        s = prepare_series(self.DUP, "x", "y", "median")
        assert s.duplicate_rows == 2
        assert s.unique_rows == 2
        np.testing.assert_allclose(s.x, [1.0, 2.0], atol=TOL)
        np.testing.assert_allclose(s.y, [2.0, 7.0], atol=TOL)

    def test_duplicate_policy_mean(self):
        s = prepare_series(self.DUP, "x", "y", "mean")
        np.testing.assert_allclose(s.y, [4.0, 7.0], atol=TOL)

    def test_duplicate_policy_first(self):
        s = prepare_series(self.DUP, "x", "y", "first")
        np.testing.assert_allclose(s.y, [1.0, 7.0], atol=TOL)

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="median, mean, first"):
            prepare_series(self.DUP, "x", "y", "sum")

    def test_numeric_strings_coerced(self):
        df = pd.DataFrame({"x": ["1", "2"], "y": ["3.5", "bad"]})
        s = prepare_series(df, "x", "y", "median")
        assert s.valid_rows == 1
        np.testing.assert_allclose(s.x, [1.0], atol=TOL)
        np.testing.assert_allclose(s.y, [3.5], atol=TOL)


# ---------------------------------------------------------------------------
# comparison_grid
# ---------------------------------------------------------------------------

class TestComparisonGrid:
    def test_non_overlapping_ranges_raise(self):
        with pytest.raises(ValueError, match="No overlapping X range"):
            comparison_grid(
                np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                np.array([2.0, 3.0]), np.array([0.0, 1.0]),
                "original_x", 1000,
            )

    def test_touching_ranges_raise(self):
        # overlap_end == overlap_start is rejected (needs strictly positive span)
        with pytest.raises(ValueError, match="No overlapping X range"):
            comparison_grid(
                np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                np.array([1.0, 2.0]), np.array([0.0, 1.0]),
                "original_x", 1000,
            )

    def test_too_few_points_in_overlap_raise(self):
        # overlap is [4.9, 5.1] but the original grid has no samples inside it
        with pytest.raises(ValueError, match="fewer than 2 points"):
            comparison_grid(
                np.array([0.0, 10.0]), np.array([0.0, 10.0]),
                np.array([4.9, 5.1]), np.array([1.0, 2.0]),
                "original_x", 1000,
            )

    def test_unsupported_mode_raises(self):
        with pytest.raises(ValueError, match="Unsupported grid mode"):
            comparison_grid(
                np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                "nearest", 1000,
            )

    def test_original_x_mode_hand_computed(self):
        # orig: y = 10x on x=0..3; dig: same line sampled at 0.5 and 2.5
        x_cmp, y_ref, y_cmp, meta = comparison_grid(
            np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 10.0, 20.0, 30.0]),
            np.array([0.5, 2.5]), np.array([5.0, 25.0]),
            "original_x", 1000,
        )
        assert meta["overlap_start"] == 0.5
        assert meta["overlap_end"] == 2.5
        np.testing.assert_allclose(x_cmp, [1.0, 2.0], atol=TOL)
        np.testing.assert_allclose(y_ref, [10.0, 20.0], atol=TOL)
        # linear interp of (0.5,5)-(2.5,25) at x=1: 5 + (0.5/2)*20 = 10; at 2: 20
        np.testing.assert_allclose(y_cmp, [10.0, 20.0], atol=TOL)

    def test_digitized_x_mode_hand_computed(self):
        # dig grid is kept, original is interpolated onto it
        x_cmp, y_ref, y_cmp, _ = comparison_grid(
            np.array([0.0, 2.0]), np.array([0.0, 20.0]),
            np.array([0.5, 1.0, 1.5, 2.5]), np.array([7.0, 8.0, 9.0, 10.0]),
            "digitized_x", 1000,
        )
        # overlap [0.5, 2.0]; dig points inside: 0.5, 1.0, 1.5
        np.testing.assert_allclose(x_cmp, [0.5, 1.0, 1.5], atol=TOL)
        np.testing.assert_allclose(y_ref, [5.0, 10.0, 15.0], atol=TOL)  # y = 10x
        np.testing.assert_allclose(y_cmp, [7.0, 8.0, 9.0], atol=TOL)

    def test_common_uniform_mode_hand_computed(self):
        x_cmp, y_ref, y_cmp, _ = comparison_grid(
            np.array([0.0, 4.0]), np.array([0.0, 8.0]),   # y = 2x
            np.array([1.0, 5.0]), np.array([3.0, 15.0]),  # y = 3x
            "common_uniform", 5,
        )
        # overlap [1, 4]; 5 uniform points: 1, 1.75, 2.5, 3.25, 4
        np.testing.assert_allclose(x_cmp, [1.0, 1.75, 2.5, 3.25, 4.0], atol=TOL)
        np.testing.assert_allclose(y_ref, [2.0, 3.5, 5.0, 6.5, 8.0], atol=TOL)
        np.testing.assert_allclose(y_cmp, [3.0, 5.25, 7.5, 9.75, 12.0], atol=TOL)

    def test_common_uniform_grid_points_floor_of_two(self):
        x_cmp, _, _, _ = comparison_grid(
            np.array([0.0, 4.0]), np.array([0.0, 8.0]),
            np.array([1.0, 5.0]), np.array([3.0, 15.0]),
            "common_uniform", 1,
        )
        np.testing.assert_allclose(x_cmp, [1.0, 4.0], atol=TOL)


# ---------------------------------------------------------------------------
# compute_metrics — hand-computed expectations
# ---------------------------------------------------------------------------

class TestComputeMetricsHandComputed:
    def test_perfect_match(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        m = compute_metrics(y, y.copy())
        for key in ("mse", "rmse", "mae", "median_ae", "p95_ae", "max_ae",
                    "bias", "std_residual", "mape_pct", "smape_pct", "wape_pct",
                    "nrmse_range_pct"):
            assert m[key] == pytest.approx(0.0, abs=TOL), key
        assert m["r2"] == pytest.approx(1.0, abs=TOL)
        assert m["corr"] == pytest.approx(1.0, abs=TOL)

    def test_constant_offset(self):
        # digitized reads 0.5 LOW everywhere -> residual (ref - cmp) = +0.5
        y_ref = np.array([1.0, 2.0, 3.0, 4.0])
        y_cmp = y_ref - 0.5
        m = compute_metrics(y_ref, y_cmp)
        assert m["mse"] == pytest.approx(0.25, abs=TOL)
        assert m["rmse"] == pytest.approx(0.5, abs=TOL)
        assert m["mae"] == pytest.approx(0.5, abs=TOL)
        assert m["median_ae"] == pytest.approx(0.5, abs=TOL)
        assert m["p95_ae"] == pytest.approx(0.5, abs=TOL)
        assert m["max_ae"] == pytest.approx(0.5, abs=TOL)
        assert m["bias"] == pytest.approx(0.5, abs=TOL)
        assert m["std_residual"] == pytest.approx(0.0, abs=TOL)
        # ss_res = 4 * 0.25 = 1; ss_tot = 2.25+0.25+0.25+2.25 = 5 -> r2 = 0.8
        assert m["r2"] == pytest.approx(0.8, abs=TOL)
        assert m["corr"] == pytest.approx(1.0, abs=TOL)
        # mape = mean(0.5/1, 0.5/2, 0.5/3, 0.5/4)*100 = 26.041666...%
        assert m["mape_pct"] == pytest.approx(100.0 * (0.5 + 0.25 + 0.5 / 3 + 0.125) / 4, abs=1e-6)
        # smape terms: 200*0.5/(|ref|+|cmp|) = 100/1.5, 100/3.5, 100/5.5, 100/7.5
        expected_smape = (100 / 1.5 + 100 / 3.5 + 100 / 5.5 + 100 / 7.5) / 4
        assert m["smape_pct"] == pytest.approx(expected_smape, abs=1e-6)
        # wape = sum|res| / sum|ref| = 2 / 10 -> 20%
        assert m["wape_pct"] == pytest.approx(20.0, abs=TOL)
        # nrmse = rmse / (4-1) * 100 = 16.666...%
        assert m["nrmse_range_pct"] == pytest.approx(50.0 / 3.0, abs=1e-6)

    def test_scaling_error(self):
        # digitized reads exactly double -> residual = -y_ref
        y_ref = np.array([1.0, 2.0, 3.0])
        y_cmp = 2.0 * y_ref
        m = compute_metrics(y_ref, y_cmp)
        assert m["mse"] == pytest.approx(14.0 / 3.0, abs=TOL)          # (1+4+9)/3
        assert m["rmse"] == pytest.approx(math.sqrt(14.0 / 3.0), abs=TOL)
        assert m["mae"] == pytest.approx(2.0, abs=TOL)
        assert m["median_ae"] == pytest.approx(2.0, abs=TOL)
        assert m["max_ae"] == pytest.approx(3.0, abs=TOL)
        # np.percentile linear interp: index 0.95*(3-1)=1.9 -> 2 + 0.9*(3-2)
        assert m["p95_ae"] == pytest.approx(2.9, abs=TOL)
        assert m["bias"] == pytest.approx(-2.0, abs=TOL)
        # population std of [-1,-2,-3]: sqrt(mean([1,0,1])) = sqrt(2/3)
        assert m["std_residual"] == pytest.approx(math.sqrt(2.0 / 3.0), abs=TOL)
        # ss_res = 14; ss_tot = (1-2)^2+(2-2)^2+(3-2)^2 = 2 -> r2 = 1 - 7 = -6
        assert m["r2"] == pytest.approx(-6.0, abs=TOL)
        assert m["corr"] == pytest.approx(1.0, abs=TOL)  # scaling keeps corr = 1
        assert m["mape_pct"] == pytest.approx(100.0, abs=TOL)
        assert m["wape_pct"] == pytest.approx(100.0, abs=TOL)
        # nrmse = rmse / range(y_ref)=2 * 100
        assert m["nrmse_range_pct"] == pytest.approx(50.0 * math.sqrt(14.0 / 3.0), abs=1e-6)

    def test_zero_variance_reference_gives_nan_r2_corr_nrmse(self):
        m = compute_metrics(np.array([2.0, 2.0, 2.0]), np.array([1.0, 2.0, 3.0]))
        assert math.isnan(m["r2"])
        assert math.isnan(m["corr"])
        assert math.isnan(m["nrmse_range_pct"])
        # mape still defined: mean(0.5, 0, 0.5)*100
        assert m["mape_pct"] == pytest.approx(100.0 / 3.0, abs=1e-6)

    def test_zero_variance_comparison_gives_nan_corr(self):
        m = compute_metrics(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0]))
        assert math.isnan(m["corr"])
        assert not math.isnan(m["r2"])  # reference varies, so R2 is defined

    def test_all_zero_reference_masks_percentage_errors(self):
        m = compute_metrics(np.array([0.0, 0.0, 0.0]), np.array([1.0, -1.0, 0.5]))
        assert math.isnan(m["mape_pct"])
        assert math.isnan(m["wape_pct"])
        assert math.isnan(m["r2"])
        assert math.isnan(m["nrmse_range_pct"])
        assert m["mae"] == pytest.approx(2.5 / 3.0, abs=TOL)

    def test_single_point(self):
        m = compute_metrics(np.array([1.0]), np.array([2.0]))
        assert m["mse"] == pytest.approx(1.0, abs=TOL)
        assert math.isnan(m["r2"])   # zero total sum of squares
        assert math.isnan(m["corr"])  # needs >= 2 points

    def test_near_zero_reference_values_masked_in_mape(self):
        # |y_ref| <= eps (=1e-9 * max|y_ref|) is excluded from MAPE
        y_ref = np.array([1e-15, 1.0, 2.0])
        y_cmp = np.array([1.0, 1.5, 2.5])
        m = compute_metrics(y_ref, y_cmp)
        # only the last two points count: mean(0.5/1, 0.25)*100 = 37.5
        assert m["mape_pct"] == pytest.approx(37.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Reference-library cross-checks (sklearn / scipy / numpy)
# ---------------------------------------------------------------------------

class TestReferenceCrossChecks:
    @pytest.fixture(scope="class")
    def synthetic(self):
        rng = np.random.default_rng(7)
        y_ref = np.sin(np.linspace(0, 6, 400)) * 3.0 + np.linspace(0, 2, 400)
        y_cmp = y_ref + rng.normal(0.0, 0.2, y_ref.size) + 0.05
        return y_ref, y_cmp, compute_metrics(y_ref, y_cmp)

    def test_rmse_vs_sklearn(self, synthetic):
        y_ref, y_cmp, m = synthetic
        ref_rmse = math.sqrt(mean_squared_error(y_ref, y_cmp))
        assert m["rmse"] == pytest.approx(ref_rmse, rel=REF_TOL)

    def test_mse_vs_sklearn(self, synthetic):
        y_ref, y_cmp, m = synthetic
        assert m["mse"] == pytest.approx(mean_squared_error(y_ref, y_cmp), rel=REF_TOL)

    def test_mae_vs_sklearn(self, synthetic):
        y_ref, y_cmp, m = synthetic
        assert m["mae"] == pytest.approx(mean_absolute_error(y_ref, y_cmp), rel=REF_TOL)

    def test_r2_vs_sklearn(self, synthetic):
        y_ref, y_cmp, m = synthetic
        assert m["r2"] == pytest.approx(r2_score(y_ref, y_cmp), rel=REF_TOL)

    def test_corr_vs_scipy_pearsonr(self, synthetic):
        y_ref, y_cmp, m = synthetic
        ref_corr = scipy_stats.pearsonr(y_ref, y_cmp).statistic
        assert m["corr"] == pytest.approx(ref_corr, rel=REF_TOL)

    def test_mse_equals_rmse_squared(self, synthetic):
        _, _, m = synthetic
        assert m["mse"] == pytest.approx(m["rmse"] ** 2, rel=1e-12)

    def test_mse_equals_rmse_squared_on_multiple_random_series(self):
        rng = np.random.default_rng(123)
        for _ in range(20):
            n = int(rng.integers(2, 300))
            y_ref = rng.normal(0, 10, n)
            y_cmp = y_ref + rng.normal(0, 1, n)
            m = compute_metrics(y_ref, y_cmp)
            assert m["mse"] == pytest.approx(m["rmse"] ** 2, rel=1e-12)
            assert m["rmse"] == pytest.approx(math.sqrt(mean_squared_error(y_ref, y_cmp)), rel=REF_TOL)
            assert m["r2"] == pytest.approx(r2_score(y_ref, y_cmp), rel=REF_TOL)


# ---------------------------------------------------------------------------
# End-to-end: prepare -> grid -> metrics on a tiny hand-checked example
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_pipeline_hand_checked(self):
        # ground truth: y = x on 0..4 ; digitized: y = x + 1 sampled off-grid
        gt = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0], "y": [0.0, 1.0, 2.0, 3.0, 4.0]})
        dg = pd.DataFrame({"x": [0.5, 1.5, 2.5, 3.5], "y": [1.5, 2.5, 3.5, 4.5]})
        s_gt = prepare_series(gt, "x", "y", "median")
        s_dg = prepare_series(dg, "x", "y", "median")
        x_cmp, y_ref, y_cmp, _ = comparison_grid(s_gt.x, s_gt.y, s_dg.x, s_dg.y, "original_x", 1000)
        # overlap [0.5, 3.5]; original x inside: 1, 2, 3; dig interp = x + 1 exactly
        np.testing.assert_allclose(x_cmp, [1.0, 2.0, 3.0], atol=TOL)
        m = compute_metrics(y_ref, y_cmp)
        assert m["rmse"] == pytest.approx(1.0, abs=TOL)
        assert m["mse"] == pytest.approx(1.0, abs=TOL)
        assert m["bias"] == pytest.approx(-1.0, abs=TOL)  # digitized reads high
        assert m["corr"] == pytest.approx(1.0, abs=TOL)
        # ss_res = 3, ss_tot = 2 -> r2 = -0.5
        assert m["r2"] == pytest.approx(-0.5, abs=TOL)

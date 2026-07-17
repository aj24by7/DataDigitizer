"""Unit tests for batch_scoring's matching and status taxonomy.

Covers the defects found in the adversarial review: suffix-stripping breaking
stems that themselves end in a strippable word, parse failures being
misclassified, lexicographic color-slot ordering, ambiguous matches being
resolved by guessing, and the report-file filter swallowing real data files.

Run:  py -m pytest test_batch_scoring.py -v   (from the AccuracyTester folder)
"""

from pathlib import Path

import pandas as pd
import pytest

from batch_scoring import (
    MatchSettings,
    ScoreSettings,
    index_folder,
    load_series_frame,
    match_pairs,
    score_pair,
)

SETTINGS = MatchSettings()


def write_series(path: Path, x=(0.0, 1.0, 2.0), y=(0.0, 1.0, 2.0)) -> Path:
    pd.DataFrame({"x": x, "y": y}).to_csv(path, index=False)
    return path


def indexes(tmp_path: Path, gt_names, dig_names):
    gt_dir = tmp_path / "gt"
    dig_dir = tmp_path / "dig"
    gt_dir.mkdir()
    dig_dir.mkdir()
    for name in gt_names:
        write_series(gt_dir / name)
    for name in dig_names:
        write_series(dig_dir / name)
    return index_folder(gt_dir, SETTINGS), index_folder(dig_dir, SETTINGS)


class TestMatching:
    def test_classic_suffix_match(self, tmp_path):
        gt, dig = indexes(tmp_path, ["chart_001.csv"], ["chart_001_digitized_points.csv"])
        result = match_pairs(gt, dig)
        assert len(result.pairs) == 1
        assert result.pairs[0][0] == "chart_001"
        assert not result.gt_unmatched and not result.method_unmatched and not result.ambiguous

    def test_stem_ending_in_strippable_suffix_still_matches(self, tmp_path):
        # 'sensor_data' ends in '_data'; the gt stem must not be over-stripped
        gt, dig = indexes(tmp_path, ["sensor_data.csv"], ["sensor_data_digitized_points.csv"])
        result = match_pairs(gt, dig)
        assert len(result.pairs) == 1
        assert result.pairs[0][0] == "sensor_data"
        assert not result.gt_unmatched and not result.method_unmatched

    def test_raw_equal_raw_matches(self, tmp_path):
        gt, dig = indexes(tmp_path, ["trace_gt.csv"], ["trace_gt.csv"])
        result = match_pairs(gt, dig)
        assert len(result.pairs) == 1

    def test_unmatched_both_sides_reported(self, tmp_path):
        gt, dig = indexes(tmp_path, ["a.csv", "b.csv"], ["b_digitized_points.csv", "orphan.csv"])
        result = match_pairs(gt, dig)
        assert [p[0] for p in result.pairs] == ["b"]
        assert [u[0] for u in result.gt_unmatched] == ["a"]
        assert [u[0] for u in result.method_unmatched] == ["orphan"]

    def test_ambiguous_match_not_guessed(self, tmp_path):
        # both normalize to 'chart_a'; neither is preferred - report, don't score
        gt, dig = indexes(tmp_path, ["chart_a.csv"],
                          ["chart_a_data.csv", "chart_a_digitized_points.csv"])
        result = match_pairs(gt, dig)
        assert not result.pairs
        assert len(result.ambiguous) == 1
        image_id, _gt_path, candidates = result.ambiguous[0]
        assert image_id == "chart_a"
        assert {p.name for p in candidates} == {"chart_a_data.csv", "chart_a_digitized_points.csv"}

    def test_exact_raw_match_beats_suffix_candidates(self, tmp_path):
        # an exact raw twin exists -> no ambiguity with the suffixed file
        gt, dig = indexes(tmp_path, ["chart_a.csv"],
                          ["chart_a.csv", "chart_a_digitized_points.csv"])
        result = match_pairs(gt, dig)
        assert len(result.pairs) == 1
        assert result.pairs[0][2].name == "chart_a.csv"
        assert [u[0] for u in result.method_unmatched] == ["chart_a_digitized_points"]

    def test_case_insensitive(self, tmp_path):
        gt, dig = indexes(tmp_path, ["Chart_B.csv"], ["chart_b_digitized_points.csv"])
        result = match_pairs(gt, dig)
        assert len(result.pairs) == 1


class TestReportFileFilter:
    def test_exact_report_names_ignored_but_lookalike_data_kept(self, tmp_path):
        folder = tmp_path / "dig"
        folder.mkdir()
        write_series(folder / "accuracy_curve_digitized_points.csv")  # real data
        write_series(folder / "batch_report.csv")                     # pipeline report
        write_series(folder / "accuracy_report.csv")                  # pipeline report
        index = index_folder(folder, SETTINGS)
        assert [e.path.name for e in index.entries] == ["accuracy_curve_digitized_points.csv"]
        assert sorted(index.ignored) == ["accuracy_report.csv", "batch_report.csv"]


class TestStatusTaxonomy:
    def test_unreadable_csv_is_parse_error(self, tmp_path):
        gt = write_series(tmp_path / "gt.csv")
        empty = tmp_path / "empty.csv"
        empty.write_bytes(b"")
        score = score_pair(gt, empty, ScoreSettings())
        assert score.status == "parse_error"
        assert "empty.csv" in score.reason

    def test_malformed_csv_is_parse_error(self, tmp_path):
        gt = write_series(tmp_path / "gt.csv")
        bad = tmp_path / "bad.csv"
        bad.write_text('a,b\n"unclosed quote, 1\n2,3,4,5\n', encoding="utf-8")
        score = score_pair(gt, bad, ScoreSettings())
        assert score.status == "parse_error"

    def test_wrong_columns_is_invalid_data(self, tmp_path):
        gt = write_series(tmp_path / "gt.csv")
        wrong = tmp_path / "wrong.csv"
        pd.DataFrame({"time": [1, 2], "value": [3, 4]}).to_csv(wrong, index=False)
        score = score_pair(gt, wrong, ScoreSettings())
        assert score.status == "invalid_data"

    def test_disjoint_ranges_is_insufficient_overlap(self, tmp_path):
        gt = write_series(tmp_path / "gt.csv", x=(0.0, 1.0), y=(0.0, 1.0))
        far = write_series(tmp_path / "far.csv", x=(10.0, 11.0), y=(0.0, 1.0))
        score = score_pair(gt, far, ScoreSettings())
        assert score.status == "insufficient_overlap"

    def test_missing_color_slot_is_invalid_data(self, tmp_path):
        gt = write_series(tmp_path / "gt.csv")
        slotted = tmp_path / "slotted.csv"
        pd.DataFrame({"color_slot": [1, 1], "x": [0.0, 1.0], "y": [0.0, 1.0]}).to_csv(slotted, index=False)
        score = score_pair(gt, slotted, ScoreSettings(color_slot="9"))
        assert score.status == "invalid_data"
        assert "requested color slot" in score.reason


class TestColorSlotOrdering:
    def test_numeric_slots_sort_numerically(self, tmp_path):
        path = tmp_path / "multi.csv"
        rows = []
        for slot in (10, 2):
            for x in (0.0, 1.0, 2.0):
                rows.append({"color_slot": slot, "x": x, "y": x * slot})
        pd.DataFrame(rows).to_csv(path, index=False)
        df, note = load_series_frame(path, ScoreSettings())
        assert set(df["color_slot"].unique()) == {2}
        assert "used slot 2" in note

    def test_explicit_slot_still_honored(self, tmp_path):
        path = tmp_path / "multi.csv"
        pd.DataFrame({"color_slot": [2, 10], "x": [0.0, 0.0], "y": [1.0, 2.0]}).to_csv(path, index=False)
        df, _note = load_series_frame(path, ScoreSettings(color_slot="10"))
        assert set(df["color_slot"].unique()) == {10}

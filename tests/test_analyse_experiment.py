"""Tests for the experiment analysis.

The interpretation function turns numbers into sentences, which is the most
dangerous code in the repository: its failure mode is a confident claim rather
than an exception. An earlier version reported "the advantage travels with the
angular geometry" from two zero-sized gaps, because a guard added to avoid
dividing by zero made the first branch fire on degenerate input. Each branch is
therefore driven here by constructed data whose correct reading is known in
advance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyse_experiment import (  # noqa: E402
    geometry_outcome_table,
    interpret,
    load_experiment,
    predictor_correlations,
    variant_summary,
    within_variant_correlations,
)


def _runs(
    variant: str,
    accuracies: list[float],
    concentration: list[float] | None = None,
    rms: list[float] | None = None,
) -> list[dict]:
    rows = []
    for seed, accuracy in enumerate(accuracies):
        rows.append(
            {
                "variant": variant,
                "seed": seed,
                "accuracy": accuracy,
                "placement_concentration": (
                    concentration[seed] if concentration else 1.0
                ),
                "placement_residual_rms": rms[seed] if rms else 0.05,
                "placement_nullspace_fraction": 0.97,
                "placement_logit_shift_rms": 0.01,
            }
        )
    return rows


def _experiment(runs: list[dict], comparisons: list[dict], geometry=None) -> dict:
    return {
        "backbone": "test",
        "runs": runs,
        "comparisons": comparisons,
        "geometry": geometry or {},
    }


def _comparison(variant: str, difference: float, p_value: float = 0.5, n: int = 15):
    return {
        "variant_a": "source",
        "variant_b": variant,
        "mean_paired_difference": difference,
        "unpaired_p": p_value,
        "pooled_sd": 1.5,
        "n_seeds": n,
    }


class TestVariantSummary:
    def test_computes_mean_and_sample_std(self):
        runs = _runs("source", [30.0, 32.0, 34.0])
        summary = variant_summary(runs)
        assert summary["source"]["mean"] == pytest.approx(32.0)
        assert summary["source"]["std"] == pytest.approx(2.0)
        assert summary["source"]["n_seeds"] == 3

    def test_bootstrap_interval_brackets_the_mean(self):
        runs = _runs("source", [30.0, 31.0, 32.0, 33.0, 34.0])
        entry = variant_summary(runs)["source"]
        assert entry["ci_low"] <= entry["mean"] <= entry["ci_high"]

    def test_handles_several_variants(self):
        runs = _runs("source", [33.0, 34.0]) + _runs("etf", [29.0, 30.0])
        summary = variant_summary(runs)
        assert set(summary) == {"source", "etf"}
        assert summary["source"]["mean"] > summary["etf"]["mean"]


class TestPredictorCorrelations:
    def test_detects_a_planted_relationship(self):
        """Concentration is constructed to track accuracy inversely."""
        runs = _runs(
            "source",
            [30.0, 31.0, 32.0, 33.0, 34.0],
            concentration=[1.5, 1.4, 1.3, 1.2, 1.1],
        )
        results = {row["predictor"]: row for row in predictor_correlations(runs)}
        assert results["placement_concentration"]["pearson_r"] < -0.95

    def test_reports_no_relationship_when_there_is_none(self):
        runs = _runs(
            "source",
            [30.0, 34.0, 31.0, 33.0, 32.0],
            concentration=[1.2, 1.2, 1.2, 1.2, 1.2],
        )
        results = {row["predictor"]: row for row in predictor_correlations(runs)}
        assert "note" in results["placement_concentration"]

    def test_orders_predictors_by_strength(self):
        runs = _runs(
            "source",
            [30.0, 31.0, 32.0, 33.0, 34.0],
            concentration=[1.5, 1.4, 1.3, 1.2, 1.1],
            rms=[0.05, 0.07, 0.04, 0.08, 0.05],
        )
        results = predictor_correlations(runs)
        assert results[0]["predictor"] == "placement_concentration"

    def test_within_variant_separates_pooled_effects(self):
        """A pooled correlation can come entirely from between-variant
        differences; the within-variant version holds prototypes fixed."""
        runs = _runs(
            "source", [30.0, 31.0, 32.0, 33.0], concentration=[1.4, 1.3, 1.2, 1.1]
        ) + _runs("etf", [26.0, 27.0, 28.0, 29.0], concentration=[1.9, 1.8, 1.7, 1.6])
        within = within_variant_correlations(runs)
        variants = {row["variant"] for row in within}
        assert variants == {"source", "etf"}


class TestInterpretation:
    def test_flags_a_degenerate_comparison(self):
        """The regression that motivated these tests: with no gap to explain,
        the analysis must say so rather than attribute it to geometry."""
        experiment = _experiment(
            _runs("source", [30.0, 30.0, 30.0]),
            [_comparison("etf", 0.0), _comparison("rotated", 0.0)],
        )
        findings = interpret(experiment, [])
        assert any("no advantage here to attribute" in text for text in findings)
        assert not any("travels with the angular geometry" in text for text in findings)

    def test_attributes_to_geometry_when_rotation_is_cheap(self):
        """Rotation costs 0.1 pp while the frame costs 4.0: geometry carries it."""
        experiment = _experiment(
            _runs("source", [33.0, 34.0, 32.0]),
            [_comparison("etf", 4.0), _comparison("rotated", 0.1)],
        )
        findings = interpret(experiment, [])
        assert any("travels with the angular geometry" in text for text in findings)

    def test_attributes_to_directions_when_rotation_is_costly(self):
        """Rotation costs nearly as much as the frame: the directions matter."""
        experiment = _experiment(
            _runs("source", [33.0, 34.0, 32.0]),
            [_comparison("etf", 4.0), _comparison("rotated", 3.6)],
        )
        findings = interpret(experiment, [])
        assert any("Angular geometry alone does not account" in text for text in findings)

    def test_reports_an_intermediate_result(self):
        experiment = _experiment(
            _runs("source", [33.0, 34.0, 32.0]),
            [_comparison("etf", 4.0), _comparison("rotated", 2.0)],
        )
        findings = interpret(experiment, [])
        assert any("intermediate result" in text for text in findings)

    def test_reports_placement_beating_magnitude(self):
        correlations = [
            {"predictor": "placement_concentration", "pearson_r": -0.72},
            {"predictor": "placement_residual_rms", "pearson_r": -0.10},
        ]
        experiment = _experiment(_runs("source", [33.0, 34.0]), [])
        findings = interpret(experiment, correlations)
        assert any("placement-aware objective" in text for text in findings)

    def test_reports_magnitude_beating_placement(self):
        """The negative result must be stated as plainly as the positive one."""
        correlations = [
            {"predictor": "placement_concentration", "pearson_r": -0.05},
            {"predictor": "placement_residual_rms", "pearson_r": -0.68},
        ]
        experiment = _experiment(_runs("source", [33.0, 34.0]), [])
        findings = interpret(experiment, correlations)
        assert any("analogy" in text and "does not hold" in text for text in findings)

    def test_reports_an_undecided_contrast(self):
        correlations = [
            {"predictor": "placement_concentration", "pearson_r": -0.40},
            {"predictor": "placement_residual_rms", "pearson_r": -0.35},
        ]
        experiment = _experiment(_runs("source", [33.0, 34.0]), [])
        findings = interpret(experiment, correlations)
        assert any("do not separate the two accounts" in text for text in findings)

    def test_warns_when_a_null_result_is_underpowered(self):
        """A null result from an underpowered comparison must not read as
        evidence of no difference - the central lesson of the whole analysis."""
        experiment = _experiment(
            _runs("source", [33.0, 34.0]),
            [_comparison("mmd", 1.5, p_value=0.14, n=5)],
        )
        findings = interpret(experiment, [])
        assert any("too little to treat the null" in text for text in findings)

    def test_stays_silent_when_a_null_result_is_well_powered(self):
        """A 1.0 pp gap at 200 seeds has essentially full power, so a null there
        really is evidence of no difference and needs no caveat. Note that a
        0.1 pp gap would still be underpowered at 200 seeds - a small enough
        effect stays hard to detect however many runs are available."""
        experiment = _experiment(
            _runs("source", [33.0, 34.0]),
            [_comparison("mmd", 1.0, p_value=0.9, n=200)],
        )
        findings = interpret(experiment, [])
        assert not any("too little to treat the null" in text for text in findings)


class TestGeometryTable:
    def test_joins_geometry_to_outcome(self):
        experiment = _experiment(
            _runs("source", [33.0, 34.0]) + _runs("etf", [29.0, 30.0]),
            [],
            geometry={
                "source": {
                    "cosine_mean": 0.94, "cosine_range": 0.03,
                    "effective_rank": 4.7, "etf_deviation": 1.1,
                },
                "etf": {
                    "cosine_mean": -0.111, "cosine_range": 0.0,
                    "effective_rank": 9.0, "etf_deviation": 0.0,
                },
            },
        )
        table = geometry_outcome_table(experiment)
        assert [row["variant"] for row in table] == ["source", "etf"]
        assert table[0]["mean_accuracy"] > table[1]["mean_accuracy"]


class TestLoading:
    def test_missing_file_raises_with_guidance(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="run_variant_experiment"):
            load_experiment(tmp_path / "absent.json")


class TestUnitsAndMissingKeys:
    """Regressions found by code review.

    Accuracies are recorded in percentage points by run_variant_experiment, and
    the interpretation thresholds assume that. When the producer wrote fractions
    instead, a genuine 4-percentage-point effect - larger than the paper's
    headline result - was reported as "costs only 0.04 pp, so there is no
    advantage here to attribute to anything", the exact false null this module
    exists to prevent.
    """

    def test_a_real_effect_is_not_dismissed_as_no_advantage(self):
        experiment = _experiment(
            _runs("source", [33.0, 34.0, 32.0]),
            [_comparison("etf", 4.0), _comparison("rotated", 0.1)],
        )
        findings = interpret(experiment, [])
        assert not any("no advantage here to attribute" in text for text in findings)
        assert any("travels with the angular geometry" in text for text in findings)

    def test_power_warning_requires_a_test_to_have_been_run(self):
        """paired_comparison omits unpaired_p for a degenerate comparison.
        Defaulting the missing key to 1.0 read as "not significant" and produced
        a power warning about a test that never happened."""
        degenerate = {
            "variant_a": "source",
            "variant_b": "etf",
            "mean_paired_difference": 1.5,
            "pooled_sd": 1.5,
            "n_seeds": 5,
        }
        experiment = _experiment(_runs("source", [33.0, 34.0]), [degenerate])
        findings = interpret(experiment, [])
        assert not any("too little to treat the null" in text for text in findings)

    def test_power_warning_still_fires_when_a_test_did_run(self):
        experiment = _experiment(
            _runs("source", [33.0, 34.0]),
            [_comparison("etf", 1.5, p_value=0.14, n=5)],
        )
        findings = interpret(experiment, [])
        assert any("too little to treat the null" in text for text in findings)

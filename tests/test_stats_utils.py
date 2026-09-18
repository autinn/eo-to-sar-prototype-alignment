"""Tests for the statistical helpers.

The reference values come from the results table of arXiv 2609.07753 and are
recomputed here so that a regression in the statistics code shows up as a failing
test rather than a quietly different conclusion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from stats_utils import (  # noqa: E402
    achieved_power,
    bootstrap_ci,
    cohens_d,
    permutation_test,
    pooled_sd,
    population_std_from_sample_std,
    sample_std_from_population_std,
    seeds_for_power,
    welch_ttest,
)


# Published results: method -> (accuracy mean, std, seeds). Table 1 of arXiv 2609.07753.
REPORTED = {
    "frozen": (26.9, 0.2, 5),
    "sar_only": (29.8, 1.4, 5),
    "mmd": (31.8, 1.6, 5),
    "eo_proto": (33.3, 1.3, 5),
    "synth_proto": (29.3, 2.0, 5),
}


class TestWelchTTest:
    def test_reproduces_published_comparisons(self):
        """The four headline comparisons must reproduce to two decimal places."""
        expected = {
            ("eo_proto", "mmd"): 1.63,
            ("eo_proto", "synth_proto"): 3.75,
            ("eo_proto", "frozen"): 10.88,
            ("eo_proto", "sar_only"): 4.10,
        }
        for (a, b), expected_t in expected.items():
            result = welch_ttest(*REPORTED[a], *REPORTED[b])
            assert result["t_statistic"] == pytest.approx(expected_t, abs=0.005), (
                f"{a} vs {b}: got t={result['t_statistic']:.4f}, expected {expected_t}"
            )

    def test_eo_versus_mmd_is_not_significant(self):
        """The central claim of the analysis: their method does not beat their
        own label-free baseline at the conventional threshold."""
        result = welch_ttest(*REPORTED["eo_proto"], *REPORTED["mmd"])
        assert result["p_value"] > 0.05

    def test_eo_versus_synthetic_is_significant(self):
        """The paper's own control comparison does hold up."""
        result = welch_ttest(*REPORTED["eo_proto"], *REPORTED["synth_proto"])
        assert result["p_value"] < 0.05

    def test_matches_scipy_on_raw_samples(self):
        """Agreement with scipy's implementation on data where both apply."""
        generator = np.random.default_rng(0)
        a = generator.normal(10.0, 2.0, size=8)
        b = generator.normal(12.0, 3.0, size=11)

        reference = stats.ttest_ind(a, b, equal_var=False)
        result = welch_ttest(
            a.mean(), a.std(ddof=1), a.size,
            b.mean(), b.std(ddof=1), b.size,
        )

        assert result["t_statistic"] == pytest.approx(reference.statistic, rel=1e-10)
        assert result["p_value"] == pytest.approx(reference.pvalue, rel=1e-10)
        assert result["degrees_of_freedom"] == pytest.approx(reference.df, rel=1e-10)

    def test_symmetric_under_group_swap(self):
        forward = welch_ttest(33.3, 1.3, 5, 31.8, 1.6, 5)
        backward = welch_ttest(31.8, 1.6, 5, 33.3, 1.3, 5)
        assert forward["t_statistic"] == pytest.approx(-backward["t_statistic"])
        assert forward["p_value"] == pytest.approx(backward["p_value"])

    def test_rejects_degenerate_input(self):
        with pytest.raises(ValueError):
            welch_ttest(1.0, 0.0, 5, 2.0, 0.0, 5)
        with pytest.raises(ValueError):
            welch_ttest(1.0, 1.0, 1, 2.0, 1.0, 5)


class TestStandardDeviationConversion:
    def test_round_trip(self):
        for n in (2, 5, 10, 100):
            assert population_std_from_sample_std(
                sample_std_from_population_std(3.0, n), n
            ) == pytest.approx(3.0)

    def test_matches_numpy_conventions(self):
        """The conversion must agree with numpy's own ddof handling."""
        generator = np.random.default_rng(1)
        values = generator.normal(size=5)
        assert sample_std_from_population_std(
            float(values.std()), values.size
        ) == pytest.approx(float(values.std(ddof=1)))

    def test_inflation_factor_at_five_seeds(self):
        """At n=5 the sample std is about 11.8% larger than the population std.

        This is the size of the error made by feeding a summary.json std straight
        into a t-test, which is why the conversion exists.
        """
        assert sample_std_from_population_std(1.0, 5) == pytest.approx(
            np.sqrt(5 / 4), rel=1e-12
        )

    def test_rejects_single_observation(self):
        with pytest.raises(ValueError):
            sample_std_from_population_std(1.0, 1)


class TestPower:
    def test_seeds_needed_for_eo_versus_mmd(self):
        """Fifteen seeds per arm are needed to resolve the 1.5pp gap."""
        assert seeds_for_power(1.5, pooled_sd(1.3, 1.6)) == 15

    def test_seeds_needed_for_eo_versus_synthetic(self):
        """The comparison the paper could detect needed only three seeds."""
        assert seeds_for_power(4.0, pooled_sd(1.3, 2.0)) == 3

    def test_published_study_was_underpowered_for_the_key_comparison(self):
        """At five seeds the EO-vs-MMD comparison had well under 50% power,
        so failing to reach significance there is uninformative."""
        power = achieved_power(1.5, pooled_sd(1.3, 1.6), n_per_arm=5)
        assert power < 0.5

    def test_published_study_was_well_powered_for_the_control(self):
        power = achieved_power(4.0, pooled_sd(1.3, 2.0), n_per_arm=5)
        assert power > 0.9

    def test_power_increases_with_sample_size(self):
        powers = [achieved_power(1.5, 1.458, n) for n in (5, 10, 15, 30)]
        assert powers == sorted(powers)

    def test_larger_effects_need_fewer_seeds(self):
        sizes = [seeds_for_power(delta, 1.5) for delta in (1.0, 2.0, 4.0)]
        assert sizes == sorted(sizes, reverse=True)

    def test_seeds_for_power_is_consistent_with_achieved_power(self):
        """The sizing formula and the power formula must agree at the boundary."""
        needed = seeds_for_power(1.5, 1.458, power=0.80)
        assert achieved_power(1.5, 1.458, needed) >= 0.80
        assert achieved_power(1.5, 1.458, needed - 1) < 0.80

    def test_rejects_invalid_arguments(self):
        with pytest.raises(ValueError):
            seeds_for_power(0.0, 1.0)
        with pytest.raises(ValueError):
            seeds_for_power(1.0, 1.0, power=1.5)


class TestEffectSize:
    def test_sign_follows_difference(self):
        assert cohens_d(33.3, 1.3, 31.8, 1.6) > 0
        assert cohens_d(31.8, 1.6, 33.3, 1.3) < 0

    def test_known_value(self):
        """Equal std of 1.0 makes d equal to the raw difference."""
        assert cohens_d(2.0, 1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_eo_versus_mmd_effect_is_large_despite_non_significance(self):
        """A reminder that the null result is about power, not about a tiny effect:
        the point estimate is a large effect by conventional thresholds."""
        assert abs(cohens_d(33.3, 1.3, 31.8, 1.6)) > 0.8


class TestResampling:
    def test_bootstrap_brackets_the_mean(self):
        generator = np.random.default_rng(2)
        values = generator.normal(5.0, 1.0, size=200)
        lower, upper = bootstrap_ci(values, seed=0)
        assert lower < values.mean() < upper

    def test_bootstrap_interval_narrows_with_more_data(self):
        generator = np.random.default_rng(3)
        small = generator.normal(0.0, 1.0, size=20)
        large = generator.normal(0.0, 1.0, size=2000)
        small_lower, small_upper = bootstrap_ci(small, seed=0)
        large_lower, large_upper = bootstrap_ci(large, seed=0)
        assert (large_upper - large_lower) < (small_upper - small_lower)

    def test_bootstrap_is_deterministic_given_a_seed(self):
        values = np.arange(50, dtype=float)
        assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)

    def test_permutation_detects_a_clear_difference(self):
        generator = np.random.default_rng(4)
        a = generator.normal(0.0, 1.0, size=30)
        b = generator.normal(5.0, 1.0, size=30)
        assert permutation_test(a, b, n_perm=2000, seed=0) < 0.01

    def test_permutation_reports_no_difference_for_identical_distributions(self):
        generator = np.random.default_rng(5)
        a = generator.normal(0.0, 1.0, size=40)
        b = generator.normal(0.0, 1.0, size=40)
        assert permutation_test(a, b, n_perm=2000, seed=0) > 0.05

    def test_permutation_p_value_is_strictly_positive(self):
        """The add-one correction avoids reporting p = 0, which would overstate
        what a finite number of permutations can establish."""
        a = np.zeros(10)
        b = np.ones(10) * 100.0
        assert permutation_test(a, b, n_perm=100, seed=0) > 0.0

    def test_rejects_empty_samples(self):
        with pytest.raises(ValueError):
            permutation_test(np.array([]), np.array([1.0]))
        with pytest.raises(ValueError):
            bootstrap_ci(np.array([]))

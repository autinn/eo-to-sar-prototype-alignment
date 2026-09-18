"""Statistical tests, effect sizes and power analysis for method comparisons.

The published results report only mean and standard deviation over five seeds, so
most helpers here take summary statistics rather than raw samples. Where raw
per-seed values are available the resampling helpers give exact answers instead.

One subtlety runs through this module: ``summarize_runs`` in ``utils.py`` reports
numpy's default standard deviation, which is the *population* std (``ddof=0``).
Every classical test below expects the *sample* std (``ddof=1``). At five seeds
the two differ by a factor of ``sqrt(5/4) = 1.118``, so mixing them understates
variance by about 12% and overstates significance. Use
``sample_std_from_population_std`` when feeding numbers taken from a
``summary.json`` into these functions.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats


# Convert between the two standard-deviation conventions used in this project.
def sample_std_from_population_std(pop_std: float, n: int) -> float:
    """Convert a population std (ddof=0) to a sample std (ddof=1)."""
    if n < 2:
        raise ValueError("At least two observations are required.")
    return float(pop_std) * float(np.sqrt(n / (n - 1)))


def population_std_from_sample_std(sample_std: float, n: int) -> float:
    """Convert a sample std (ddof=1) back to a population std (ddof=0)."""
    if n < 2:
        raise ValueError("At least two observations are required.")
    return float(sample_std) * float(np.sqrt((n - 1) / n))


# Welch's test does not assume the two groups share a variance.
def welch_ttest(
    mean_a: float,
    std_a: float,
    n_a: int,
    mean_b: float,
    std_b: float,
    n_b: int,
) -> dict[str, float]:
    """Welch's unequal-variance t-test from summary statistics.

    ``std_a`` and ``std_b`` must be sample standard deviations (ddof=1).
    Returns the difference, standard error, t statistic, Welch-Satterthwaite
    degrees of freedom and the two-sided p-value.
    """
    if n_a < 2 or n_b < 2:
        raise ValueError("Each group needs at least two observations.")

    var_a = float(std_a) ** 2 / n_a
    var_b = float(std_b) ** 2 / n_b
    standard_error = float(np.sqrt(var_a + var_b))

    if standard_error == 0.0:
        raise ValueError("Both groups have zero variance; the t statistic is undefined.")

    difference = float(mean_a) - float(mean_b)
    t_statistic = difference / standard_error

    # Welch-Satterthwaite effective degrees of freedom.
    numerator = (var_a + var_b) ** 2
    denominator = var_a**2 / (n_a - 1) + var_b**2 / (n_b - 1)
    degrees_of_freedom = numerator / denominator

    p_value = 2.0 * stats.t.sf(abs(t_statistic), degrees_of_freedom)

    return {
        "difference": difference,
        "standard_error": standard_error,
        "t_statistic": t_statistic,
        "degrees_of_freedom": float(degrees_of_freedom),
        "p_value": float(p_value),
    }


# Effect size complements the p-value, which is sensitive to the number of seeds.
def cohens_d(mean_a: float, std_a: float, mean_b: float, std_b: float) -> float:
    """Cohen's d using the root-mean-square of the two standard deviations."""
    pooled = float(np.sqrt((float(std_a) ** 2 + float(std_b) ** 2) / 2.0))
    if pooled == 0.0:
        raise ValueError("Both groups have zero variance; Cohen's d is undefined.")
    return (float(mean_a) - float(mean_b)) / pooled


def pooled_sd(std_a: float, std_b: float) -> float:
    """Root-mean-square of two standard deviations, as used by ``cohens_d``."""
    return float(np.sqrt((float(std_a) ** 2 + float(std_b) ** 2) / 2.0))


# How many seeds a future run needs to resolve a gap of a given size.
def seeds_for_power(
    delta: float,
    pooled_sd: float,
    power: float = 0.80,
    alpha: float = 0.05,
) -> int:
    """Seeds per arm needed to detect ``delta`` at the given power.

    Uses the normal approximation ``n = 2 (z_{1-alpha/2} + z_{power})^2 sd^2 / delta^2``,
    which is the standard two-sample sizing formula. The result is rounded up.
    """
    if delta == 0.0:
        raise ValueError("Cannot size a study to detect a zero difference.")
    if not 0.0 < power < 1.0:
        raise ValueError("power must lie strictly between 0 and 1.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")

    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_power = stats.norm.ppf(power)
    required = 2.0 * (z_alpha + z_power) ** 2 * float(pooled_sd) ** 2 / float(delta) ** 2
    return int(np.ceil(required))


def achieved_power(
    delta: float,
    pooled_sd: float,
    n_per_arm: int,
    alpha: float = 0.05,
) -> float:
    """Power of a two-sample test with ``n_per_arm`` observations in each arm."""
    if n_per_arm < 2:
        raise ValueError("At least two observations per arm are required.")

    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    noncentrality = abs(float(delta)) / (float(pooled_sd) * np.sqrt(2.0 / n_per_arm))
    return float(stats.norm.cdf(noncentrality - z_alpha))


# Resampling alternatives for when the raw per-seed numbers are available.
def bootstrap_ci(
    values: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a statistic."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty sample.")

    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(n_boot, values.size))
    resampled = np.array([statistic(values[row]) for row in indices])

    lower = float(np.percentile(resampled, 100.0 * alpha / 2.0))
    upper = float(np.percentile(resampled, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 10_000,
    seed: int = 0,
) -> float:
    """Two-sided permutation test on the difference in means.

    Makes no distributional assumption, which matters at five seeds where the
    normality assumption behind the t-test is untestable.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("Both samples must be non-empty.")

    observed = abs(float(a.mean()) - float(b.mean()))
    combined = np.concatenate([a, b])
    n_a = a.size

    generator = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        generator.shuffle(combined)
        difference = abs(float(combined[:n_a].mean()) - float(combined[n_a:].mean()))
        if difference >= observed:
            count += 1

    # Add-one correction keeps the p-value strictly positive.
    return (count + 1) / (n_perm + 1)

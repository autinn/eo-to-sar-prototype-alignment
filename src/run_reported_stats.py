"""Statistical analysis of the published results.

Reads ``analysis/paper_reported_results.yaml`` and computes every pairwise
comparison, effect size and power figure, writing the result to
``analysis/reported_stats.json`` and ``analysis/reported_stats.csv``.

The question this answers is not "are the reported numbers correct" - they are
taken as given - but "what do they support". Two things emerge that the paper
does not report:

1. The proposed method is not significantly better than the paper's own MMD
   baseline, which uses no EO labels at all. The comparison that would establish
   transfer of class-specific structure is the one that fails to reach
   significance.

2. That comparison was badly underpowered. At five seeds it had roughly a one in
   three chance of detecting an effect of the observed size, so its null result
   carries very little information either way. Fifteen seeds per arm would be
   needed for a conventional 80% chance.

Run with::

    python src/run_reported_stats.py
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import yaml

from stats_utils import (
    achieved_power,
    cohens_d,
    pooled_sd,
    sample_std_from_population_std,
    seeds_for_power,
    welch_ttest,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "analysis" / "paper_reported_results.yaml"
OUTPUT_DIR = REPO_ROOT / "analysis"

SIGNIFICANCE_LEVEL = 0.05
TARGET_POWER = 0.80


def load_reported_results(path: Path = RESULTS_PATH) -> dict:
    """Load the transcribed published results."""
    if not path.is_file():
        raise FileNotFoundError(f"Reported results not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def compare_pair(
    methods: dict,
    name_a: str,
    name_b: str,
    treat_std_as_population: bool,
) -> dict[str, float | str | bool]:
    """Full comparison of two methods under one standard-deviation convention."""
    a, b = methods[name_a], methods[name_b]

    std_a = float(a["accuracy_std"])
    std_b = float(b["accuracy_std"])
    n_a = int(a["seeds"])
    n_b = int(b["seeds"])

    # The published std may be a population std; convert if so.
    if treat_std_as_population:
        std_a = sample_std_from_population_std(std_a, n_a)
        std_b = sample_std_from_population_std(std_b, n_b)

    test = welch_ttest(
        float(a["accuracy_mean"]), std_a, n_a,
        float(b["accuracy_mean"]), std_b, n_b,
    )
    spread = pooled_sd(std_a, std_b)
    difference = test["difference"]

    return {
        "method_a": name_a,
        "method_b": name_b,
        "mean_a": float(a["accuracy_mean"]),
        "mean_b": float(b["accuracy_mean"]),
        "difference": difference,
        "t_statistic": test["t_statistic"],
        "degrees_of_freedom": test["degrees_of_freedom"],
        "p_value": test["p_value"],
        "significant": bool(test["p_value"] < SIGNIFICANCE_LEVEL),
        "cohens_d": cohens_d(
            float(a["accuracy_mean"]), std_a, float(b["accuracy_mean"]), std_b
        ),
        "pooled_sd": spread,
        "achieved_power": achieved_power(difference, spread, n_per_arm=n_a),
        "seeds_for_80_percent_power": seeds_for_power(
            difference, spread, power=TARGET_POWER
        ),
    }


def analyse(reported: dict, treat_std_as_population: bool) -> list[dict]:
    """Every pairwise comparison, ordered by descending absolute difference."""
    methods = reported["methods"]
    comparisons = [
        compare_pair(methods, a, b, treat_std_as_population)
        for a, b in itertools.combinations(methods, 2)
    ]
    return sorted(comparisons, key=lambda row: abs(row["difference"]), reverse=True)


def format_table(comparisons: list[dict]) -> str:
    """Render the comparisons as a fixed-width table."""
    header = (
        f"{'comparison':<42}{'diff':>7}{'t':>8}{'p':>9}"
        f"{'d':>7}{'power':>8}{'n@80%':>7}  sig"
    )
    lines = [header, "-" * len(header)]
    for row in comparisons:
        label = f"{row['method_a']} vs {row['method_b']}"
        lines.append(
            f"{label:<42}{row['difference']:+7.2f}{row['t_statistic']:8.2f}"
            f"{row['p_value']:9.4f}{row['cohens_d']:7.2f}"
            f"{row['achieved_power']:8.2f}{row['seeds_for_80_percent_power']:7d}"
            f"  {'yes' if row['significant'] else 'no'}"
        )
    return "\n".join(lines)


def key_findings(comparisons: list[dict], reported: dict) -> list[str]:
    """The conclusions the numbers support, stated plainly."""
    by_pair = {(row["method_a"], row["method_b"]): row for row in comparisons}

    def find(a: str, b: str) -> dict:
        if (a, b) in by_pair:
            return by_pair[(a, b)]
        return by_pair[(b, a)]

    findings: list[str] = []

    versus_mmd = find("eo_prototype", "mmd")
    if not versus_mmd["significant"]:
        findings.append(
            f"EO prototype alignment is not significantly better than the paper's "
            f"own MMD baseline (difference {abs(versus_mmd['difference']):.1f} pp, "
            f"p = {versus_mmd['p_value']:.3f}). MMD discards EO labels entirely, so "
            f"the comparison that would demonstrate transfer of class-specific "
            f"optical structure is the one that does not reach significance."
        )
        findings.append(
            f"That comparison was underpowered: at {reported['methods']['mmd']['seeds']} "
            f"seeds it had {versus_mmd['achieved_power']:.0%} power against an effect "
            f"of the observed size, so the null result is weak evidence rather than "
            f"evidence of no effect. Detecting it reliably would need "
            f"{versus_mmd['seeds_for_80_percent_power']} seeds per arm."
        )

    versus_synthetic = find("eo_prototype", "synthetic_prototype")
    if versus_synthetic["significant"]:
        findings.append(
            f"The paper's own control comparison does hold: EO prototypes beat the "
            f"synthetic equiangular frame by {abs(versus_synthetic['difference']):.1f} pp "
            f"(p = {versus_synthetic['p_value']:.3f}), with "
            f"{versus_synthetic['achieved_power']:.0%} power. Note however that the two "
            f"conditions also differ in learning rate and seed set; see "
            f"analysis/CONFOUNDS.md."
        )

    synthetic_versus_baseline = find("synthetic_prototype", "sar_only")
    if not synthetic_versus_baseline["significant"]:
        findings.append(
            f"Prototype geometry alone does not help: the synthetic frame is "
            f"indistinguishable from plain SAR-only fine-tuning "
            f"(p = {synthetic_versus_baseline['p_value']:.3f}), so the gain is not "
            f"explained by class separation in the loss by itself."
        )

    return findings


def main() -> None:
    reported = load_reported_results()

    # Report both readings of the published standard deviation, since the paper
    # does not say which convention it used and the repository's own summary code
    # produces population stds.
    as_sample = analyse(reported, treat_std_as_population=False)
    as_population = analyse(reported, treat_std_as_population=True)

    print("=" * 88)
    print("Published results analysed as SAMPLE standard deviations (ddof=1)")
    print("=" * 88)
    print(format_table(as_sample))
    print()
    print("=" * 88)
    print("Published results analysed as POPULATION standard deviations (ddof=0),")
    print("converted to sample standard deviations before testing")
    print("=" * 88)
    print(format_table(as_population))
    print()

    # A conclusion that flips between conventions would not be trustworthy.
    flipped = [
        f"{a['method_a']} vs {a['method_b']}"
        for a, b in zip(as_sample, as_population)
        if a["significant"] != b["significant"]
    ]
    if flipped:
        print(f"WARNING: significance depends on the std convention for: {flipped}")
    else:
        print("Every conclusion is unchanged under either convention.")
    print()

    print("=" * 88)
    print("Key findings")
    print("=" * 88)
    findings = key_findings(as_sample, reported)
    for index, finding in enumerate(findings, start=1):
        print(f"{index}. {finding}")
        print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "source": reported["source"],
        "significance_level": SIGNIFICANCE_LEVEL,
        "target_power": TARGET_POWER,
        "comparisons_sample_std": as_sample,
        "comparisons_population_std": as_population,
        "conclusion_stable_across_conventions": not flipped,
        "key_findings": findings,
    }
    json_path = OUTPUT_DIR / "reported_stats.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    csv_path = OUTPUT_DIR / "reported_stats.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(as_sample[0]))
        writer.writeheader()
        writer.writerows(as_sample)

    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()

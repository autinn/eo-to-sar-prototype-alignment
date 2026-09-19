"""Analyse the output of the variant experiment.

``run_variant_experiment.py`` produces one row per variant per seed, recording
accuracy alongside the geometry of the prototypes used and the placement of the
resulting alignment residual. This module turns those rows into the three answers
the project is after.

**Q1. Does the advantage over label-free alignment survive adequate power?**
Paired per-seed comparisons of every variant against the real optical
prototypes, reported with the achieved power so a null result can be
distinguished from an underpowered one.

**Q2. Is it geometry or the specific optical directions?** ``rotated`` preserves
every pairwise angle while destroying the correspondence to particular optical
directions. If it matches ``source`` the gain is geometric; if it falls back
toward the equiangular frame, the directions carry something.

**Q3. Does placement predict outcome better than magnitude?** This is the
question ported from cross-model KV cache transfer (arXiv:2608.03893), where
reconstruction quality did not predict downstream performance (r = -0.20) while
error placement did (r = +0.57). Here the two candidate predictors are residual
concentration against the classifier's read subspace, and residual RMS - the
quantity the training objective actually minimises. If concentration correlates
with accuracy and RMS does not, the alignment loss is optimising the wrong thing.

Run with::

    python src/analyse_experiment.py analysis/variant_experiment_dinov3.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from stats_utils import achieved_power, bootstrap_ci, pooled_sd


REPO_ROOT = Path(__file__).resolve().parents[1]

# Candidate predictors of accuracy. The contrast between the first two is the
# point: one is what the loss minimises, the other is where the error lands.
PREDICTORS = {
    "placement_residual_rms": "residual magnitude (what the loss minimises)",
    "placement_concentration": "residual placement (what the head reads)",
    "placement_nullspace_fraction": "fraction invisible to the classifier",
    "placement_logit_shift_rms": "induced logit displacement",
}


def load_experiment(path: Path) -> dict:
    """Load a variant-experiment result file."""
    if not path.is_file():
        raise FileNotFoundError(
            f"No experiment output at {path}. Run src/run_variant_experiment.py first."
        )
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def variant_summary(runs: list[dict], metric: str = "accuracy") -> dict[str, dict]:
    """Mean, sample std and bootstrap interval for each variant."""
    summary: dict[str, dict] = {}
    for variant in sorted({row["variant"] for row in runs}):
        values = np.array(
            [row[metric] for row in runs if row["variant"] == variant], dtype=float
        )
        entry = {
            "n_seeds": int(values.size),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        }
        if values.size > 2:
            lower, upper = bootstrap_ci(values, seed=0)
            entry["ci_low"], entry["ci_high"] = lower, upper
        summary[variant] = entry
    return summary


def predictor_correlations(
    runs: list[dict],
    metric: str = "accuracy",
) -> list[dict]:
    """Correlate each candidate predictor against the outcome.

    Reports Pearson and Spearman. Spearman matters because the relationship
    between placement and accuracy need not be linear - the claim is only that
    one ordering tracks the other.
    """
    outcomes = np.array([row[metric] for row in runs], dtype=float)
    results: list[dict] = []

    for predictor, description in PREDICTORS.items():
        if not all(predictor in row for row in runs):
            continue
        values = np.array([row[predictor] for row in runs], dtype=float)

        # A predictor that never varies cannot correlate with anything.
        if float(values.std()) < 1e-12 or float(outcomes.std()) < 1e-12:
            results.append(
                {
                    "predictor": predictor,
                    "description": description,
                    "note": "no variance; correlation undefined",
                }
            )
            continue

        pearson = stats.pearsonr(values, outcomes)
        spearman = stats.spearmanr(values, outcomes)
        results.append(
            {
                "predictor": predictor,
                "description": description,
                "n": int(values.size),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
            }
        )

    return sorted(
        results,
        key=lambda row: abs(row.get("pearson_r", 0.0)),
        reverse=True,
    )


def within_variant_correlations(runs: list[dict], metric: str = "accuracy") -> list[dict]:
    """Correlate predictors against outcome *within* each variant.

    The pooled correlation can be driven entirely by between-variant differences:
    if one variant is both better and differently placed, the association appears
    without placement explaining anything about individual runs. Repeating the
    correlation inside each variant, where the prototypes are identical and only
    the seed differs, separates the two.
    """
    results: list[dict] = []
    for variant in sorted({row["variant"] for row in runs}):
        subset = [row for row in runs if row["variant"] == variant]
        if len(subset) < 4:
            continue

        outcomes = np.array([row[metric] for row in subset], dtype=float)
        for predictor in PREDICTORS:
            if not all(predictor in row for row in subset):
                continue
            values = np.array([row[predictor] for row in subset], dtype=float)
            if float(values.std()) < 1e-12 or float(outcomes.std()) < 1e-12:
                continue
            correlation = stats.pearsonr(values, outcomes)
            results.append(
                {
                    "variant": variant,
                    "predictor": predictor,
                    "n": len(subset),
                    "pearson_r": float(correlation.statistic),
                    "pearson_p": float(correlation.pvalue),
                }
            )
    return results


def geometry_outcome_table(
    experiment: dict,
    metric: str = "accuracy",
) -> list[dict]:
    """Join each variant's prototype geometry to its mean outcome."""
    summary = variant_summary(experiment["runs"], metric)
    table: list[dict] = []
    for variant, geometry in experiment.get("geometry", {}).items():
        if variant not in summary:
            continue
        table.append(
            {
                "variant": variant,
                "mean_accuracy": summary[variant]["mean"],
                "std": summary[variant]["std"],
                "cosine_mean": geometry["cosine_mean"],
                "cosine_range": geometry["cosine_range"],
                "effective_rank": geometry["effective_rank"],
                "etf_deviation": geometry["etf_deviation"],
            }
        )
    return sorted(table, key=lambda row: row["mean_accuracy"], reverse=True)


def interpret(experiment: dict, correlations: list[dict]) -> list[str]:
    """State what the numbers support, in the terms the project set out."""
    findings: list[str] = []
    comparisons = {
        (row["variant_a"], row["variant_b"]): row
        for row in experiment.get("comparisons", [])
    }

    def comparison(variant: str) -> dict | None:
        return comparisons.get(("source", variant))

    # Q2: geometry against specific directions.
    rotated = comparison("rotated")
    etf = comparison("etf")
    if rotated and etf:
        rotated_gap = abs(rotated["mean_paired_difference"])
        etf_gap = abs(etf["mean_paired_difference"])

        # The comparison is only meaningful if replacing the prototypes costs
        # something. With no reference gap every ratio is degenerate, and an
        # earlier version of this branch reported "the advantage travels with the
        # angular geometry" from two zeros - a confident conclusion drawn from no
        # evidence, which is precisely the failure this project exists to catch.
        if etf_gap < 0.05:
            findings.append(
                f"Replacing the optical prototypes with an equiangular frame "
                f"costs only {etf_gap:.2f} pp, so there is no advantage here to "
                f"attribute to anything. Either alignment is not helping in this "
                f"configuration, or the runs have not separated."
            )
        elif rotated_gap < 0.25 * etf_gap:
            findings.append(
                f"Rotating the optical prototypes costs little "
                f"({rotated_gap:.2f} pp) while replacing them with an equiangular "
                f"frame costs {etf_gap:.2f} pp. The advantage therefore travels "
                f"with the angular geometry rather than with the particular "
                f"optical directions - rotation preserves every pairwise angle "
                f"and destroys the correspondence to specific directions."
            )
        elif rotated_gap > 0.75 * etf_gap:
            findings.append(
                f"Rotation costs {rotated_gap:.2f} pp against {etf_gap:.2f} pp for "
                f"the equiangular frame, so most of the advantage is lost as soon "
                f"as the prototypes stop pointing in their original directions. "
                f"Angular geometry alone does not account for the gain."
            )
        else:
            findings.append(
                f"Rotation costs {rotated_gap:.2f} pp and the equiangular frame "
                f"{etf_gap:.2f} pp, an intermediate result: geometry carries part "
                f"of the advantage and the specific directions carry part."
            )

    # Q3: placement against magnitude.
    by_predictor = {row["predictor"]: row for row in correlations}
    concentration = by_predictor.get("placement_concentration", {})
    magnitude = by_predictor.get("placement_residual_rms", {})
    if "pearson_r" in concentration and "pearson_r" in magnitude:
        concentration_r = concentration["pearson_r"]
        magnitude_r = magnitude["pearson_r"]
        if abs(concentration_r) > abs(magnitude_r) + 0.2:
            findings.append(
                f"Residual placement predicts accuracy (r = {concentration_r:+.2f}) "
                f"better than residual magnitude (r = {magnitude_r:+.2f}), the "
                f"quantity the alignment loss minimises. This matches the pattern "
                f"found for cross-model KV cache transfer and suggests a "
                f"placement-aware objective would be better targeted than the "
                f"cosine distance currently used."
            )
        elif abs(magnitude_r) > abs(concentration_r) + 0.2:
            findings.append(
                f"Residual magnitude predicts accuracy (r = {magnitude_r:+.2f}) "
                f"better than placement (r = {concentration_r:+.2f}). The analogy "
                f"to cross-model KV transfer does not hold here, and the existing "
                f"cosine objective is targeting the right quantity."
            )
        else:
            findings.append(
                f"Placement (r = {concentration_r:+.2f}) and magnitude "
                f"(r = {magnitude_r:+.2f}) predict accuracy about equally well, so "
                f"these data do not separate the two accounts."
            )

    # Power for every null result, so silence is not mistaken for absence.
    for (_, variant), row in comparisons.items():
        # The key must be present, not defaulted. paired_comparison omits
        # unpaired_p entirely for a degenerate comparison, and defaulting the
        # absent key to 1.0 read as "not significant" - producing a power
        # warning about a test that was never run.
        if "unpaired_p" in row and row["unpaired_p"] > 0.05 and "pooled_sd" in row:
            power = achieved_power(
                row["mean_paired_difference"], row["pooled_sd"], row["n_seeds"]
            )
            if power < 0.8:
                findings.append(
                    f"The comparison against {variant} is not significant, but it "
                    f"had {power:.0%} power at {row['n_seeds']} seeds - too little "
                    f"to treat the null as evidence of no difference."
                )

    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        type=Path,
        nargs="?",
        default=REPO_ROOT / "analysis" / "variant_experiment_dinov3.json",
    )
    parser.add_argument("--metric", default="accuracy")
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()

    experiment = load_experiment(arguments.experiment)
    runs = experiment["runs"]
    backbone = experiment.get("backbone", "unknown")

    if backbone == "stub":
        print("WARNING: this experiment used the stub backbone.")
        print("The numbers below exercise the analysis; they are not findings.\n")

    summary = variant_summary(runs, arguments.metric)
    correlations = predictor_correlations(runs, arguments.metric)
    within = within_variant_correlations(runs, arguments.metric)
    geometry = geometry_outcome_table(experiment, arguments.metric)

    print("=" * 76)
    print(f"Variant outcomes ({arguments.metric})")
    print("=" * 76)
    print(f"  {'variant':<22}{'n':>4}{'mean':>10}{'std':>9}{'95% CI':>22}")
    for variant, entry in sorted(
        summary.items(), key=lambda item: item[1]["mean"], reverse=True
    ):
        interval = (
            f"[{entry['ci_low']:.4f}, {entry['ci_high']:.4f}]"
            if "ci_low" in entry
            else "-"
        )
        print(
            f"  {variant:<22}{entry['n_seeds']:>4}{entry['mean']:>10.4f}"
            f"{entry['std']:>9.4f}{interval:>22}"
        )

    print()
    print("=" * 76)
    print("What predicts the outcome")
    print("=" * 76)
    print(f"  {'predictor':<32}{'pearson r':>11}{'p':>9}{'spearman':>11}")
    for row in correlations:
        if "pearson_r" not in row:
            print(f"  {row['predictor']:<32}{'(' + row['note'] + ')':>31}")
            continue
        print(
            f"  {row['predictor']:<32}{row['pearson_r']:>+11.3f}"
            f"{row['pearson_p']:>9.4f}{row['spearman_rho']:>+11.3f}"
        )
    if within:
        print()
        print("  Within-variant correlations (seed-level, prototypes held fixed):")
        for row in within:
            print(
                f"    {row['variant']:<20}{row['predictor']:<32}"
                f"r = {row['pearson_r']:+.3f}  p = {row['pearson_p']:.3f}"
            )

    print()
    print("=" * 76)
    print("Prototype geometry against outcome")
    print("=" * 76)
    print(f"  {'variant':<22}{'accuracy':>10}{'cos_mean':>10}{'cos_range':>11}{'eff_rank':>10}")
    for row in geometry:
        print(
            f"  {row['variant']:<22}{row['mean_accuracy']:>10.4f}"
            f"{row['cosine_mean']:>10.4f}{row['cosine_range']:>11.4f}"
            f"{row['effective_rank']:>10.2f}"
        )

    findings = interpret(experiment, correlations)
    if findings:
        print()
        print("=" * 76)
        print("Interpretation")
        print("=" * 76)
        for index, finding in enumerate(findings, start=1):
            print(f"{index}. {finding}\n")

    output_path = arguments.out or arguments.experiment.with_name(
        arguments.experiment.stem + "_analysis.json"
    )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": str(arguments.experiment),
                "backbone": backbone,
                "metric": arguments.metric,
                "variant_summary": summary,
                "predictor_correlations": correlations,
                "within_variant_correlations": within,
                "geometry_outcome": geometry,
                "findings": findings,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

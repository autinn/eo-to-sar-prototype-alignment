"""Reproduce the published results, then check them against the paper.

Deliverable D4. Runs the five released training scripts unmodified, at their own
settings and their own seeds, and reports whether each reproduced accuracy falls
within seed variance of the published figure.

This runs *before* the corrected experiment for a reason. The corrected
experiment changes the protocol, so a discrepancy in its results could come
either from the protocol change or from something in the environment - a library
version, a hardware difference, a data revision. Reproducing first separates
those: if the published numbers come back, the environment is sound and any later
difference is attributable to the change of protocol.

The scripts are invoked as subprocesses rather than imported, because each one
executes its work at module level and hardcodes its own hyperparameters. That is
deliberate - they must run exactly as the authors released them, with no
modification and no monkey-patching.

Note that the published scripts select their best epoch on the test set, so the
figures reproduced here carry that optimism too. This script does not correct
that; correcting it is what ``run_variant_experiment.py`` is for.

Run with::

    python src/run_reproduction.py                 # all five methods
    python src/run_reproduction.py --methods frozen_dino sar_only
    python src/run_reproduction.py --dry-run       # show what would run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

from stats_utils import sample_std_from_population_std, welch_ttest


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTED_PATH = REPO_ROOT / "analysis" / "paper_reported_results.yaml"

# Script name -> key in paper_reported_results.yaml.
METHODS = {
    "frozen_dino": "frozen",
    "sar_only": "sar_only",
    "mmd_alignment": "mmd",
    "eo_prototype_alignment": "eo_prototype",
    "synthetic_prototype_alignment": "synthetic_prototype",
}


def published_figures() -> dict:
    """The transcribed published results."""
    with REPORTED_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)["methods"]


def run_method(method: str, timeout_s: int | None = None) -> dict:
    """Run one published training script as a subprocess."""
    script = REPO_ROOT / "src" / f"train_{method}.py"
    if not script.is_file():
        raise FileNotFoundError(f"Training script not found: {script}")

    started = time.time()
    process = subprocess.run(
        [sys.executable, str(script)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    elapsed = time.time() - started

    if process.returncode != 0:
        return {
            "method": method,
            "status": "failed",
            "returncode": process.returncode,
            "stderr_tail": process.stderr[-2000:],
            "elapsed_s": elapsed,
        }

    # Each script writes summary.json under outputs/<method>/.
    summary_path = REPO_ROOT / "outputs" / method / "summary.json"
    if not summary_path.is_file():
        return {
            "method": method,
            "status": "no_summary",
            "expected": str(summary_path),
            "elapsed_s": elapsed,
        }

    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)

    return {
        "method": method,
        "status": "ok",
        "elapsed_s": elapsed,
        "summary": summary,
    }


def compare_to_published(
    method_key: str,
    observed_mean: float,
    observed_std: float,
    observed_n: int,
    published: dict,
) -> dict:
    """Test a reproduced figure against the published one.

    The published standard deviation may be a population std, since
    ``summarize_runs`` in ``utils.py`` reports numpy's default. Both readings are
    computed; a reproduction judgement that flipped between them would not be
    trustworthy.
    """
    entry = published[method_key]
    published_mean = float(entry["accuracy_mean"])
    published_std = float(entry["accuracy_std"])
    published_n = int(entry["seeds"])

    result = {
        "published_mean": published_mean,
        "published_std": published_std,
        "observed_mean": observed_mean,
        "observed_std": observed_std,
        "difference": observed_mean - published_mean,
    }

    for label, std in (
        ("as_sample_std", published_std),
        ("as_population_std", sample_std_from_population_std(published_std, published_n)),
    ):
        if observed_std <= 0 and std <= 0:
            continue
        test = welch_ttest(
            observed_mean, max(observed_std, 1e-9), observed_n,
            published_mean, max(std, 1e-9), published_n,
        )
        result[label] = {
            "t": test["t_statistic"],
            "p": test["p_value"],
            # A reproduction succeeds when the difference is NOT significant.
            "reproduced": bool(test["p_value"] > 0.05),
        }

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="*", default=list(METHODS))
    parser.add_argument("--timeout", type=int, default=None, help="Per-method seconds.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "analysis" / "reproduction.json"
    )
    arguments = parser.parse_args()

    unknown = set(arguments.methods) - set(METHODS)
    if unknown:
        parser.error(f"Unknown methods: {sorted(unknown)}. Choose from {list(METHODS)}.")

    config_path = REPO_ROOT / "config.yaml"
    if not config_path.is_file() and not arguments.dry_run:
        parser.error(
            "config.yaml not found. The published scripts need real paths to "
            "DINOv3 and UNICORNv2; copy config-example.yaml and fill it in."
        )

    if arguments.dry_run:
        print("Would run, in order:")
        for method in arguments.methods:
            print(f"  {sys.executable} src/train_{method}.py")
        print("\nEach writes outputs/<method>/summary.json, which is then compared")
        print("against analysis/paper_reported_results.yaml.")
        return

    published = published_figures()
    results = []

    for method in arguments.methods:
        print(f"\n{'=' * 70}\nRunning {method}\n{'=' * 70}")
        outcome = run_method(method, arguments.timeout)

        if outcome["status"] != "ok":
            print(f"  FAILED ({outcome['status']})")
            if "stderr_tail" in outcome:
                print(outcome["stderr_tail"][-600:])
            results.append(outcome)
            continue

        metrics = outcome["summary"]["metrics"]["accuracy"]
        # summarize_runs reports a population std; the comparison needs a sample std.
        n_seeds = len(outcome["summary"]["seeds"])
        observed_mean = float(metrics["mean"]) * 100.0
        observed_std = sample_std_from_population_std(
            float(metrics["std"]) * 100.0, n_seeds
        )

        comparison = compare_to_published(
            METHODS[method], observed_mean, observed_std, n_seeds, published
        )
        outcome["comparison"] = comparison
        results.append(outcome)

        verdict = comparison.get("as_sample_std", {}).get("reproduced")
        print(
            f"  published {comparison['published_mean']:.1f}  "
            f"observed {observed_mean:.1f}  "
            f"difference {comparison['difference']:+.1f} pp  "
            f"-> {'reproduced' if verdict else 'DIFFERS'}"
        )

    # Persist before printing. Every run above is a full training sweep, so a
    # formatting failure in the summary table must not discard the results.
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    with arguments.out.open("w", encoding="utf-8") as handle:
        json.dump({"results": results}, handle, indent=2)
        handle.write("\n")
    print(f"\nWrote {arguments.out}")

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    print(f"  {'method':<32}{'published':>11}{'observed':>11}{'verdict':>14}")
    for outcome in results:
        if outcome["status"] != "ok":
            print(f"  {outcome['method']:<32}{'-':>11}{'-':>11}{outcome['status']:>14}")
            continue
        comparison = outcome["comparison"]
        # compare_to_published skips a convention entirely when both standard
        # deviations are zero, so this key may be absent. A bare lookup here
        # would raise after all five training runs have completed, discarding
        # the whole sweep.
        sample = comparison.get("as_sample_std")
        verdict = (
            ("reproduced" if sample["reproduced"] else "DIFFERS")
            if sample
            else "no variance"
        )
        print(
            f"  {outcome['method']:<32}{comparison['published_mean']:>11.1f}"
            f"{comparison['observed_mean']:>11.1f}{verdict:>14}"
        )

    flipped = [
        outcome["method"]
        for outcome in results
        if outcome["status"] == "ok"
        and "as_population_std" in outcome["comparison"]
        and outcome["comparison"]["as_sample_std"]["reproduced"]
        != outcome["comparison"]["as_population_std"]["reproduced"]
    ]
    if flipped:
        print(f"\n  NOTE: verdict depends on the std convention for: {flipped}")


if __name__ == "__main__":
    main()

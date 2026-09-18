"""Figures for the statistical analysis and the prototype ablation ladder.

Produces three figures in ``analysis/``:

``power_analysis.png``
    Why the published comparisons reach the conclusions they do. Shows achieved
    power against effect size at five seeds, with each comparison marked, making
    visible that the EO-versus-MMD comparison sat in the region where a null
    result carries almost no information.

``comparison_forest.png``
    Every pairwise comparison as a difference with a confidence interval, in the
    style of a forest plot, so significant and non-significant comparisons can be
    read at a glance.

``prototype_geometry.png``
    Where each ablation variant sits in geometry space, showing which properties
    the published two-point control does and does not separate.

Run with::

    python src/make_plots.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import stats  # noqa: E402

from feature_simulator import (  # noqa: E402
    SimulatorConfig,
    class_means_from_features,
    simulate_features,
)
from geometry import geometry_report  # noqa: E402
from prototype_variants import build_ladder  # noqa: E402
from run_reported_stats import analyse, load_reported_results  # noqa: E402
from stats_utils import achieved_power, pooled_sd  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "analysis"

SIGNIFICANT_COLOUR = "#2c6fbb"
NON_SIGNIFICANT_COLOUR = "#c44e52"


def _short_label(name: str) -> str:
    return {
        "frozen": "frozen",
        "sar_only": "SAR-only",
        "mmd": "MMD",
        "eo_prototype": "EO proto",
        "synthetic_prototype": "ETF",
    }.get(name, name)


def plot_power_analysis(comparisons: list[dict], output_path: Path) -> None:
    """Achieved power against effect size, with each comparison marked."""
    figure, axis = plt.subplots(figsize=(8.5, 5.5))

    spread = float(np.mean([row["pooled_sd"] for row in comparisons]))
    effect_range = np.linspace(0.1, 7.0, 300)
    for seeds, style in ((5, "-"), (15, "--"), (30, ":")):
        powers = [achieved_power(effect, spread, seeds) for effect in effect_range]
        axis.plot(
            effect_range, powers, style, color="#666666", linewidth=1.4,
            label=f"{seeds} seeds per arm",
        )

    axis.axhline(0.8, color="#999999", linewidth=0.9, alpha=0.7)
    axis.text(6.9, 0.815, "80% power", ha="right", fontsize=8, color="#666666")

    for row in comparisons:
        colour = SIGNIFICANT_COLOUR if row["significant"] else NON_SIGNIFICANT_COLOUR
        axis.scatter(
            abs(row["difference"]), row["achieved_power"],
            s=55, color=colour, zorder=5, edgecolor="white", linewidth=0.8,
        )

    # Label only the comparison the analysis turns on, to keep the figure legible.
    for row in comparisons:
        if {row["method_a"], row["method_b"]} == {"eo_prototype", "mmd"}:
            axis.annotate(
                "EO proto vs MMD\n(the comparison that would show\n"
                "class structure transfers)",
                xy=(abs(row["difference"]), row["achieved_power"]),
                xytext=(2.1, 0.30),
                fontsize=8.5,
                arrowprops={"arrowstyle": "->", "color": "#444444", "linewidth": 0.9},
            )

    axis.set_xlabel("Effect size (accuracy, percentage points)")
    axis.set_ylabel("Probability of detecting the effect")
    axis.set_title(
        "Published comparisons against the power available at five seeds",
        fontsize=11.5,
    )
    axis.set_xlim(0, 7.0)
    axis.set_ylim(0, 1.03)
    seed_legend = axis.legend(loc="lower right", frameon=False, fontsize=9)
    axis.add_artist(seed_legend)
    axis.spines[["top", "right"]].set_visible(False)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=SIGNIFICANT_COLOUR,
                   label="significant at 0.05"),
        plt.Line2D([], [], marker="o", linestyle="", color=NON_SIGNIFICANT_COLOUR,
                   label="not significant"),
    ]
    axis.legend(handles=handles, loc="upper left", frameon=False, fontsize=9)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_comparison_forest(comparisons: list[dict], output_path: Path) -> None:
    """Every pairwise difference with a 95% interval."""
    ordered = sorted(comparisons, key=lambda row: row["difference"])
    positions = np.arange(len(ordered))

    figure, axis = plt.subplots(figsize=(8.5, 6.0))
    axis.axvline(0.0, color="#999999", linewidth=1.0, zorder=1)

    for position, row in zip(positions, ordered):
        # The interval must use the same t distribution as the reported test.
        # A normal 1.96 multiplier would draw intervals that clear zero for
        # comparisons the test calls non-significant, since at about seven
        # degrees of freedom the t critical value is nearer 2.4.
        standard_error = row["difference"] / row["t_statistic"]
        critical = stats.t.ppf(0.975, row["degrees_of_freedom"])
        half_width = critical * standard_error
        colour = SIGNIFICANT_COLOUR if row["significant"] else NON_SIGNIFICANT_COLOUR
        axis.plot(
            [row["difference"] - half_width, row["difference"] + half_width],
            [position, position],
            color=colour, linewidth=2.0, solid_capstyle="round", zorder=2,
        )
        axis.scatter(
            row["difference"], position, s=45, color=colour,
            zorder=3, edgecolor="white", linewidth=0.8,
        )

    axis.set_yticks(positions)
    axis.set_yticklabels(
        [
            f"{_short_label(row['method_a'])} − {_short_label(row['method_b'])}"
            for row in ordered
        ],
        fontsize=9,
    )
    axis.set_xlabel("Difference in accuracy (percentage points)")
    axis.set_title(
        "Pairwise comparisons with 95% intervals\n"
        "Intervals crossing zero are not statistically resolved",
        fontsize=11.5,
    )
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.tick_params(axis="y", length=0)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _realistic_source_prototypes(
    num_classes: int = 10,
    feature_dim: int = 384,
    seed: int = 0,
) -> torch.Tensor:
    """Class means with the correlation structure real encoder features show.

    Features from a single backbone share a large common component, so their
    class means have high and unevenly spread pairwise cosines - around 0.98 in
    the stub pipeline, and similar values are typical of real ViT features. The
    default simulator configuration produces near-orthogonal means, which would
    make the figure understate the gap between real prototypes and the
    equiangular frame the published control uses.
    """
    generator = torch.Generator().manual_seed(seed)
    shared = torch.randn(1, feature_dim, generator=generator)
    individual = torch.randn(num_classes, feature_dim, generator=generator)
    return torch.nn.functional.normalize(shared + 0.25 * individual, dim=1)


def plot_prototype_geometry(output_path: Path) -> None:
    """Where each ablation variant sits in geometry space."""
    source = _realistic_source_prototypes()
    num_classes = source.shape[0]
    ladder = build_ladder(source, seed=0)

    reports = {name: geometry_report(variant) for name, variant in ladder.items()}
    names = list(reports)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # Left: mean pairwise cosine against its spread. Several variants preserve
    # the source geometry exactly and so land on the same point; they are
    # annotated as a group rather than stacked on top of one another.
    axis = axes[0]
    coordinates: dict[tuple[float, float], list[str]] = {}
    for name in names:
        report = reports[name]
        key = (round(report["cosine_mean"], 2), round(report["cosine_range"], 2))
        coordinates.setdefault(key, []).append(name)

    for (x, y), group in coordinates.items():
        contains_source = "source" in group
        axis.scatter(
            x, y,
            s=190 if contains_source else 70,
            marker="*" if contains_source else "o",
            color=SIGNIFICANT_COLOUR if contains_source else "#7f9db9",
            zorder=4, edgecolor="white", linewidth=0.8,
        )
        label = "\n".join(group)
        axis.annotate(
            label, (x, y),
            textcoords="offset points", xytext=(9, 0), fontsize=8,
            va="center",
        )

    axis.axvline(
        -1.0 / (num_classes - 1), color="#999999", linestyle="--", linewidth=0.9
    )
    axis.text(
        -1.0 / (num_classes - 1), axis.get_ylim()[1] * 0.97,
        " ideal ETF cosine", fontsize=8, color="#666666", va="top",
    )
    axis.set_xlim(-0.25, 1.25)
    axis.set_xlabel("Mean pairwise cosine between prototypes")
    axis.set_ylabel("Spread of pairwise cosines")
    axis.set_title("Angular structure", fontsize=11)
    axis.spines[["top", "right"]].set_visible(False)

    # Right: what each variant keeps and destroys. Effective rank is nearly
    # identical across variants and so discriminates nothing; what distinguishes
    # them is which properties of the source they preserve.
    axis = axes[1]

    properties = ["angular geometry", "class identity", "optical directions"]

    preserved = {
        "source": [1, 1, 1],
        "rotated": [1, 1, 0],
        "label_permuted": [1, 0, 1],
        "cosine_matched": [1, 0, 0],
        "mean_shuffled": [1, 0, 0],
        "centered_shuffled": [0, 0, 0],
        "gaussian": [0, 0, 0],
        "etf": [0, 0, 0],
    }
    ordered = [name for name in preserved if name in reports]
    matrix = np.array([preserved[name] for name in ordered])

    axis.imshow(matrix, cmap="Blues", vmin=-0.4, vmax=1.3, aspect="auto")
    axis.set_xticks(range(len(properties)))
    axis.set_xticklabels(properties, fontsize=9)
    axis.set_yticks(range(len(ordered)))
    axis.set_yticklabels(ordered, fontsize=9)

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            kept = matrix[row_index, column_index]
            axis.text(
                column_index, row_index,
                "kept" if kept else "destroyed",
                ha="center", va="center", fontsize=8,
                color="white" if kept else "#444444",
            )

    axis.set_title(
        "What each variant preserves\n"
        "(the published control only contrasts the top and bottom rows)",
        fontsize=10,
    )
    axis.spines[:].set_visible(False)
    axis.tick_params(length=0)

    figure.suptitle(
        "Prototype variants in geometry space — the published control compares "
        "only 'source' against 'etf'",
        fontsize=11.5,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reported = load_reported_results()
    comparisons = analyse(reported, treat_std_as_population=False)

    figures = [
        ("power_analysis.png", lambda path: plot_power_analysis(comparisons, path)),
        ("comparison_forest.png", lambda path: plot_comparison_forest(comparisons, path)),
        ("prototype_geometry.png", plot_prototype_geometry),
    ]

    for filename, plotter in figures:
        path = OUTPUT_DIR / filename
        plotter(path)
        print(f"Wrote {path} ({path.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()

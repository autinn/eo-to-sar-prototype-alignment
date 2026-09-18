"""Tests for the corrected variant experiment.

Covers the two pieces that carry the corrections from ``analysis/CONFOUNDS.md``:
the stratified validation split, which moves model selection off the test set,
and the paired comparison, which is only possible because every variant runs on
the same seed set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_variant_experiment import (  # noqa: E402
    make_weighted_sampler,
    paired_comparison,
    split_train_validation,
)


class _FakeDataset:
    """Minimal stand-in exposing the ``targets`` attribute the split reads."""

    def __init__(self, targets: list[int]) -> None:
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return torch.zeros(3, 8, 8), self.targets[index]


def _results(source_values: list[float], other_values: list[float], name: str = "etf"):
    rows = []
    for seed, (a, b) in enumerate(zip(source_values, other_values)):
        rows.append({"variant": "source", "seed": seed, "accuracy": a})
        rows.append({"variant": name, "seed": seed, "accuracy": b})
    return rows


class TestValidationSplit:
    def test_split_is_disjoint_and_complete(self):
        dataset = _FakeDataset([index % 5 for index in range(100)])
        train, validation = split_train_validation(dataset, 0.2, seed=0)

        assert set(train.indices) & set(validation.indices) == set()
        assert len(train) + len(validation) == 100

    def test_every_class_appears_in_validation(self):
        """Stratification matters because macro-F1 is the selection metric; a
        class missing from validation makes its F1 undefined."""
        dataset = _FakeDataset([index % 10 for index in range(200)])
        _, validation = split_train_validation(dataset, 0.15, seed=0)

        validation_classes = {dataset.targets[index] for index in validation.indices}
        assert validation_classes == set(range(10))

    def test_handles_severe_imbalance(self):
        """UNICORNv2 is imbalanced about 1000:1, so rare classes must survive."""
        targets = [0] * 1000 + [1] * 100 + [2] * 3
        dataset = _FakeDataset(targets)
        _, validation = split_train_validation(dataset, 0.15, seed=0)

        validation_classes = {dataset.targets[index] for index in validation.indices}
        assert validation_classes == {0, 1, 2}

    def test_single_example_class_goes_to_training(self):
        """A class with one image cannot be split; it belongs in training rather
        than leaving training without it."""
        dataset = _FakeDataset([0] * 50 + [1])
        train, validation = split_train_validation(dataset, 0.2, seed=0)

        assert 50 in train.indices
        assert 50 not in validation.indices

    def test_is_deterministic(self):
        dataset = _FakeDataset([index % 4 for index in range(80)])
        first, _ = split_train_validation(dataset, 0.2, seed=7)
        second, _ = split_train_validation(dataset, 0.2, seed=7)
        assert first.indices == second.indices

    def test_different_seeds_give_different_splits(self):
        dataset = _FakeDataset([index % 4 for index in range(80)])
        first, _ = split_train_validation(dataset, 0.2, seed=0)
        second, _ = split_train_validation(dataset, 0.2, seed=1)
        assert first.indices != second.indices

    def test_fraction_is_approximately_respected(self):
        dataset = _FakeDataset([index % 5 for index in range(500)])
        _, validation = split_train_validation(dataset, 0.2, seed=0)
        assert 0.15 < len(validation) / 500 < 0.25


class TestPairedComparison:
    def test_reports_the_paired_difference(self):
        rows = _results([33.0, 34.0, 32.0], [31.0, 31.5, 30.5])
        result = paired_comparison(rows, "source", "etf")

        assert result["n_seeds"] == 3
        assert result["mean_paired_difference"] == pytest.approx(2.0, abs=1e-6)

    def test_pairing_is_more_sensitive_than_not_pairing(self):
        """The reason a shared seed set matters. Both variants vary a lot across
        seeds but differ consistently, so pairing removes the shared variance and
        the unpaired test does not."""
        rows = _results(
            [30.0, 35.0, 28.0, 37.0, 32.0],
            [28.7, 33.2, 26.9, 35.3, 30.4],
        )
        result = paired_comparison(rows, "source", "etf")
        assert abs(result["paired_t"]) > abs(result["unpaired_t"])

    def test_identical_differences_are_flagged_not_reported(self):
        """A perfectly constant difference makes the paired t diverge; reporting
        3e15 would look like overwhelming evidence rather than a degenerate
        denominator."""
        rows = _results([33.0, 34.0, 32.0], [31.0, 32.0, 30.0])
        result = paired_comparison(rows, "source", "etf")

        assert result["paired_t"] is None
        assert "paired_note" in result

    def test_uses_only_shared_seeds(self):
        rows = [
            {"variant": "source", "seed": 0, "accuracy": 33.0},
            {"variant": "source", "seed": 1, "accuracy": 34.0},
            {"variant": "source", "seed": 2, "accuracy": 35.0},
            {"variant": "etf", "seed": 0, "accuracy": 31.0},
            {"variant": "etf", "seed": 1, "accuracy": 30.0},
        ]
        assert paired_comparison(rows, "source", "etf")["n_seeds"] == 2

    def test_requires_at_least_two_shared_seeds(self):
        rows = [
            {"variant": "source", "seed": 0, "accuracy": 33.0},
            {"variant": "etf", "seed": 1, "accuracy": 31.0},
        ]
        with pytest.raises(ValueError):
            paired_comparison(rows, "source", "etf")

    def test_sign_follows_the_argument_order(self):
        rows = _results([33.0, 34.0, 32.0], [31.5, 33.5, 29.0])
        forward = paired_comparison(rows, "source", "etf")
        backward = paired_comparison(rows, "etf", "source")
        assert forward["mean_paired_difference"] == pytest.approx(
            -backward["mean_paired_difference"]
        )

    def test_supports_alternative_metrics(self):
        rows = []
        for seed, (a, b) in enumerate([(0.30, 0.28), (0.31, 0.27), (0.29, 0.26)]):
            rows.append({"variant": "source", "seed": seed, "macro_f1": a})
            rows.append({"variant": "etf", "seed": seed, "macro_f1": b})
        result = paired_comparison(rows, "source", "etf", metric="macro_f1")
        assert result["metric"] == "macro_f1"
        assert result["mean_paired_difference"] > 0


class TestWeightedSampler:
    def test_rebalances_an_imbalanced_dataset(self):
        """The published scripts rely on this for a 1000:1 imbalance; a rare
        class must be drawn far more often than its frequency."""
        labels = [0] * 990 + [1] * 10
        sampler = make_weighted_sampler(labels, seed=0)
        drawn = [labels[index] for index in sampler]

        rare_fraction = sum(1 for label in drawn if label == 1) / len(drawn)
        assert rare_fraction > 0.3

    def test_is_deterministic(self):
        labels = [index % 3 for index in range(60)]
        first = list(make_weighted_sampler(labels, seed=5))
        second = list(make_weighted_sampler(labels, seed=5))
        assert first == second

    def test_draws_the_dataset_length(self):
        labels = [index % 4 for index in range(80)]
        assert len(list(make_weighted_sampler(labels, seed=0))) == 80

"""Tests for the synthetic feature simulator.

The simulator is only useful if its knobs do what they say. Each test here
varies one parameter and checks that the corresponding measured property moves
in the expected direction, so that a later experiment can attribute a result to
the condition it varied rather than to a quirk of the generator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feature_simulator import (  # noqa: E402
    SimulatorConfig,
    class_counts,
    class_means_from_features,
    linear_probe_accuracy,
    simulate_features,
    simulate_paired_modalities,
)
from geometry import effective_rank, pairwise_cosine  # noqa: E402


class TestConfiguration:
    def test_rejects_invalid_parameters(self):
        with pytest.raises(ValueError):
            SimulatorConfig(num_classes=1)
        with pytest.raises(ValueError):
            SimulatorConfig(num_classes=10, feature_dim=5)
        with pytest.raises(ValueError):
            SimulatorConfig(n_per_class=0)
        with pytest.raises(ValueError):
            SimulatorConfig(imbalance_ratio=0.5)
        with pytest.raises(ValueError):
            SimulatorConfig(intrinsic_dim=0)

    def test_accepts_defaults(self):
        config = SimulatorConfig()
        assert config.num_classes == 10
        assert config.feature_dim == 384


class TestShapesAndDeterminism:
    def test_shapes(self):
        config = SimulatorConfig(num_classes=5, feature_dim=64, n_per_class=20)
        features, labels, means = simulate_features(config)
        assert features.shape == (100, 64)
        assert labels.shape == (100,)
        assert means.shape == (5, 64)

    def test_every_class_is_present(self):
        config = SimulatorConfig(num_classes=7, feature_dim=64, n_per_class=10)
        _, labels, _ = simulate_features(config)
        assert set(labels.tolist()) == set(range(7))

    def test_same_seed_reproduces(self):
        config = SimulatorConfig(num_classes=5, feature_dim=32, n_per_class=10, seed=3)
        first, _, _ = simulate_features(config)
        second, _, _ = simulate_features(config)
        assert torch.allclose(first, second)

    def test_different_seeds_differ(self):
        a, _, _ = simulate_features(
            SimulatorConfig(num_classes=5, feature_dim=32, n_per_class=10, seed=0)
        )
        b, _, _ = simulate_features(
            SimulatorConfig(num_classes=5, feature_dim=32, n_per_class=10, seed=1)
        )
        assert not torch.allclose(a, b)

    def test_features_are_not_normalized(self):
        """extract_features returns raw output; normalization happens later."""
        config = SimulatorConfig(num_classes=5, feature_dim=64, n_per_class=20)
        features, _, _ = simulate_features(config)
        norms = features.norm(dim=1)
        assert float(norms.std()) > 1e-3


class TestSeparationControl:
    def test_greater_separation_is_more_separable(self):
        """The headline control: separation must govern class separability."""
        accuracies = []
        for separation in (0.2, 1.0, 4.0):
            config = SimulatorConfig(
                num_classes=5,
                feature_dim=64,
                n_per_class=60,
                separation=separation,
                seed=0,
            )
            features, labels, _ = simulate_features(config)
            accuracies.append(linear_probe_accuracy(features, labels, num_classes=5))

        assert accuracies == sorted(accuracies), accuracies
        assert accuracies[0] < 0.6
        assert accuracies[-1] > 0.9

    def test_wider_within_class_noise_reduces_separability(self):
        config_tight = SimulatorConfig(
            num_classes=5, feature_dim=64, n_per_class=60,
            separation=1.5, within_class_cov_scale=0.3, seed=0,
        )
        config_loose = SimulatorConfig(
            num_classes=5, feature_dim=64, n_per_class=60,
            separation=1.5, within_class_cov_scale=3.0, seed=0,
        )
        tight_features, labels, _ = simulate_features(config_tight)
        loose_features, _, _ = simulate_features(config_loose)

        assert linear_probe_accuracy(
            tight_features, labels, 5
        ) > linear_probe_accuracy(loose_features, labels, 5)


class TestDimensionalityControl:
    def test_intrinsic_dim_governs_class_mean_rank(self):
        """Class means must occupy the requested number of dimensions."""
        for intrinsic in (2, 5, 9):
            config = SimulatorConfig(
                num_classes=10, feature_dim=128, n_per_class=5,
                intrinsic_dim=intrinsic, seed=0,
            )
            _, _, means = simulate_features(config)
            assert effective_rank(means) <= intrinsic + 0.5

    def test_default_intrinsic_dim_spans_k_minus_one_dimensions(self):
        """Defaulting to num_classes - 1 mirrors real class means, which lose one
        degree of freedom to centering.

        Checked as a matrix rank rather than an effective rank. Effective rank is
        the exponential of spectral entropy and equals the true rank only when
        the spectrum is flat; random class means give an uneven spectrum, so the
        entropy measure reads lower (about 7.5 here) while the subspace really is
        nine-dimensional. Effective rank is the right measure for how evenly a
        cloud fills its subspace, and the wrong one for how large that subspace
        is.
        """
        config = SimulatorConfig(num_classes=10, feature_dim=128, n_per_class=5, seed=0)
        _, _, means = simulate_features(config)

        centered = means - means.mean(dim=0, keepdim=True)
        singular_values = torch.linalg.svdvals(centered.double())

        # The tenth singular value is float32 rounding noise, around 1e-7 rather
        # than the 1e-16 exact arithmetic would give, so the threshold is set
        # relative to the largest value rather than absolutely.
        significant = singular_values > 1e-4 * float(singular_values[0])
        assert int(significant.sum()) == 9, singular_values

        # The entropy measure must still be bounded by the true rank.
        assert effective_rank(means) <= 9.0 + 1e-6

    def test_anisotropy_concentrates_variance(self):
        isotropic = SimulatorConfig(
            num_classes=4, feature_dim=64, n_per_class=100, anisotropy=0.0, seed=0
        )
        skewed = SimulatorConfig(
            num_classes=4, feature_dim=64, n_per_class=100, anisotropy=5.0, seed=0
        )
        isotropic_features, _, _ = simulate_features(isotropic)
        skewed_features, _, _ = simulate_features(skewed)

        assert effective_rank(skewed_features) < effective_rank(isotropic_features)


class TestImbalance:
    def test_balanced_by_default(self):
        counts = class_counts(SimulatorConfig(num_classes=5, n_per_class=20))
        assert counts == [20] * 5

    def test_imbalance_spans_the_requested_ratio(self):
        config = SimulatorConfig(num_classes=10, n_per_class=1000, imbalance_ratio=100.0)
        counts = class_counts(config)
        assert counts == sorted(counts, reverse=True)
        assert counts[0] / counts[-1] == pytest.approx(100.0, rel=0.15)

    def test_no_class_is_empty(self):
        """UNICORNv2 is imbalanced about 1000:1; even the rarest class must be
        represented or the labels become inconsistent."""
        config = SimulatorConfig(num_classes=10, n_per_class=100, imbalance_ratio=1000.0)
        assert min(class_counts(config)) >= 1

    def test_generated_data_matches_the_counts(self):
        config = SimulatorConfig(
            num_classes=5, feature_dim=32, n_per_class=100, imbalance_ratio=10.0
        )
        _, labels, _ = simulate_features(config)
        counts = class_counts(config)
        for class_index, expected in enumerate(counts):
            assert int((labels == class_index).sum()) == expected


class TestPairedModalities:
    def test_shapes_and_alignment(self):
        config = SimulatorConfig(num_classes=5, feature_dim=64, n_per_class=20)
        eo, sar, labels, means = simulate_paired_modalities(config)
        assert eo.shape == sar.shape == (100, 64)
        assert labels.shape == (100,)
        assert means.shape == (5, 64)

    def test_zero_gap_keeps_the_modalities_close(self):
        config = SimulatorConfig(
            num_classes=5, feature_dim=64, n_per_class=50,
            within_class_cov_scale=0.05, separation=2.0, seed=0,
        )
        eo, sar, labels, _ = simulate_paired_modalities(
            config, modality_gap=0.0, shared_structure=1.0
        )
        eo_means = class_means_from_features(eo, labels, 5)
        sar_means = class_means_from_features(sar, labels, 5)

        similarity = torch.nn.functional.cosine_similarity(eo_means, sar_means, dim=1)
        assert float(similarity.mean()) > 0.9

    def test_larger_gap_separates_the_modalities(self):
        config = SimulatorConfig(
            num_classes=5, feature_dim=64, n_per_class=50,
            within_class_cov_scale=0.05, separation=1.0, seed=0,
        )
        similarities = []
        for gap in (0.0, 1.0, 5.0):
            eo, sar, labels, _ = simulate_paired_modalities(config, modality_gap=gap)
            eo_means = class_means_from_features(eo, labels, 5)
            sar_means = class_means_from_features(sar, labels, 5)
            similarities.append(
                float(
                    torch.nn.functional.cosine_similarity(
                        eo_means, sar_means, dim=1
                    ).mean()
                )
            )

        assert similarities == sorted(similarities, reverse=True), similarities

    def test_shared_structure_governs_geometric_agreement(self):
        """The knob that matters for the research question: with shared_structure
        at zero, optical prototypes carry no information about SAR geometry, so
        alignment to them should be useless. This makes that condition testable.
        """
        config = SimulatorConfig(
            num_classes=6, feature_dim=64, n_per_class=50,
            within_class_cov_scale=0.05, separation=2.0, seed=0,
        )
        agreements = []
        for shared in (0.0, 0.5, 1.0):
            eo, sar, labels, _ = simulate_paired_modalities(
                config, modality_gap=0.5, shared_structure=shared
            )
            eo_gram = pairwise_cosine(class_means_from_features(eo, labels, 6))
            sar_gram = pairwise_cosine(class_means_from_features(sar, labels, 6))
            agreements.append(-float((eo_gram - sar_gram).abs().mean()))

        assert agreements == sorted(agreements), agreements

    def test_rejects_invalid_shared_structure(self):
        config = SimulatorConfig(num_classes=5, feature_dim=32, n_per_class=10)
        with pytest.raises(ValueError):
            simulate_paired_modalities(config, shared_structure=1.5)


class TestClassMeans:
    def test_matches_the_published_construction(self):
        """Mean-then-normalize, the order build_eo_prototypes uses."""
        features = torch.tensor([[2.0, 0.0], [4.0, 0.0], [0.0, 3.0], [0.0, 9.0]])
        labels = torch.tensor([0, 0, 1, 1])
        means = class_means_from_features(features, labels, num_classes=2)

        assert torch.allclose(means[0], torch.tensor([1.0, 0.0]), atol=1e-6)
        assert torch.allclose(means[1], torch.tensor([0.0, 1.0]), atol=1e-6)

    def test_output_is_normalized(self):
        config = SimulatorConfig(num_classes=5, feature_dim=64, n_per_class=20)
        features, labels, _ = simulate_features(config)
        means = class_means_from_features(features, labels, 5)
        assert torch.allclose(means.norm(dim=1), torch.ones(5), atol=1e-5)


class TestLinearProbe:
    def test_perfectly_separable_data_scores_high(self):
        features = torch.eye(4).repeat_interleave(25, dim=0) * 5.0
        labels = torch.arange(4).repeat_interleave(25)
        assert linear_probe_accuracy(features, labels, num_classes=4) > 0.95

    def test_pure_noise_scores_near_chance(self):
        torch.manual_seed(0)
        features = torch.randn(400, 64)
        labels = torch.randint(0, 4, (400,))
        assert linear_probe_accuracy(features, labels, num_classes=4) < 0.75


class TestEmptyClassGuard:
    """Regression: a class with no samples produced silent NaN prototypes.

    Reachable on the real pipeline, where run_variant_experiment takes the class
    count from the SAR dataset while the features come from the EO dataset. A
    NaN prototype propagates through the alignment loss into every batch
    containing that label, and NaN does not raise.
    """

    def test_raises_rather_than_returning_nan(self):
        features = torch.randn(20, 8)
        labels = torch.randint(0, 2, (20,))
        with pytest.raises(ValueError, match="No samples for class indices"):
            class_means_from_features(features, labels, num_classes=3)

    def test_error_names_the_offending_classes(self):
        features = torch.randn(30, 8)
        labels = torch.zeros(30, dtype=torch.long)
        with pytest.raises(ValueError, match=r"\[1, 2, 3\]"):
            class_means_from_features(features, labels, num_classes=4)

    def test_fully_populated_input_still_works(self):
        features = torch.randn(30, 8)
        labels = torch.arange(30) % 3
        means = class_means_from_features(features, labels, num_classes=3)
        assert means.shape == (3, 8)
        assert not torch.isnan(means).any()

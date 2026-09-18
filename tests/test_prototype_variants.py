"""Tests for the prototype ablation ladder.

The most important test here is ``test_matches_published_implementation``: the
ETF constructor is a reimplementation of ``build_synthetic_prototypes``, which
lives inside a training script and cannot be imported. If the two ever diverge,
every comparison against the published control becomes invalid, so the agreement
is asserted directly against the original source.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from geometry import etf_deviation, off_diagonal, pairwise_cosine  # noqa: E402
from prototype_variants import (  # noqa: E402
    build_ladder,
    centered_shuffled,
    cosine_matched,
    etf_prototypes,
    gaussian_prototypes,
    interpolate,
    label_permuted,
    mean_shuffled,
    rotated,
)


NUM_CLASSES = 10
FEATURE_DIM = 384


def _load_published_etf_builder():
    """Import build_synthetic_prototypes from the training script.

    The script executes its imports at module level, so it is loaded through a
    spec rather than a plain import to keep the failure mode obvious if the
    upstream file moves.
    """
    script = REPO_ROOT / "src" / "train_synthetic_prototype_alignment.py"
    if not script.is_file():
        pytest.skip(f"Upstream script not found: {script}")

    spec = importlib.util.spec_from_file_location("published_synthetic", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.build_synthetic_prototypes


def _mock_class_means(num_classes: int, feature_dim: int, seed: int) -> torch.Tensor:
    """Stand-in for real optical class means: correlated, non-uniform angles.

    Built from a shared component plus per-class variation, which reproduces the
    qualitative property that distinguishes real class means from an ETF - their
    pairwise cosines are high and unevenly spread.
    """
    generator = torch.Generator().manual_seed(seed)
    shared = torch.randn(1, feature_dim, generator=generator)
    individual = torch.randn(num_classes, feature_dim, generator=generator)
    return F.normalize(shared + 0.6 * individual, dim=1)


class TestPublishedAgreement:
    def test_matches_published_implementation(self):
        """The reimplemented ETF must equal the one used for the reported results."""
        published = _load_published_etf_builder()
        for seed in (0, 1, 42):
            expected = published(NUM_CLASSES, FEATURE_DIM, seed)
            actual = etf_prototypes(NUM_CLASSES, FEATURE_DIM, seed)
            assert torch.allclose(expected, actual, atol=1e-6), (
                f"ETF reimplementation diverges from upstream at seed {seed}"
            )

    def test_matches_across_shapes(self):
        published = _load_published_etf_builder()
        for num_classes, feature_dim in ((5, 64), (10, 384), (20, 256)):
            assert torch.allclose(
                published(num_classes, feature_dim, 0),
                etf_prototypes(num_classes, feature_dim, 0),
                atol=1e-6,
            )


class TestConstructors:
    def test_all_variants_are_normalized(self):
        source = _mock_class_means(NUM_CLASSES, FEATURE_DIM, seed=0)
        for name, prototypes in build_ladder(source, seed=0).items():
            norms = prototypes.norm(dim=1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), (
                f"{name} is not unit-normalized"
            )

    def test_all_variants_preserve_shape(self):
        source = _mock_class_means(NUM_CLASSES, FEATURE_DIM, seed=0)
        for name, prototypes in build_ladder(source, seed=0).items():
            assert prototypes.shape == source.shape, f"{name} changed shape"

    def test_etf_rejects_too_few_dimensions(self):
        with pytest.raises(ValueError):
            etf_prototypes(num_classes=10, feature_dim=5, seed=0)

    def test_constructors_are_deterministic(self):
        for builder in (etf_prototypes, gaussian_prototypes):
            assert torch.equal(
                builder(NUM_CLASSES, FEATURE_DIM, 7),
                builder(NUM_CLASSES, FEATURE_DIM, 7),
            )

    def test_different_seeds_give_different_prototypes(self):
        a = gaussian_prototypes(NUM_CLASSES, FEATURE_DIM, 0)
        b = gaussian_prototypes(NUM_CLASSES, FEATURE_DIM, 1)
        assert not torch.allclose(a, b, atol=1e-3)

    def test_gaussian_prototypes_are_near_orthogonal_in_high_dimension(self):
        """Independent Gaussian directions concentrate near orthogonality as the
        dimension grows, which is what places this variant between an ETF and
        real correlated class means."""
        prototypes = gaussian_prototypes(NUM_CLASSES, FEATURE_DIM, seed=0)
        values = off_diagonal(pairwise_cosine(prototypes))
        assert float(values.abs().max()) < 0.2


class TestRotated:
    def test_preserves_angles_exactly(self):
        source = _mock_class_means(NUM_CLASSES, FEATURE_DIM, seed=0)
        turned = rotated(source, seed=1)
        assert torch.allclose(pairwise_cosine(source), pairwise_cosine(turned), atol=1e-5)

    def test_changes_directions(self):
        source = _mock_class_means(NUM_CLASSES, FEATURE_DIM, seed=0)
        turned = rotated(source, seed=1)
        assert not torch.allclose(source, turned, atol=1e-2)

    def test_is_deterministic(self):
        source = _mock_class_means(NUM_CLASSES, FEATURE_DIM, seed=0)
        assert torch.equal(rotated(source, seed=2), rotated(source, seed=2))


class TestLabelPermuted:
    def test_preserves_the_multiset_of_angles(self):
        """Permuting which class owns which prototype cannot change the set of
        pairwise angles, only their assignment."""
        source = _mock_class_means(NUM_CLASSES, FEATURE_DIM, seed=0)
        permuted = label_permuted(source, seed=1)
        original = off_diagonal(pairwise_cosine(source)).sort().values
        shuffled = off_diagonal(pairwise_cosine(permuted)).sort().values
        assert torch.allclose(original, shuffled, atol=1e-6)

    def test_changes_the_assignment(self):
        source = _mock_class_means(NUM_CLASSES, FEATURE_DIM, seed=0)
        permuted = label_permuted(source, seed=1)
        assert not torch.allclose(source, permuted, atol=1e-6)

    def test_every_prototype_is_reused_exactly_once(self):
        source = _mock_class_means(NUM_CLASSES, FEATURE_DIM, seed=0)
        permuted = label_permuted(source, seed=1)
        for row in permuted:
            matches = [torch.allclose(row, other, atol=1e-6) for other in source]
            assert sum(matches) == 1


class TestCosineMatched:
    def test_reproduces_the_source_gram_matrix(self):
        source = _mock_class_means(NUM_CLASSES, FEATURE_DIM, seed=0)
        matched = cosine_matched(source, seed=1)
        assert torch.allclose(
            pairwise_cosine(source), pairwise_cosine(matched), atol=1e-4
        )

    def test_uses_different_directions(self):
        source = _mock_class_means(NUM_CLASSES, FEATURE_DIM, seed=0)
        matched = cosine_matched(source, seed=1)
        assert not torch.allclose(source, matched, atol=1e-2)

    def test_matching_an_etf_yields_an_etf(self):
        source = etf_prototypes(NUM_CLASSES, FEATURE_DIM, seed=0)
        assert etf_deviation(cosine_matched(source, seed=1)) < 1e-4


class TestMeanShuffled:
    def test_barely_changes_angles(self):
        """Per-coordinate shuffling leaves pairwise cosines essentially intact.

        This pins a measured negative result. The variant was written expecting
        it would decorrelate the prototypes, but correlation between class means
        lives in the shared component of each coordinate, and permuting values
        within a coordinate preserves that component exactly. Documented in the
        function's docstring; ``centered_shuffled`` is the variant that works.
        """
        source = _mock_class_means(NUM_CLASSES, FEATURE_DIM, seed=0)
        shuffled = mean_shuffled(source, seed=1)
        source_mean = float(off_diagonal(pairwise_cosine(source)).mean())
        shuffled_mean = float(off_diagonal(pairwise_cosine(shuffled)).mean())
        assert abs(shuffled_mean - source_mean) < 0.01

    def test_preserves_coordinate_marginals_up_to_renormalization(self):
        """Each column holds the same multiset of values, before the final
        row-normalization rescales them slightly.

        The renormalization is what keeps every variant unit-norm, so the
        marginals are preserved only approximately in the returned tensor. The
        correlation between the sorted columns is what actually pins the
        property.
        """
        source = _mock_class_means(NUM_CLASSES, FEATURE_DIM, seed=0)
        shuffled = mean_shuffled(source, seed=1)
        for dimension in (0, 17, 100, 383):
            original = source[:, dimension].sort().values
            permuted = shuffled[:, dimension].sort().values
            correlation = torch.corrcoef(torch.stack([original, permuted]))[0, 1]
            assert float(correlation) > 0.99

    def test_changes_the_assignment(self):
        source = _mock_class_means(NUM_CLASSES, FEATURE_DIM, seed=0)
        assert not torch.allclose(source, mean_shuffled(source, seed=1), atol=1e-6)

    def test_is_deterministic(self):
        source = _mock_class_means(NUM_CLASSES, FEATURE_DIM, seed=0)
        assert torch.equal(mean_shuffled(source, seed=3), mean_shuffled(source, seed=3))


class TestCenteredShuffled:
    def test_destroys_angular_structure(self):
        """Removing the shared mean before shuffling genuinely decorrelates the
        prototypes, which is what mean_shuffled was expected to do."""
        source = _mock_class_means(NUM_CLASSES, FEATURE_DIM, seed=0)
        shuffled = centered_shuffled(source, seed=1)
        source_mean = float(off_diagonal(pairwise_cosine(source)).mean())
        shuffled_mean = float(off_diagonal(pairwise_cosine(shuffled)).mean())
        assert shuffled_mean < source_mean - 0.3, (
            f"expected a large drop, got {source_mean:.3f} -> {shuffled_mean:.3f}"
        )

    def test_is_deterministic(self):
        source = _mock_class_means(NUM_CLASSES, FEATURE_DIM, seed=0)
        assert torch.equal(
            centered_shuffled(source, seed=3), centered_shuffled(source, seed=3)
        )


class TestInterpolate:
    def test_endpoints_recover_the_inputs(self):
        a = gaussian_prototypes(NUM_CLASSES, FEATURE_DIM, seed=0)
        b = etf_prototypes(NUM_CLASSES, FEATURE_DIM, seed=1)
        assert torch.allclose(interpolate(a, b, 0.0), a, atol=1e-5)
        assert torch.allclose(interpolate(a, b, 1.0), b, atol=1e-5)

    def test_midpoint_lies_between(self):
        a = gaussian_prototypes(NUM_CLASSES, FEATURE_DIM, seed=0)
        b = gaussian_prototypes(NUM_CLASSES, FEATURE_DIM, seed=1)
        midpoint = interpolate(a, b, 0.5)
        to_a = float(F.cosine_similarity(midpoint, a, dim=1).mean())
        to_b = float(F.cosine_similarity(midpoint, b, dim=1).mean())
        assert to_a > 0.5 and to_b > 0.5

    def test_sweep_moves_monotonically(self):
        a = gaussian_prototypes(NUM_CLASSES, FEATURE_DIM, seed=0)
        b = gaussian_prototypes(NUM_CLASSES, FEATURE_DIM, seed=1)
        similarities = [
            float(F.cosine_similarity(interpolate(a, b, alpha), b, dim=1).mean())
            for alpha in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        assert similarities == sorted(similarities)

    def test_rejects_bad_arguments(self):
        a = gaussian_prototypes(NUM_CLASSES, FEATURE_DIM, seed=0)
        b = gaussian_prototypes(NUM_CLASSES, 128, seed=1)
        with pytest.raises(ValueError):
            interpolate(a, b, 0.5)
        with pytest.raises(ValueError):
            interpolate(a, a, 1.5)


class TestLadder:
    def test_contains_every_variant(self):
        source = _mock_class_means(NUM_CLASSES, FEATURE_DIM, seed=0)
        ladder = build_ladder(source, seed=0)
        assert set(ladder) == {
            "source",
            "etf",
            "gaussian",
            "rotated",
            "label_permuted",
            "cosine_matched",
            "mean_shuffled",
            "centered_shuffled",
        }

    def test_separates_geometry_from_identity(self):
        """The ladder's purpose: rotated and cosine_matched keep the source
        geometry while etf and gaussian do not. A gap between these groups
        isolates geometry from the specific optical directions."""
        source = _mock_class_means(NUM_CLASSES, FEATURE_DIM, seed=0)
        ladder = build_ladder(source, seed=0)
        source_deviation = etf_deviation(ladder["source"])

        for name in ("rotated", "cosine_matched", "label_permuted"):
            assert etf_deviation(ladder[name]) == pytest.approx(
                source_deviation, abs=1e-3
            ), f"{name} should preserve the source geometry"

        assert etf_deviation(ladder["etf"]) < source_deviation

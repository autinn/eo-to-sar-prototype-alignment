"""Tests for the error-placement diagnostics.

Every assertion here is constructed so the answer is known before running it.
Residual is placed deliberately - entirely in the leading singular direction, or
entirely in the head's nullspace - and the metric must report the value the
geometry dictates. This matters more than usual because the failure mode of a
mistake in this module is not a crash but a plausible number, and a plausible
number is exactly what would end up in a write-up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from error_placement import (  # noqa: E402
    alignment_residual,
    head_singular_basis,
    logit_perturbation,
    nullspace_energy_fraction,
    placement_report,
    residual_concentration,
)


NUM_CLASSES = 10
FEATURE_DIM = 384
N_SAMPLES = 2000


def _head(seed: int = 0) -> torch.Tensor:
    """A classifier head shaped like the one in DINOClassifier."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(NUM_CLASSES, FEATURE_DIM, generator=generator) / (FEATURE_DIM**0.5)


def _full_nullspace_basis(head_weight: torch.Tensor) -> torch.Tensor:
    """Rows spanning the part of feature space the head cannot see."""
    _, _, right_vectors = torch.linalg.svd(head_weight.to(torch.float64), full_matrices=True)
    return right_vectors[head_weight.shape[0] :]


class TestSingularBasis:
    def test_shapes(self):
        right_vectors, singular_values = head_singular_basis(_head())
        assert right_vectors.shape == (NUM_CLASSES, FEATURE_DIM)
        assert singular_values.shape == (NUM_CLASSES,)

    def test_rows_are_orthonormal(self):
        """Pins the Vh convention. If a transpose were wrong this would fail."""
        right_vectors, _ = head_singular_basis(_head())
        product = right_vectors @ right_vectors.T
        assert torch.allclose(
            product, torch.eye(NUM_CLASSES, dtype=torch.float64), atol=1e-10
        )

    def test_reconstructs_the_head(self):
        """The strongest check that the decomposition is the right way round."""
        head = _head()
        left, singular_values, right_vectors = torch.linalg.svd(
            head.to(torch.float64), full_matrices=False
        )
        reconstructed = left @ torch.diag(singular_values) @ right_vectors
        assert torch.allclose(reconstructed, head.to(torch.float64), atol=1e-10)

    def test_singular_values_are_sorted_descending(self):
        _, singular_values = head_singular_basis(_head())
        assert torch.all(singular_values[:-1] >= singular_values[1:])

    def test_rejects_non_matrix(self):
        with pytest.raises(ValueError):
            head_singular_basis(torch.randn(10))


class TestConcentrationReferencePoints:
    """The three values that define the scale, each analytically determined."""

    def test_nullspace_residual_scores_zero(self):
        """Residual the head cannot read must score exactly zero."""
        head = _head()
        nullspace = _full_nullspace_basis(head)
        generator = torch.Generator().manual_seed(1)
        residual = torch.randn(
            N_SAMPLES, nullspace.shape[0], generator=generator, dtype=torch.float64
        ) @ nullspace

        right_vectors, singular_values = head_singular_basis(head)
        assert residual_concentration(residual, right_vectors, singular_values) < 1e-8

    def test_isotropic_residual_scores_one(self):
        """Uniformly spread residual is the neutral reference."""
        head = _head()
        generator = torch.Generator().manual_seed(2)
        residual = torch.randn(N_SAMPLES, FEATURE_DIM, generator=generator)

        right_vectors, singular_values = head_singular_basis(head)
        value = residual_concentration(residual, right_vectors, singular_values)
        assert value == pytest.approx(1.0, abs=0.1)

    def test_single_direction_residual_has_an_exact_value(self):
        """Residual confined to singular direction ``i`` scores exactly ``D * w_i``.

        With all energy on one direction, ``p_i`` equals the total energy and every
        other ``p_j`` is zero, so the weighted sum is ``w_i * E_total`` and the
        ratio against ``E_total / D`` collapses to ``D * w_i``. Asserting the
        closed form rather than an inequality pins the normalization exactly.
        """
        head = _head()
        right_vectors, singular_values = head_singular_basis(head)
        weights = singular_values**2 / (singular_values**2).sum()
        generator = torch.Generator().manual_seed(3)

        for index in (0, 4, NUM_CLASSES - 1):
            amplitudes = torch.randn(
                N_SAMPLES, 1, generator=generator, dtype=torch.float64
            )
            residual = amplitudes @ right_vectors[index : index + 1]
            expected = FEATURE_DIM * float(weights[index])
            assert residual_concentration(
                residual, right_vectors, singular_values
            ) == pytest.approx(expected, rel=1e-6)

    def test_ordering_across_the_reference_cases(self):
        """The metric must rank the reference cases in the expected order.

        Concentrating on the most-read direction is worst, then the least-read
        read-direction, then isotropic, then invisible.
        """
        head = _head()
        right_vectors, singular_values = head_singular_basis(head)
        generator = torch.Generator().manual_seed(4)
        nullspace = _full_nullspace_basis(head)

        leading = (
            torch.randn(N_SAMPLES, 1, generator=generator, dtype=torch.float64)
            @ right_vectors[:1]
        )
        weakest = (
            torch.randn(N_SAMPLES, 1, generator=generator, dtype=torch.float64)
            @ right_vectors[-1:]
        )
        isotropic = torch.randn(
            N_SAMPLES, FEATURE_DIM, generator=generator, dtype=torch.float64
        )
        invisible = (
            torch.randn(
                N_SAMPLES, nullspace.shape[0], generator=generator, dtype=torch.float64
            )
            @ nullspace
        )

        values = [
            residual_concentration(residual, right_vectors, singular_values)
            for residual in (leading, weakest, isotropic, invisible)
        ]
        assert values == sorted(values, reverse=True), values
        assert values[2] == pytest.approx(1.0, abs=0.05)
        assert values[3] < 1e-8


class TestScaleInvariance:
    def test_scaling_the_residual_does_not_change_concentration(self):
        """Placement must be independent of magnitude; this is the property that
        makes concentration and residual size separable measurements."""
        head = _head()
        right_vectors, singular_values = head_singular_basis(head)
        generator = torch.Generator().manual_seed(5)
        residual = torch.randn(N_SAMPLES, FEATURE_DIM, generator=generator)

        baseline = residual_concentration(residual, right_vectors, singular_values)
        for scale in (1e-4, 0.1, 10.0, 1e4):
            scaled = residual_concentration(
                residual * scale, right_vectors, singular_values
            )
            assert scaled == pytest.approx(baseline, rel=1e-8)

    def test_zero_residual_returns_zero(self):
        head = _head()
        right_vectors, singular_values = head_singular_basis(head)
        residual = torch.zeros(N_SAMPLES, FEATURE_DIM)
        assert residual_concentration(residual, right_vectors, singular_values) == 0.0

    def test_rejects_mismatched_dimensions(self):
        head = _head()
        right_vectors, singular_values = head_singular_basis(head)
        with pytest.raises(ValueError):
            residual_concentration(
                torch.randn(10, 128), right_vectors, singular_values
            )
        with pytest.raises(ValueError):
            residual_concentration(torch.randn(384), right_vectors, singular_values)


class TestNullspaceFraction:
    def test_nullspace_residual_is_entirely_invisible(self):
        head = _head()
        nullspace = _full_nullspace_basis(head)
        generator = torch.Generator().manual_seed(6)
        residual = torch.randn(
            N_SAMPLES, nullspace.shape[0], generator=generator, dtype=torch.float64
        ) @ nullspace

        assert nullspace_energy_fraction(residual, head) == pytest.approx(1.0, abs=1e-8)

    def test_read_subspace_residual_is_entirely_visible(self):
        head = _head()
        right_vectors, _ = head_singular_basis(head)
        generator = torch.Generator().manual_seed(7)
        residual = (
            torch.randn(N_SAMPLES, NUM_CLASSES, generator=generator, dtype=torch.float64)
            @ right_vectors
        )

        assert nullspace_energy_fraction(residual, head) == pytest.approx(0.0, abs=1e-8)

    def test_isotropic_residual_is_mostly_invisible(self):
        """With ten classes in 384 dimensions the head reads about 2.6% of the
        space, so most isotropic error is invisible to it. This is the structural
        fact that makes placement worth measuring."""
        head = _head()
        generator = torch.Generator().manual_seed(8)
        residual = torch.randn(N_SAMPLES, FEATURE_DIM, generator=generator)

        expected = 1.0 - NUM_CLASSES / FEATURE_DIM
        assert nullspace_energy_fraction(residual, head) == pytest.approx(
            expected, abs=0.02
        )

    def test_zero_residual_returns_zero(self):
        assert nullspace_energy_fraction(torch.zeros(10, FEATURE_DIM), _head()) == 0.0


class TestLogitPerturbation:
    def test_nullspace_residual_leaves_logits_untouched(self):
        """The ground-truth check that the nullspace really is invisible."""
        head = _head()
        nullspace = _full_nullspace_basis(head)
        generator = torch.Generator().manual_seed(9)
        residual = torch.randn(
            100, nullspace.shape[0], generator=generator, dtype=torch.float64
        ) @ nullspace

        assert logit_perturbation(residual, head)["logit_shift_max_abs"] < 1e-10

    def test_read_subspace_residual_moves_logits(self):
        head = _head()
        right_vectors, _ = head_singular_basis(head)
        generator = torch.Generator().manual_seed(10)
        residual = (
            torch.randn(100, NUM_CLASSES, generator=generator, dtype=torch.float64)
            @ right_vectors
        )

        assert logit_perturbation(residual, head)["logit_shift_max_abs"] > 1e-3

    def test_equal_magnitude_residuals_differ_in_effect(self):
        """The central claim of the module, stated as a test: two residuals of
        identical size can have completely different downstream consequences
        depending only on where they point."""
        head = _head()
        right_vectors, _ = head_singular_basis(head)
        nullspace = _full_nullspace_basis(head)
        generator = torch.Generator().manual_seed(11)

        visible = (
            torch.randn(500, NUM_CLASSES, generator=generator, dtype=torch.float64)
            @ right_vectors
        )
        hidden = (
            torch.randn(500, nullspace.shape[0], generator=generator, dtype=torch.float64)
            @ nullspace
        )

        # Match the two residuals to the same overall magnitude.
        hidden = hidden * (visible.norm() / hidden.norm())
        assert float(visible.norm()) == pytest.approx(float(hidden.norm()), rel=1e-6)

        visible_shift = logit_perturbation(visible, head)["logit_shift_rms"]
        hidden_shift = logit_perturbation(hidden, head)["logit_shift_rms"]
        assert visible_shift > 1000 * hidden_shift


class TestAlignmentResidual:
    def test_perfect_alignment_gives_zero_residual(self):
        generator = torch.Generator().manual_seed(12)
        prototypes = torch.randn(NUM_CLASSES, FEATURE_DIM, generator=generator)
        labels = torch.arange(NUM_CLASSES)
        residual = alignment_residual(prototypes, prototypes, labels)
        assert float(residual.abs().max()) < 1e-10

    def test_norm_matches_the_cosine_objective(self):
        """Squared residual norm is 2 - 2cos, the quantity the loss minimizes."""
        generator = torch.Generator().manual_seed(13)
        features = torch.randn(50, FEATURE_DIM, generator=generator)
        prototypes = torch.randn(NUM_CLASSES, FEATURE_DIM, generator=generator)
        labels = torch.randint(0, NUM_CLASSES, (50,), generator=generator)

        residual = alignment_residual(features, prototypes, labels)
        normalized_features = torch.nn.functional.normalize(
            features.to(torch.float64), dim=1
        )
        normalized_prototypes = torch.nn.functional.normalize(
            prototypes.to(torch.float64), dim=1
        )
        cosine = (normalized_features * normalized_prototypes[labels]).sum(dim=1)

        assert torch.allclose(residual.pow(2).sum(dim=1), 2 - 2 * cosine, atol=1e-8)

    def test_rejects_mismatched_labels(self):
        with pytest.raises(ValueError):
            alignment_residual(
                torch.randn(10, FEATURE_DIM),
                torch.randn(NUM_CLASSES, FEATURE_DIM),
                torch.arange(5),
            )


class TestPlacementReport:
    def test_contains_every_diagnostic(self):
        head = _head()
        generator = torch.Generator().manual_seed(14)
        residual = torch.randn(100, FEATURE_DIM, generator=generator)

        report = placement_report(residual, head)
        assert set(report) == {
            "concentration",
            "nullspace_fraction",
            "residual_rms",
            "residual_mean_norm",
            "logit_shift_mean_abs",
            "logit_shift_max_abs",
            "logit_shift_rms",
        }

    def test_values_are_json_serializable(self):
        import json

        head = _head()
        residual = torch.randn(100, FEATURE_DIM)
        report = placement_report(residual, head)
        json.dumps(report)
        assert all(isinstance(value, float) for value in report.values())

    def test_separates_placement_from_magnitude(self):
        """Doubling the residual doubles the magnitude measures and leaves the
        placement measures alone."""
        head = _head()
        generator = torch.Generator().manual_seed(15)
        residual = torch.randn(200, FEATURE_DIM, generator=generator)

        base = placement_report(residual, head)
        doubled = placement_report(residual * 2.0, head)

        assert doubled["concentration"] == pytest.approx(base["concentration"], rel=1e-8)
        assert doubled["nullspace_fraction"] == pytest.approx(
            base["nullspace_fraction"], rel=1e-8
        )
        assert doubled["residual_rms"] == pytest.approx(2.0 * base["residual_rms"], rel=1e-8)

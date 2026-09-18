"""Tests for the geometric descriptors.

Most assertions here have analytic answers: a simplex ETF has a known pairwise
cosine and a known rank, and a rotation is an isometry. Pinning those exactly is
what makes it safe to trust the same functions on real prototypes, where no
ground truth is available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from geometry import (  # noqa: E402
    cosine_summary,
    effective_rank,
    etf_deviation,
    geometry_report,
    isotropy,
    mean_norm,
    off_diagonal,
    pairwise_cosine,
    participation_ratio,
)
from prototype_variants import etf_prototypes, gaussian_prototypes, rotated  # noqa: E402


NUM_CLASSES = 10
FEATURE_DIM = 384


class TestPairwiseCosine:
    def test_diagonal_is_one(self):
        prototypes = gaussian_prototypes(NUM_CLASSES, FEATURE_DIM, seed=0)
        gram = pairwise_cosine(prototypes)
        assert torch.allclose(gram.diagonal(), torch.ones(NUM_CLASSES, dtype=torch.float64))

    def test_symmetric(self):
        prototypes = gaussian_prototypes(NUM_CLASSES, FEATURE_DIM, seed=0)
        gram = pairwise_cosine(prototypes)
        assert torch.allclose(gram, gram.T)

    def test_invariant_to_row_scaling(self):
        """Cosine ignores magnitude, so scaling rows must not change the result."""
        prototypes = gaussian_prototypes(NUM_CLASSES, FEATURE_DIM, seed=0)
        scaled = prototypes * torch.arange(1, NUM_CLASSES + 1).unsqueeze(1).float()
        assert torch.allclose(
            pairwise_cosine(prototypes), pairwise_cosine(scaled), atol=1e-10
        )

    def test_rejects_non_matrix(self):
        with pytest.raises(ValueError):
            pairwise_cosine(torch.randn(5))

    def test_off_diagonal_count(self):
        matrix = torch.arange(9.0).reshape(3, 3)
        assert off_diagonal(matrix).numel() == 6


class TestEtfProperties:
    """A simplex ETF is fully determined analytically, so every quantity below
    has an exact expected value."""

    def test_pairwise_cosine_is_exactly_uniform(self):
        prototypes = etf_prototypes(NUM_CLASSES, FEATURE_DIM, seed=0)
        values = off_diagonal(pairwise_cosine(prototypes))
        expected = -1.0 / (NUM_CLASSES - 1)
        assert torch.allclose(
            values, torch.full_like(values, expected), atol=1e-6
        ), f"spread was {float(values.max() - values.min()):.2e}"

    def test_deviation_from_ideal_is_negligible(self):
        prototypes = etf_prototypes(NUM_CLASSES, FEATURE_DIM, seed=0)
        assert etf_deviation(prototypes) < 1e-6

    def test_cosine_spread_is_zero(self):
        summary = cosine_summary(etf_prototypes(NUM_CLASSES, FEATURE_DIM, seed=0))
        assert summary["cosine_range"] < 1e-6
        assert summary["cosine_std"] < 1e-6
        assert summary["cosine_mean"] == pytest.approx(-1.0 / (NUM_CLASSES - 1), abs=1e-6)

    def test_spans_exactly_k_minus_one_dimensions(self):
        """The centering step removes one degree of freedom, so K prototypes
        occupy a (K-1)-dimensional subspace."""
        prototypes = etf_prototypes(NUM_CLASSES, FEATURE_DIM, seed=0)
        assert effective_rank(prototypes) == pytest.approx(NUM_CLASSES - 1, abs=0.1)

    def test_rows_are_unit_norm(self):
        prototypes = etf_prototypes(NUM_CLASSES, FEATURE_DIM, seed=0)
        assert mean_norm(prototypes) == pytest.approx(1.0, abs=1e-6)

    def test_holds_across_class_counts(self):
        for num_classes in (3, 5, 10, 20):
            prototypes = etf_prototypes(num_classes, FEATURE_DIM, seed=0)
            assert etf_deviation(prototypes) < 1e-6

    def test_real_prototypes_are_not_an_etf(self):
        """The discriminating power of etf_deviation: random prototypes should
        register a clearly non-zero deviation."""
        prototypes = gaussian_prototypes(NUM_CLASSES, FEATURE_DIM, seed=0)
        assert etf_deviation(prototypes) > 0.01


class TestRotationInvariance:
    """A rotation preserves every angle, so all angle-derived descriptors must be
    unchanged while the prototypes themselves move."""

    def test_gram_matrix_is_preserved(self):
        prototypes = gaussian_prototypes(NUM_CLASSES, FEATURE_DIM, seed=0)
        turned = rotated(prototypes, seed=1)
        assert torch.allclose(
            pairwise_cosine(prototypes), pairwise_cosine(turned), atol=1e-5
        )

    def test_prototypes_actually_move(self):
        prototypes = gaussian_prototypes(NUM_CLASSES, FEATURE_DIM, seed=0)
        turned = rotated(prototypes, seed=1)
        assert not torch.allclose(prototypes, turned, atol=1e-3)

    def test_descriptors_are_preserved(self):
        prototypes = gaussian_prototypes(NUM_CLASSES, FEATURE_DIM, seed=0)
        turned = rotated(prototypes, seed=1)
        assert effective_rank(turned) == pytest.approx(effective_rank(prototypes), rel=1e-3)
        assert etf_deviation(turned) == pytest.approx(etf_deviation(prototypes), abs=1e-5)

    def test_rotated_etf_is_still_an_etf(self):
        prototypes = etf_prototypes(NUM_CLASSES, FEATURE_DIM, seed=0)
        assert etf_deviation(rotated(prototypes, seed=3)) < 1e-5


class TestOccupancyMeasures:
    def test_effective_rank_of_a_single_direction(self):
        """All variance in one direction gives an effective rank of 1."""
        features = torch.zeros(50, FEATURE_DIM)
        features[:, 0] = torch.linspace(-1.0, 1.0, 50)
        assert effective_rank(features) == pytest.approx(1.0, abs=0.01)

    def test_effective_rank_of_isotropic_noise(self):
        """Isotropic noise in D dimensions should occupy close to D dimensions."""
        torch.manual_seed(0)
        features = torch.randn(4000, 20)
        assert effective_rank(features) > 15.0

    def test_effective_rank_is_bounded_by_dimension(self):
        torch.manual_seed(0)
        features = torch.randn(100, 8)
        assert effective_rank(features) <= 8.0 + 1e-6

    def test_participation_ratio_agrees_with_effective_rank_ordering(self):
        """Two different occupancy measures should rank the same clouds the same
        way, which guards against a conclusion that is an artefact of one metric."""
        torch.manual_seed(0)
        spread = torch.randn(500, 16)
        concentrated = torch.randn(500, 16) * torch.tensor([10.0] + [0.1] * 15)
        assert effective_rank(spread) > effective_rank(concentrated)
        assert participation_ratio(spread) > participation_ratio(concentrated)

    def test_isotropy_is_bounded(self):
        torch.manual_seed(0)
        value = isotropy(torch.randn(200, 10))
        assert 0.0 <= value <= 1.0

    def test_isotropy_detects_a_flattened_cloud(self):
        torch.manual_seed(0)
        features = torch.randn(200, 10)
        features[:, -1] *= 1e-6
        assert isotropy(features) < 0.01

    def test_degenerate_input_returns_zero(self):
        assert effective_rank(torch.zeros(10, 5)) == 0.0
        assert participation_ratio(torch.zeros(10, 5)) == 0.0
        assert isotropy(torch.zeros(10, 5)) == 0.0


class TestGeometryReport:
    def test_contains_every_descriptor(self):
        report = geometry_report(etf_prototypes(NUM_CLASSES, FEATURE_DIM, seed=0))
        expected_keys = {
            "num_classes",
            "feature_dim",
            "mean_norm",
            "etf_deviation",
            "effective_rank",
            "participation_ratio",
            "isotropy",
            "cosine_mean",
            "cosine_std",
            "cosine_min",
            "cosine_max",
            "cosine_range",
        }
        assert set(report) == expected_keys

    def test_values_are_json_serializable(self):
        """save_json in utils.py has no encoder for numpy or torch types, so the
        report must contain only builtins."""
        import json

        report = geometry_report(gaussian_prototypes(NUM_CLASSES, FEATURE_DIM, seed=0))
        json.dumps(report)
        assert all(isinstance(value, (int, float)) for value in report.values())

    def test_reports_correct_shape(self):
        report = geometry_report(etf_prototypes(7, 128, seed=0))
        assert report["num_classes"] == 7
        assert report["feature_dim"] == 128

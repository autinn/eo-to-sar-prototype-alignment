"""Geometric descriptors for class prototypes and feature clouds.

The alignment method compares two kinds of prototype: class means of real optical
features, and a simplex equiangular tight frame carrying no optical information.
These differ in several geometric properties at once - the spread of their
pairwise angles, how many dimensions they occupy, how evenly they fill those
dimensions - and the published comparison does not separate them. The functions
here measure each property individually so that prototype variants can be placed
on a common scale.

All functions accept a ``(K, D)`` prototype matrix or ``(N, D)`` feature matrix
and return plain Python floats, so results can be written straight to JSON by
``save_json`` in ``utils.py``, which has no numpy or torch encoder.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_cosine(prototypes: torch.Tensor) -> torch.Tensor:
    """Full ``(K, K)`` matrix of pairwise cosine similarities."""
    if prototypes.ndim != 2:
        raise ValueError(f"Expected a 2-D (K, D) matrix, got shape {tuple(prototypes.shape)}.")
    normalized = F.normalize(prototypes.to(torch.float64), dim=1)
    return normalized @ normalized.T


def off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    """The off-diagonal entries of a square matrix as a flat vector."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Expected a square matrix.")
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def cosine_summary(prototypes: torch.Tensor) -> dict[str, float]:
    """Summary of the off-diagonal pairwise cosine distribution.

    A simplex ETF has zero spread by construction; real class means typically do
    not, and the spread is one of the properties that distinguishes them.
    """
    values = off_diagonal(pairwise_cosine(prototypes))
    return {
        "cosine_mean": float(values.mean()),
        "cosine_std": float(values.std(unbiased=False)),
        "cosine_min": float(values.min()),
        "cosine_max": float(values.max()),
        "cosine_range": float(values.max() - values.min()),
    }


def etf_deviation(prototypes: torch.Tensor) -> float:
    """Largest departure from the ideal ETF cosine of ``-1/(K-1)``.

    Zero exactly when the prototypes form a simplex equiangular tight frame.
    """
    num_classes = prototypes.shape[0]
    if num_classes < 2:
        raise ValueError("At least two prototypes are required.")
    ideal = -1.0 / (num_classes - 1)
    values = off_diagonal(pairwise_cosine(prototypes))
    return float((values - ideal).abs().max())


def _singular_values(features: torch.Tensor) -> torch.Tensor:
    """Singular values of a mean-centred feature matrix, in float64."""
    if features.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got shape {tuple(features.shape)}.")
    centered = features.to(torch.float64) - features.to(torch.float64).mean(dim=0, keepdim=True)
    return torch.linalg.svdvals(centered)


def effective_rank(features: torch.Tensor) -> float:
    """Effective rank: the exponential of the spectral entropy.

    Equals the true rank when the spectrum is flat and drops toward 1 as variance
    concentrates in a single direction. Roy and Vetterli (2007).
    """
    values = _singular_values(features)
    positive = values[values > 1e-12]
    if positive.numel() == 0:
        return 0.0
    probabilities = positive / positive.sum()
    entropy = -(probabilities * probabilities.log()).sum()
    return float(entropy.exp())


def participation_ratio(features: torch.Tensor) -> float:
    """Participation ratio of the eigenvalue spectrum.

    A second occupancy measure that weights the spectrum differently from
    ``effective_rank``; reporting both guards against a conclusion that depends on
    the choice of measure.
    """
    values = _singular_values(features)
    eigenvalues = values**2
    total = eigenvalues.sum()
    if float(total) <= 1e-12:
        return 0.0
    return float(total**2 / (eigenvalues**2).sum())


def isotropy(features: torch.Tensor) -> float:
    """Ratio of smallest to largest singular value, in ``[0, 1]``.

    One means the cloud is perfectly spherical; near zero means it is flattened
    into a lower-dimensional subspace.

    Only the singular values the data can actually support are considered. An
    ``(N, D)`` matrix with ``N < D`` has at most ``N - 1`` non-zero singular
    values after mean-centring, so taking the raw minimum over all ``D`` of them
    returns floating-point noise rather than a measurement. On the ``(10, 384)``
    prototype matrices this module is mostly used with, that made the descriptor
    report ~1e-16 for every input and discriminate nothing.
    """
    values = _singular_values(features)
    # Rank after centring is bounded by both dimensions, minus one for the mean.
    supported = max(1, min(features.shape[0] - 1, features.shape[1]))
    values = values[:supported]
    largest = float(values.max())
    if largest <= 1e-12:
        return 0.0
    return float(values.min() / values.max())


def mean_norm(prototypes: torch.Tensor) -> float:
    """Mean L2 norm of the rows, before any normalization."""
    return float(prototypes.to(torch.float64).norm(dim=1).mean())


def geometry_report(prototypes: torch.Tensor) -> dict[str, float]:
    """Every descriptor in this module, as a flat JSON-serializable dict."""
    report: dict[str, float] = {
        "num_classes": int(prototypes.shape[0]),
        "feature_dim": int(prototypes.shape[1]),
        "mean_norm": mean_norm(prototypes),
        "etf_deviation": etf_deviation(prototypes),
        "effective_rank": effective_rank(prototypes),
        "participation_ratio": participation_ratio(prototypes),
        "isotropy": isotropy(prototypes),
    }
    report.update(cosine_summary(prototypes))
    return report

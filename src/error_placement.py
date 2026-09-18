"""Where alignment error lands relative to what the classifier reads.

The prototype alignment objective is ``1 - cos(sar_feature, prototype)``, which
asks only that the residual be *small*. Work on cross-model KV cache transfer
(arXiv 2608.03893) found that for a closely analogous problem this is the wrong
question: reconstruction quality did not predict downstream performance
(r = -0.20), while a measure of *where* the residual landed did (r = +0.57).
Errors falling in directions the downstream computation reads are damaging;
errors of the same magnitude in directions it ignores are not.

This module ports that idea from attention to classification. There, the
reference geometry came from the attention query matrix; here it comes from the
classifier head, which is the only thing that consumes the aligned features. A
head ``W`` of shape ``(K, D)`` reads at most ``K`` directions of the
``D``-dimensional feature space and is completely blind to the remaining
``D - K``. For DINOv3 ViT-S/16+ with ten classes that is ten directions read and
374 ignored, so there is a great deal of room for residual to hide.

The central quantity is ``residual_concentration``. It is scale-free by
construction: isotropic residual scores 1.0, residual confined to the head's
nullspace scores 0.0, and residual concentrated on the leading singular direction
scores above 1.0. Those three reference points are asserted in
``tests/test_error_placement.py``.

A note on why this is worth measuring at all. Two prototype sets can produce
identical mean cosine distance - identical loss - while placing their residual
very differently. If alignment helps by relocating error rather than by shrinking
it, then the published comparison between optical and synthetic prototypes is
confounded by a factor nobody has measured.
"""

from __future__ import annotations

import torch


def head_singular_basis(head_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Right singular vectors and singular values of a classifier head.

    ``head_weight`` has shape ``(num_classes, feature_dim)``, matching
    ``nn.Linear(feature_dim, num_classes).weight``. Returns ``(right_vectors,
    singular_values)`` where ``right_vectors`` is ``(num_classes, feature_dim)``
    with orthonormal rows spanning the subspace the head reads.

    ``torch.linalg.svd`` returns ``Vh``, already the matrix whose *rows* are the
    right singular vectors, so no transpose is applied here. Getting this
    backwards silently produces plausible numbers, which is why the convention is
    stated explicitly and pinned by a test.
    """
    if head_weight.ndim != 2:
        raise ValueError(
            f"Expected a 2-D (num_classes, feature_dim) matrix, "
            f"got shape {tuple(head_weight.shape)}."
        )
    _, singular_values, right_vectors = torch.linalg.svd(
        head_weight.to(torch.float64), full_matrices=False
    )
    return right_vectors, singular_values


def residual_concentration(
    residual: torch.Tensor,
    right_vectors: torch.Tensor,
    singular_values: torch.Tensor,
) -> float:
    """How strongly residual energy sits in directions the head reads.

    Projects the residual onto the head's right singular vectors, measures the
    energy along each, and compares a singular-value-weighted sum against the
    energy an isotropic residual of the same total size would place there:

    ``concentration = sum_i w_i p_i / (E_total / D)``

    with ``w_i = s_i^2 / sum_j s_j^2``, ``p_i`` the mean squared projection onto
    direction ``i``, ``E_total`` the mean total squared norm of the residual and
    ``D`` the feature dimension. Dividing by ``E_total / D`` - the per-direction
    energy of an isotropic residual - makes the result invariant to the overall
    scale of the residual, so it measures placement rather than magnitude.

    The denominator deliberately spans the whole feature space rather than only
    the ``K`` directions the head reads. Normalizing by the mean over the read
    directions alone is tempting and wrong: for residual lying entirely in the
    nullspace every read-direction energy is zero, the ratio becomes 0/0, and
    floating-point noise returns a value near 1.0 for a case whose true answer is
    0.0. That produces a plausible number rather than an error, so the convention
    is pinned by ``test_nullspace_residual_scores_zero``.

    Above 1.0 means error concentrates where the head is most sensitive; below 1.0
    means it has settled into directions the head weights less or cannot see.
    Residual lying entirely outside the read subspace gives 0.0.
    """
    if residual.ndim != 2:
        raise ValueError(
            f"Expected a 2-D (n_samples, feature_dim) residual, "
            f"got shape {tuple(residual.shape)}."
        )
    if residual.shape[1] != right_vectors.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: residual has {residual.shape[1]}, "
            f"basis has {right_vectors.shape[1]}."
        )

    residual = residual.to(torch.float64)
    right_vectors = right_vectors.to(torch.float64)
    singular_values = singular_values.to(torch.float64)

    feature_dim = residual.shape[1]
    isotropic_per_direction = float((residual**2).sum(dim=1).mean()) / feature_dim
    if isotropic_per_direction <= 1e-300:
        return 0.0

    components = residual @ right_vectors.T
    energy_per_direction = (components**2).mean(dim=0)

    weights = singular_values**2 / (singular_values**2).sum()
    weighted = float((weights * energy_per_direction).sum())
    return weighted / isotropic_per_direction


def nullspace_energy_fraction(
    residual: torch.Tensor,
    head_weight: torch.Tensor,
) -> float:
    """Fraction of residual energy the classifier cannot see at all.

    For a ``(K, D)`` head with ``K < D`` the nullspace has dimension ``D - K``.
    Residual there changes no logit, so a high fraction means most of the
    alignment error is downstream-invisible regardless of its size.
    """
    right_vectors, _ = head_singular_basis(head_weight)
    residual = residual.to(torch.float64)

    total_energy = float((residual**2).sum())
    if total_energy <= 1e-300:
        return 0.0

    read_energy = float(((residual @ right_vectors.T) ** 2).sum())
    return max(0.0, 1.0 - read_energy / total_energy)


def logit_perturbation(
    residual: torch.Tensor,
    head_weight: torch.Tensor,
) -> dict[str, float]:
    """Effect of the residual on the logits it actually produces.

    The ground truth the concentration measure is a proxy for: applying the head
    to the residual gives exactly the logit displacement it causes. Reported
    alongside concentration so the proxy can be checked against the thing it
    approximates.
    """
    residual = residual.to(torch.float64)
    displacement = residual @ head_weight.to(torch.float64).T
    return {
        "logit_shift_mean_abs": float(displacement.abs().mean()),
        "logit_shift_max_abs": float(displacement.abs().max()),
        "logit_shift_rms": float(displacement.pow(2).mean().sqrt()),
    }


def alignment_residual(
    features: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Residual the alignment loss is trying to drive to zero.

    Both inputs are L2-normalized first, matching ``DINOClassifier.forward``
    (which normalizes its features) and the prototype builders (which normalize
    their output). The result is ``feature - prototype[label]`` per sample, whose
    squared norm is ``2 - 2 cos``, a monotone function of the cosine distance the
    training objective minimizes.
    """
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"features has {features.shape[0]} rows but labels has {labels.shape[0]}."
        )
    normalized_features = torch.nn.functional.normalize(features.to(torch.float64), dim=1)
    normalized_prototypes = torch.nn.functional.normalize(
        prototypes.to(torch.float64), dim=1
    )
    return normalized_features - normalized_prototypes[labels]


def placement_report(
    residual: torch.Tensor,
    head_weight: torch.Tensor,
) -> dict[str, float]:
    """Every placement diagnostic, as a flat JSON-serializable dict.

    Includes the residual's magnitude so that placement and magnitude can be read
    side by side; the point of the analysis is that these vary independently.
    """
    right_vectors, singular_values = head_singular_basis(head_weight)
    report = {
        "concentration": residual_concentration(residual, right_vectors, singular_values),
        "nullspace_fraction": nullspace_energy_fraction(residual, head_weight),
        "residual_rms": float(residual.to(torch.float64).pow(2).mean().sqrt()),
        "residual_mean_norm": float(residual.to(torch.float64).norm(dim=1).mean()),
    }
    report.update(logit_perturbation(residual, head_weight))
    return report

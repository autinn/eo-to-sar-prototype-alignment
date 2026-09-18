r"""Prototype variants spanning pure geometry to real optical structure.

The published work compares two points: class means of real optical features, and
a simplex equiangular tight frame. Those two differ in several respects at once,
so a gap between them does not identify which property matters. This module
builds the intermediate variants that separate those properties.

Each variant holds some properties of the source prototypes fixed and destroys
others:

=====================  =================  ================  ==================
variant                pairwise angles    class identity    optical origin
=====================  =================  ================  ==================
``etf``                uniform            none              none
``gaussian``           random             none              none
``cosine_matched``     preserved          none              none
``rotated``            preserved          preserved         obscured
``label_permuted``     preserved          destroyed         preserved
``mean_shuffled``      preserved*         destroyed         partial
``centered_shuffled``  destroyed          destroyed         partial
source prototypes      preserved          preserved         preserved
=====================  =================  ================  ==================

\* ``mean_shuffled`` preserves angles almost exactly, which was not the original
intent; see its docstring. ``centered_shuffled`` is the variant that actually
decorrelates the prototypes.

``rotated`` is the variant the published control cannot express: it keeps the
exact angular geometry of the real optical prototypes while moving them to an
arbitrary orientation in feature space. If alignment gains survive rotation, the
gain comes from geometry; if they vanish, something about the specific directions
matters.

All functions return ``(K, D)`` float32 tensors with L2-normalized rows, matching
what ``build_eo_prototypes`` and ``build_synthetic_prototypes`` produce in the
training scripts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _random_orthonormal_basis(
    feature_dim: int,
    num_vectors: int,
    seed: int,
) -> torch.Tensor:
    """A ``(feature_dim, num_vectors)`` matrix with orthonormal columns."""
    generator = torch.Generator().manual_seed(seed)
    random_matrix = torch.randn(feature_dim, num_vectors, generator=generator)
    basis, _ = torch.linalg.qr(random_matrix, mode="reduced")
    return basis


def etf_prototypes(num_classes: int, feature_dim: int, seed: int) -> torch.Tensor:
    """Simplex equiangular tight frame embedded in ``feature_dim`` dimensions.

    Reimplements ``build_synthetic_prototypes`` from
    ``train_synthetic_prototype_alignment.py``, which is defined inside a script
    and so cannot be imported. ``tests/test_prototype_variants.py`` asserts the
    two agree.

    Every pair of prototypes has cosine exactly ``-1/(num_classes - 1)``; the QR
    rotation is an isometry and so does not disturb that.
    """
    if feature_dim < num_classes:
        raise ValueError(
            "The ETF construction requires feature_dim >= num_classes, "
            f"but received {feature_dim} < {num_classes}."
        )

    centered_identity = (
        torch.eye(num_classes) - torch.ones(num_classes, num_classes) / num_classes
    )
    simplex = F.normalize(centered_identity, dim=1)
    basis = _random_orthonormal_basis(feature_dim, num_classes, seed)
    return F.normalize(simplex @ basis.T, dim=1)


def gaussian_prototypes(num_classes: int, feature_dim: int, seed: int) -> torch.Tensor:
    """Independent Gaussian directions, normalized.

    In high dimensions these are near-orthogonal, so this sits between the ETF
    (exactly equiangular, maximally separated) and real class means (correlated).
    """
    generator = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(num_classes, feature_dim, generator=generator), dim=1)


def rotated(prototypes: torch.Tensor, seed: int) -> torch.Tensor:
    """Apply a random rotation, preserving all pairwise angles exactly.

    The Gram matrix is invariant under this transform, so any geometric
    descriptor computed from angles alone is unchanged while the prototypes
    themselves point in entirely different directions.
    """
    feature_dim = prototypes.shape[1]
    basis = _random_orthonormal_basis(feature_dim, feature_dim, seed)
    return F.normalize(prototypes.to(torch.float32) @ basis.T, dim=1)


def label_permuted(prototypes: torch.Tensor, seed: int) -> torch.Tensor:
    """Reassign prototypes to classes, keeping the set of prototypes intact.

    The geometry is exactly the real optical geometry; only the correspondence
    between a class and its prototype is broken. This isolates whether the
    class-to-prototype assignment carries the information, as opposed to the
    shape of the prototype constellation.
    """
    num_classes = prototypes.shape[0]
    generator = torch.Generator().manual_seed(seed)

    # Reject the identity so the variant always differs from its source.
    for _ in range(100):
        permutation = torch.randperm(num_classes, generator=generator)
        if not torch.equal(permutation, torch.arange(num_classes)):
            break
    else:  # pragma: no cover - needs num_classes < 2 to trigger
        raise RuntimeError("Could not draw a non-identity permutation.")

    return prototypes[permutation].clone()


def cosine_matched(prototypes: torch.Tensor, seed: int) -> torch.Tensor:
    """Synthetic prototypes reproducing the source Gram matrix.

    Factorizes the source Gram matrix and re-embeds it against a random
    orthonormal basis, giving prototypes with the same pairwise-angle structure
    but no dependence on the original feature directions. Unlike ``rotated``
    these are built from the Gram matrix alone, so they carry no optical
    information beyond the angles themselves.
    """
    normalized = F.normalize(prototypes.to(torch.float64), dim=1)
    gram = normalized @ normalized.T

    # Symmetric factorization; clamp guards against small negative eigenvalues.
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    eigenvalues = eigenvalues.clamp_min(0.0)
    factor = eigenvectors @ torch.diag(eigenvalues.sqrt())

    num_classes, feature_dim = prototypes.shape
    basis = _random_orthonormal_basis(feature_dim, num_classes, seed).to(torch.float64)
    return F.normalize(factor @ basis.T, dim=1).to(torch.float32)


def mean_shuffled(prototypes: torch.Tensor, seed: int) -> torch.Tensor:
    """Shuffle each feature dimension independently across classes.

    Preserves the marginal distribution of every coordinate exactly, and breaks
    which class holds which value within a coordinate.

    Note on what this does *not* do. It was originally included to destroy
    angular structure, but it barely changes pairwise cosines at all - measured
    shifts are under 0.001 even for strongly correlated prototypes. The reason is
    that correlation between class means is carried by the shared component of
    each coordinate, and permuting values within a coordinate leaves that
    component untouched. Use ``centered_shuffled`` to remove the shared component,
    or ``gaussian_prototypes`` for an unstructured reference.

    It is kept because "preserves marginals, permutes assignment" is a
    well-defined control in its own right, and because its near-zero effect on
    angles is a useful negative result: it shows that per-coordinate marginals do
    not determine the angular geometry.
    """
    generator = torch.Generator().manual_seed(seed)
    shuffled = prototypes.to(torch.float32).clone()
    num_classes = shuffled.shape[0]
    for dimension in range(shuffled.shape[1]):
        shuffled[:, dimension] = shuffled[
            torch.randperm(num_classes, generator=generator), dimension
        ]
    return F.normalize(shuffled, dim=1)


def centered_shuffled(prototypes: torch.Tensor, seed: int) -> torch.Tensor:
    """Remove the shared mean direction, then shuffle each coordinate.

    Class means from a single encoder share a large common component, which is
    what makes their pairwise cosines high. Subtracting the mean prototype before
    shuffling removes that component, so this variant genuinely decorrelates the
    prototypes rather than only permuting them.
    """
    generator = torch.Generator().manual_seed(seed)
    centered = prototypes.to(torch.float32) - prototypes.to(torch.float32).mean(
        dim=0, keepdim=True
    )
    num_classes = centered.shape[0]
    for dimension in range(centered.shape[1]):
        centered[:, dimension] = centered[
            torch.randperm(num_classes, generator=generator), dimension
        ]
    return F.normalize(centered, dim=1)


def interpolate(
    prototypes_a: torch.Tensor,
    prototypes_b: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Spherical-ish interpolation between two prototype sets.

    ``alpha = 0`` returns ``prototypes_a`` and ``alpha = 1`` returns
    ``prototypes_b``, both up to normalization. Sweeping alpha traces a path
    between two conditions, which is how a continuum between a weak and a strong
    form of alignment can be tested rather than only the two endpoints.
    """
    if prototypes_a.shape != prototypes_b.shape:
        raise ValueError(
            f"Shape mismatch: {tuple(prototypes_a.shape)} vs {tuple(prototypes_b.shape)}."
        )
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}.")

    a = F.normalize(prototypes_a.to(torch.float32), dim=1)
    b = F.normalize(prototypes_b.to(torch.float32), dim=1)
    return F.normalize((1.0 - alpha) * a + alpha * b, dim=1)


# Variants that need only a shape, versus those that transform existing prototypes.
CONSTRUCTORS = {
    "etf": etf_prototypes,
    "gaussian": gaussian_prototypes,
}

TRANSFORMS = {
    "rotated": rotated,
    "label_permuted": label_permuted,
    "cosine_matched": cosine_matched,
    "mean_shuffled": mean_shuffled,
    "centered_shuffled": centered_shuffled,
}


def build_ladder(
    source_prototypes: torch.Tensor,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Every variant plus the source, keyed by name.

    The source prototypes are included under ``"source"`` so a report over this
    dict compares like with like.
    """
    num_classes, feature_dim = source_prototypes.shape
    ladder: dict[str, torch.Tensor] = {
        "source": F.normalize(source_prototypes.to(torch.float32), dim=1),
    }
    for name, constructor in CONSTRUCTORS.items():
        ladder[name] = constructor(num_classes, feature_dim, seed)
    for name, transform in TRANSFORMS.items():
        ladder[name] = transform(source_prototypes, seed)
    return ladder

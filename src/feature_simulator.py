"""Synthetic feature clouds with controllable geometry.

The stub backbone proves the pipeline connects, but its feature geometry is
whatever a randomly initialised network happens to produce. This module goes the
other way: it generates features whose class separation, intrinsic
dimensionality, covariance structure and cross-modality gap are all specified in
advance, so a measurement can be checked against a known answer.

That matters for the questions this repository is built around. Asking whether
prototype alignment helps because of optical semantics or because of where it
places residual error is hard to settle on real data, where the two are
confounded and neither is observable. In simulation they can be varied
independently: hold class separation fixed and change the modality gap, or hold
the gap fixed and change how much of the signal lies in the classifier's read
subspace.

Nothing produced here is evidence about SAR. Simulation establishes what a method
does under conditions you control, which is a precondition for interpreting what
it does on data you do not.

Typical use::

    config = SimulatorConfig(num_classes=10, separation=2.0, imbalance_ratio=100.0)
    eo_features, sar_features, labels = simulate_paired_modalities(config, modality_gap=1.0)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SimulatorConfig:
    """Parameters controlling a synthetic feature cloud.

    Attributes:
        num_classes: Number of classes.
        feature_dim: Dimensionality, matching DINOv3 ViT-S/16+ at 384.
        n_per_class: Samples in the largest class.
        separation: Distance between class means relative to within-class spread.
            Below about 1.0 the classes overlap heavily; above 3.0 they are
            almost perfectly separable.
        intrinsic_dim: Dimensions the class means span. ``None`` uses
            ``num_classes - 1``, the dimensionality real class means occupy after
            centering. Lower values crowd the means into a smaller subspace.
        within_class_cov_scale: Multiplier on within-class noise.
        anisotropy: Spectral decay of the within-class covariance. ``0.0`` gives
            isotropic noise; larger values concentrate variance in a few
            directions, as real feature clouds do.
        imbalance_ratio: Ratio between the largest and smallest class. UNICORNv2
            is roughly 1000:1.
        seed: Seed for reproducibility.
    """

    num_classes: int = 10
    feature_dim: int = 384
    n_per_class: int = 200
    separation: float = 1.0
    intrinsic_dim: int | None = None
    within_class_cov_scale: float = 1.0
    anisotropy: float = 0.0
    imbalance_ratio: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.num_classes < 2:
            raise ValueError("At least two classes are required.")
        if self.feature_dim < self.num_classes:
            raise ValueError("feature_dim must be at least num_classes.")
        if self.n_per_class < 1:
            raise ValueError("n_per_class must be positive.")
        if self.imbalance_ratio < 1.0:
            raise ValueError("imbalance_ratio must be at least 1.0.")
        if self.intrinsic_dim is not None and not (
            1 <= self.intrinsic_dim <= self.feature_dim
        ):
            raise ValueError("intrinsic_dim must lie between 1 and feature_dim.")


def class_counts(config: SimulatorConfig) -> list[int]:
    """Samples per class, following a geometric imbalance if requested."""
    if config.imbalance_ratio == 1.0:
        return [config.n_per_class] * config.num_classes

    ratios = torch.logspace(
        0.0, -torch.log10(torch.tensor(config.imbalance_ratio)).item(), config.num_classes
    )
    return [max(1, int(round(config.n_per_class * float(ratio)))) for ratio in ratios]


def _class_means(config: SimulatorConfig, generator: torch.Generator) -> torch.Tensor:
    """Class means spanning ``intrinsic_dim`` dimensions.

    Drawn in a low-dimensional subspace and then embedded, so the effective rank
    of the resulting means is controlled rather than incidental.
    """
    intrinsic = (
        config.intrinsic_dim
        if config.intrinsic_dim is not None
        else config.num_classes - 1
    )

    low_dimensional = torch.randn(
        config.num_classes, intrinsic, generator=generator
    )
    embedding = torch.randn(intrinsic, config.feature_dim, generator=generator)
    embedding, _ = torch.linalg.qr(embedding.T, mode="reduced")

    means = low_dimensional @ embedding.T
    return config.separation * F.normalize(means, dim=1)


def _covariance_scales(config: SimulatorConfig) -> torch.Tensor:
    """Per-dimension standard deviations for the within-class noise."""
    if config.anisotropy <= 0.0:
        return torch.ones(config.feature_dim)
    decay = torch.linspace(0.0, 1.0, config.feature_dim)
    return torch.exp(-config.anisotropy * decay)


def simulate_features(
    config: SimulatorConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate a labelled feature cloud.

    Returns ``(features, labels, class_means)``. Features are *not* normalized,
    matching ``extract_features``, which returns raw backbone output; the
    prototype builders normalize afterwards.
    """
    generator = torch.Generator().manual_seed(config.seed)
    means = _class_means(config, generator)
    scales = _covariance_scales(config) * config.within_class_cov_scale
    counts = class_counts(config)

    feature_blocks, label_blocks = [], []
    for class_index, count in enumerate(counts):
        noise = torch.randn(count, config.feature_dim, generator=generator) * scales
        feature_blocks.append(means[class_index] + noise)
        label_blocks.append(torch.full((count,), class_index, dtype=torch.long))

    return torch.cat(feature_blocks), torch.cat(label_blocks), means


def simulate_paired_modalities(
    config: SimulatorConfig,
    modality_gap: float = 1.0,
    shared_structure: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate EO and SAR features sharing class structure but not geometry.

    The setting the alignment method addresses: two modalities describing the
    same classes in representations that were never trained to agree.

    Args:
        config: Geometry of the underlying cloud.
        modality_gap: How far SAR class means are displaced from EO ones. Zero
            makes the modalities identical; large values make optical prototypes
            poor targets for SAR features.
        shared_structure: How much of the EO class geometry carries over to SAR.
            ``1.0`` keeps the same relative arrangement and displaces it;
            ``0.0`` gives SAR an unrelated arrangement, so optical prototypes
            carry no usable information.

    Returns ``(eo_features, sar_features, labels, eo_class_means)``.
    """
    if not 0.0 <= shared_structure <= 1.0:
        raise ValueError("shared_structure must lie in [0, 1].")

    eo_features, labels, eo_means = simulate_features(config)

    generator = torch.Generator().manual_seed(config.seed + 10_000)
    independent_means = _class_means(config, generator)

    # Interpolate between the EO arrangement and an unrelated one, then displace.
    sar_means = shared_structure * eo_means + (1.0 - shared_structure) * independent_means
    displacement = torch.randn(1, config.feature_dim, generator=generator)
    sar_means = sar_means + modality_gap * F.normalize(displacement, dim=1)

    scales = _covariance_scales(config) * config.within_class_cov_scale
    counts = class_counts(config)

    feature_blocks = []
    for class_index, count in enumerate(counts):
        noise = torch.randn(count, config.feature_dim, generator=generator) * scales
        feature_blocks.append(sar_means[class_index] + noise)

    return eo_features, torch.cat(feature_blocks), labels, eo_means


def class_means_from_features(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Normalized class means, matching ``build_eo_prototypes``.

    Averages raw features by class and normalizes afterwards, which is the order
    the published results used.
    """
    # A class with no samples would average an empty selection to NaN, which
    # F.normalize propagates and the alignment loss then spreads to every batch
    # containing that label - silently, since NaN does not raise. This is
    # reachable on the real pipeline: run_variant_experiment takes the class
    # count from the SAR dataset while the features come from the EO dataset,
    # so any class present in one and absent from the other lands here.
    populated = torch.bincount(labels.to(torch.long), minlength=num_classes)
    empty = (populated == 0).nonzero(as_tuple=True)[0].tolist()
    if empty:
        raise ValueError(
            f"No samples for class indices {empty} out of {num_classes}. "
            "Prototypes for these classes would be NaN. Check that the feature "
            "set covers every class, and that num_classes matches the dataset "
            "the features came from."
        )

    means = torch.stack(
        [features[labels == class_index].mean(dim=0) for class_index in range(num_classes)]
    )
    return F.normalize(means, dim=1)


def linear_probe_accuracy(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    epochs: int = 100,
    learning_rate: float = 0.1,
) -> float:
    """Training accuracy of a linear classifier, as a separability measure.

    Used to check that a configuration produces the difficulty intended - a
    simulation where every class is trivially separable tests nothing.
    """
    torch.manual_seed(0)
    head = torch.nn.Linear(features.shape[1], num_classes)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    normalized = F.normalize(features, dim=1)

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.cross_entropy(head(normalized), labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predictions = head(normalized).argmax(dim=1)
    return float((predictions == labels).float().mean())

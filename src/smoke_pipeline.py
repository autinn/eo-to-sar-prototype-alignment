"""End-to-end pipeline check using the stub backbone and fake data.

Runs every stage the real experiments depend on - data loading, feature
extraction, prototype construction, LoRA fine-tuning, evaluation, and the
geometry and error-placement analyses - without DINOv3 weights or UNICORNv2.
Its purpose is to prove the code paths connect; the accuracies it prints are
meaningless and should be ignored.

Prerequisite::

    python src/make_fake_unicorn.py --per-class 10

Then::

    python src/smoke_pipeline.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from error_placement import alignment_residual, placement_report
from evaluation import evaluate_classifier, seven_class_metrics
from geometry import geometry_report
from model_utils import DINOClassifier, apply_lora, extract_features, make_dino_transform
from prototype_variants import build_ladder
from stub_backbone import load_stub_model
from utils import seed_everything


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data" / "fake"

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 2
LEARNING_RATE = 1e-4
ALIGNMENT_WEIGHT = 0.6
ALIGNMENT_LOSS_SCALE = 2.0


def build_eo_prototypes(
    eo_features: torch.Tensor,
    eo_labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Class means of raw EO features, then normalized.

    Mirrors ``build_eo_prototypes`` in ``train_eo_prototype_alignment.py``, which
    lives inside a script and cannot be imported. Mean-then-normalize is the
    order the published results used.
    """
    prototypes = torch.stack(
        [
            eo_features[eo_labels == class_index].mean(dim=0)
            for class_index in range(num_classes)
        ]
    )
    return F.normalize(prototypes, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    train_sar_dir = arguments.data / "train" / "SAR_Train"
    train_eo_dir = arguments.data / "train" / "EO_Train"
    test_sar_dir = arguments.data / "test" / "IID"

    for path in (train_sar_dir, train_eo_dir, test_sar_dir):
        if not path.is_dir():
            raise FileNotFoundError(
                f"Missing {path}. Run: python src/make_fake_unicorn.py --per-class 10"
            )

    seed_everything(arguments.seed)
    device = torch.device("cpu")
    transform = make_dino_transform(IMAGE_SIZE)

    print("=" * 70)
    print("Stage 1: load data")
    print("=" * 70)
    train_sar = ImageFolder(train_sar_dir, transform=transform)
    train_eo = ImageFolder(train_eo_dir, transform=transform)
    test_sar = ImageFolder(test_sar_dir, transform=transform)
    class_names = list(train_sar.classes)
    print(f"  {len(train_sar)} SAR train, {len(train_eo)} EO train, {len(test_sar)} test")
    print(f"  {len(class_names)} classes: {', '.join(class_names[:4])}, ...")

    print()
    print("=" * 70)
    print("Stage 2: extract EO features and build prototypes")
    print("=" * 70)
    eo_loader = DataLoader(train_eo, batch_size=BATCH_SIZE, shuffle=False)
    eo_backbone = load_stub_model().to(device)
    eo_features, eo_labels = extract_features(eo_backbone, eo_loader, device)
    print(f"  EO features: {tuple(eo_features.shape)}")

    prototypes = build_eo_prototypes(eo_features, eo_labels, len(class_names))
    print(f"  prototypes:  {tuple(prototypes.shape)}")
    del eo_backbone, eo_loader

    print()
    print("=" * 70)
    print("Stage 3: prototype geometry across the ablation ladder")
    print("=" * 70)
    ladder = build_ladder(prototypes, seed=arguments.seed)
    header = f"  {'variant':<20}{'cos_mean':>10}{'cos_range':>11}{'eff_rank':>10}{'etf_dev':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, variant in ladder.items():
        report = geometry_report(variant)
        print(
            f"  {name:<20}{report['cosine_mean']:10.4f}{report['cosine_range']:11.4f}"
            f"{report['effective_rank']:10.2f}{report['etf_deviation']:10.4f}"
        )

    print()
    print("=" * 70)
    print("Stage 4: LoRA fine-tuning with prototype alignment")
    print("=" * 70)
    model = DINOClassifier(load_stub_model(), num_classes=len(class_names))
    model = apply_lora(
        model, rank=8, alpha=16, target_modules=["qkv"], dropout=0.05
    ).to(device)

    train_loader = DataLoader(train_sar, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_sar, batch_size=BATCH_SIZE, shuffle=False)
    classification_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    device_prototypes = prototypes.to(device)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            logits, features = model(images.to(device))
            labels = labels.to(device)

            cosine_distance = 1 - F.cosine_similarity(
                features, device_prototypes[labels], dim=1
            )
            loss = (1 - ALIGNMENT_WEIGHT) * classification_loss(logits, labels) + (
                ALIGNMENT_WEIGHT * cosine_distance.mean() * ALIGNMENT_LOSS_SCALE
            )
            loss.backward()
            optimizer.step()
            total += float(loss)

        metrics, labels_out, predictions = evaluate_classifier(
            model, test_loader, device, num_classes=len(class_names)
        )
        merged = seven_class_metrics(labels_out, predictions, class_names)
        print(
            f"  epoch {epoch}  loss {total / len(train_loader):.4f}  "
            f"acc {metrics['accuracy']:.3f}  macro-F1 {metrics['macro_f1']:.3f}  "
            f"acc7 {merged['accuracy_7class']:.3f}"
        )

    print()
    print("=" * 70)
    print("Stage 5: error placement of the alignment residual")
    print("=" * 70)
    model.eval()
    with torch.inference_mode():
        collected_features, collected_labels = [], []
        for images, labels in test_loader:
            _, features = model(images.to(device))
            collected_features.append(features.cpu())
            collected_labels.append(labels)
    test_features = torch.cat(collected_features)
    test_labels = torch.cat(collected_labels)

    # The head is what consumes the aligned features, so it defines the geometry.
    head_weight = model.base_model.model.head.modules_to_save.default.weight.detach()

    print(f"  {'variant':<20}{'concentration':>15}{'nullspace':>12}{'resid_rms':>12}")
    print("  " + "-" * 57)
    for name in ("source", "etf", "rotated", "label_permuted"):
        residual = alignment_residual(test_features, ladder[name], test_labels)
        report = placement_report(residual, head_weight)
        print(
            f"  {name:<20}{report['concentration']:15.3f}"
            f"{report['nullspace_fraction']:12.3f}{report['residual_rms']:12.4f}"
        )

    print()
    print("=" * 70)
    print("Pipeline completed. Accuracies above are meaningless - the backbone is")
    print("a stand-in and the data is synthetic. What this shows is that every")
    print("stage connects, so swapping load_stub_model for load_dino_model is the")
    print("only change needed once DINOv3 access is granted.")
    print("=" * 70)


if __name__ == "__main__":
    main()

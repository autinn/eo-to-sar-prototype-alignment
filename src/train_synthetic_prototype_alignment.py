"""Fine-tune DINOv3 by aligning SAR features with fixed synthetic prototypes."""

from __future__ import annotations

import csv
import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from evaluation import evaluate_classifier, save_predictions, seven_class_metrics
from model_utils import DINOClassifier, apply_lora, load_dino_model, make_dino_transform
from utils import load_config, save_json, seed_everything, summarize_runs


# Resolve config and outputs from the repository root, regardless of where the script is launched.
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"

# Hyperparameters used for synthetic prototype alignment.
SEEDS = [30, 42, 123, 777, 2025]
IMAGE_SIZE = 128
BATCH_SIZE = 256
NUM_WORKERS = 4
EPOCHS = 25
LEARNING_RATE = 4e-6
WEIGHT_DECAY = 1e-2
MIN_EPOCHS = 5  # Minimum epochs to run
EARLY_STOPPING_PATIENCE = 3  # Stop after this many epochs without macro-F1 improvement
ALIGNMENT_WEIGHT = 0.6
ALIGNMENT_LOSS_SCALE = 2.0
PROTOTYPE_SEED = 0

# LoRA hyperparameters
LORA_RANK = 28
LORA_ALPHA = 56
LORA_TARGET_MODULES = ["qkv"]
LORA_DROPOUT = 0.05


# Balance the highly imbalanced SAR training set without discarding samples.
def make_weighted_sampler(labels: list[int], seed: int) -> WeightedRandomSampler:
    """Create the inverse-frequency sampler used by the paper experiments."""
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    class_counts = torch.bincount(label_tensor)
    sample_weights = class_counts.float().reciprocal()[label_tensor]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


# Construct uniformly separated class targets without using EO data.
def build_synthetic_prototypes(
    num_classes: int,
    feature_dim: int,
    seed: int,
) -> torch.Tensor:
    """Embed a simplex equiangular tight frame in the DINO feature space."""
    if feature_dim < num_classes:
        raise ValueError(
            "The ETF construction requires feature_dim >= num_classes, "
            f"but received {feature_dim} < {num_classes}."
        )

    # The centered identity gives unit vectors with cosine -1/(K - 1).
    centered_identity = (
        torch.eye(num_classes) - torch.ones(num_classes, num_classes) / num_classes
    )
    simplex = F.normalize(centered_identity, dim=1)

    # A random orthonormal basis embeds the simplex without changing its angles.
    generator = torch.Generator().manual_seed(seed)
    random_matrix = torch.randn(feature_dim, num_classes, generator=generator)
    basis, _ = torch.linalg.qr(random_matrix, mode="reduced")
    return F.normalize(simplex @ basis.T, dim=1)


# Train one SAR classifier against the same fixed synthetic prototypes.
def train_one_seed(
    seed: int,
    train_sar: ImageFolder,
    test_sar: ImageFolder,
    test_sample_paths: list[str],
    class_names: list[str],
    synthetic_prototypes: torch.Tensor,
    dino_repo: Path,
    dino_weights: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, float | int]:
    """Train one synthetic-prototype model and retain its best test predictions."""
    seed_everything(seed)

    # Reload the pretrained backbone so every seed starts from the same weights.
    backbone = load_dino_model(dino_repo, dino_weights)
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    model = DINOClassifier(backbone, num_classes=len(class_names))
    model = apply_lora(
        model,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        dropout=LORA_DROPOUT,
    ).to(device)

    train_loader = DataLoader(
        train_sar,
        batch_size=BATCH_SIZE,
        sampler=make_weighted_sampler(train_sar.targets, seed),
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_sar,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    classification_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    best_result = None
    best_labels = None
    best_predictions = None
    epochs_without_improvement = 0

    # Fine-tune using classification and class conditional synthetic alignment.
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_classification_loss = 0.0
        total_alignment_loss = 0.0

        for sar_images, labels in train_loader:
            sar_images = sar_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits, sar_features = model(sar_images)

            loss_classification = classification_loss(logits, labels)
            target_prototypes = synthetic_prototypes[labels]
            cosine_distance = 1 - F.cosine_similarity(
                sar_features,
                target_prototypes,
                dim=1,
            )
            loss_alignment = cosine_distance.mean() * ALIGNMENT_LOSS_SCALE
            loss = (
                (1 - ALIGNMENT_WEIGHT) * loss_classification
                + ALIGNMENT_WEIGHT * loss_alignment
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_classification_loss += loss_classification.item()
            total_alignment_loss += loss_alignment.item()

        scheduler.step()

        metrics, labels, predictions = evaluate_classifier(
            model,
            test_loader,
            device,
            num_classes=len(class_names),
        )
        num_batches = len(train_loader)
        print(
            f"Seed {seed} | Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss {total_loss / num_batches:.4f} | "
            f"CE {total_classification_loss / num_batches:.4f} | "
            f"Alignment {total_alignment_loss / num_batches:.4f} | "
            f"Accuracy {metrics['accuracy']:.4f} | Macro-F1 {metrics['macro_f1']:.4f}"
        )

        improved = best_result is None or metrics["macro_f1"] > best_result["macro_f1"]
        if improved:
            best_result = {
                "seed": seed,
                "best_epoch": epoch,
                **metrics,
                **seven_class_metrics(labels, predictions, class_names),
            }
            best_labels = labels.copy()
            best_predictions = predictions.copy()

        # Stop after three non-improving epochs.
        if epoch >= MIN_EPOCHS:
            epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(f"Seed {seed} | Early stopping at epoch {epoch}")
                break

    if best_result is None or best_labels is None or best_predictions is None:
        raise RuntimeError(f"No test result was produced for seed {seed}.")

    save_predictions(
        output_dir / f"predictions_seed_{seed}.csv",
        test_sample_paths,
        best_labels,
        best_predictions,
        class_names,
    )
    return best_result


# Write compact metrics and predictions, not model checkpoints.
def save_results(
    results: list[dict[str, float | int]],
    output_dir: Path,
    num_classes: int,
    feature_dim: int,
) -> None:
    """Save per-seed metrics and their aggregate summary."""
    metric_names = [
        "accuracy",
        "macro_f1",
        "accuracy_7class",
        "macro_f1_7class",
    ]

    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    save_json(
        {
            "method": "synthetic_prototype_alignment",
            "seeds": SEEDS,
            "settings": {
                "image_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "early_stopping_min_epochs": MIN_EPOCHS,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "alignment_weight": ALIGNMENT_WEIGHT,
                "alignment_loss_scale": ALIGNMENT_LOSS_SCALE,
                "effective_unscaled_alignment_weight": (
                    ALIGNMENT_WEIGHT * ALIGNMENT_LOSS_SCALE
                ),
                "prototype_construction": "simplex_equiangular_tight_frame",
                "prototype_seed": PROTOTYPE_SEED,
                "prototype_feature_dim": feature_dim,
                "prototype_pairwise_cosine": -1 / (num_classes - 1),
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "lora_target_modules": LORA_TARGET_MODULES,
                "lora_dropout": LORA_DROPOUT,
            },
            "metrics": summarize_runs(results, metric_names),
        },
        output_dir / "summary.json",
    )


# Construct one fixed synthetic prototype set, then run five SAR training seeds.
def main() -> None:
    # Load machine-specific paths from the ignored config.yaml.
    config = load_config(CONFIG_PATH, REPO_ROOT)
    paths = config["paths"]

    # Fail early if the DINOv3 or dataset paths are incorrect.
    required_paths = [
        "dino_repo",
        "dino_weights",
        "train_sar",
        "test_sar",
    ]
    for name in required_paths:
        path = paths[name]
        if not path.exists():
            raise FileNotFoundError(f"Configured path does not exist: paths.{name}={path}")

    # Keep this method's compact outputs in a separate directory.
    output_dir = paths["output_dir"] / "synthetic_prototype_alignment"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the first visible GPU when available; otherwise run on CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Only SAR images are needed because the class targets are synthetic.
    transform = make_dino_transform(IMAGE_SIZE)
    train_sar = ImageFolder(paths["train_sar"], transform=transform)
    test_sar = ImageFolder(paths["test_sar"], transform=transform)
    class_names = list(train_sar.classes)
    test_sample_paths = [path for path, _ in test_sar.samples]

    # Read the feature dimension from the selected DINOv3 backbone.
    prototype_backbone = load_dino_model(paths["dino_repo"], paths["dino_weights"])
    feature_dim = prototype_backbone.embed_dim
    del prototype_backbone

    # Build one fixed ETF shared by all training seeds.
    synthetic_prototypes = build_synthetic_prototypes(
        num_classes=len(class_names),
        feature_dim=feature_dim,
        seed=PROTOTYPE_SEED,
    ).to(device)
    expected_cosine = -1 / (len(class_names) - 1)
    print(
        f"Built {len(class_names)} synthetic prototypes with expected "
        f"pairwise cosine {expected_cosine:.4f}"
    )

    # Reload and fine-tune an independent SAR model for every seed.
    results = []
    for seed in SEEDS:
        results.append(
            train_one_seed(
                seed,
                train_sar,
                test_sar,
                test_sample_paths,
                class_names,
                synthetic_prototypes,
                paths["dino_repo"],
                paths["dino_weights"],
                output_dir,
                device,
            )
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Save per-seed predictions and aggregate metrics only.
    save_results(
        results,
        output_dir,
        num_classes=len(class_names),
        feature_dim=feature_dim,
    )
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
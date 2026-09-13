"""Fine-tune DINOv3 by aligning SAR features with fixed EO class prototypes."""

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
from model_utils import (
    DINOClassifier,
    apply_lora,
    extract_features,
    load_dino_model,
    make_dino_transform,
)
from utils import load_config, save_json, seed_everything, summarize_runs


# Resolve config and outputs from the repository root, regardless of where the script is launched.
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"

# Hyperparameters used for EO prototype alignment.
SEEDS = [10, 40, 41, 42, 43]
IMAGE_SIZE = 128
BATCH_SIZE = 256
FEATURE_BATCH_SIZE = 256
NUM_WORKERS = 4
EPOCHS = 25
LEARNING_RATE = 8e-6
WEIGHT_DECAY = 1e-2
MIN_EPOCHS = 5  # Minimum epochs to run
EARLY_STOPPING_PATIENCE = 3  # Stop after this many epochs without macro-F1 improvement
ALIGNMENT_WEIGHT = 0.6
ALIGNMENT_LOSS_SCALE = 2.0

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


# Reproduce the historical prototype construction used for the reported results.
def build_eo_prototypes(
    eo_features: torch.Tensor,
    eo_labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Average raw EO features by class, then L2-normalize each class mean."""
    prototypes = torch.stack(
        [eo_features[eo_labels == class_index].mean(dim=0) for class_index in range(num_classes)]
    )
    return F.normalize(prototypes, dim=1)


# Train one SAR classifier against the same fixed EO prototypes.
def train_one_seed(
    seed: int,
    train_sar: ImageFolder,
    test_sar: ImageFolder,
    test_sample_paths: list[str],
    class_names: list[str],
    eo_prototypes: torch.Tensor,
    dino_repo: Path,
    dino_weights: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, float | int]:
    """Train one prototype-aligned model and retain its best test predictions."""
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

    # Fine-tune using classification and class conditional prototype alignment.
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
            target_prototypes = eo_prototypes[labels]
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


# Write only compact metrics and predictions, never model checkpoints.
def save_results(results: list[dict[str, float | int]], output_dir: Path) -> None:
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
            "method": "eo_prototype_alignment",
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
                "prototype_construction": "mean_raw_features_then_l2_normalize",
                "lora_rank": LORA_RANK,
                "lora_alpha": LORA_ALPHA,
                "lora_target_modules": LORA_TARGET_MODULES,
                "lora_dropout": LORA_DROPOUT,
            },
            "metrics": summarize_runs(results, metric_names),
        },
        output_dir / "summary.json",
    )


# Extract fixed EO prototypes once, then run five SAR fine-tuning seeds.
def main() -> None:
    # Load machine-specific paths from the ignored config.yaml.
    config = load_config(CONFIG_PATH, REPO_ROOT)
    paths = config["paths"]

    # Fail early if the DINOv3 or dataset paths are incorrect.
    required_paths = [
        "dino_repo",
        "dino_weights",
        "train_sar",
        "train_eo",
        "test_sar",
    ]
    for name in required_paths:
        path = paths[name]
        if not path.exists():
            raise FileNotFoundError(f"Configured path does not exist: paths.{name}={path}")

    # Keep this method's compact outputs in a separate directory.
    output_dir = paths["output_dir"] / "eo_prototype_alignment"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the first visible GPU when available; otherwise run on CPU.
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # EO and SAR images use the same deterministic DINOv3 preprocessing.
    transform = make_dino_transform(IMAGE_SIZE)
    train_sar = ImageFolder(paths["train_sar"], transform=transform)
    train_eo = ImageFolder(paths["train_eo"], transform=transform)
    test_sar = ImageFolder(paths["test_sar"], transform=transform)
    class_names = list(train_sar.classes)
    test_sample_paths = [path for path, _ in test_sar.samples]

    # Extract every EO training feature once to construct the fixed class means.
    eo_loader = DataLoader(
        train_eo,
        batch_size=FEATURE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )
    eo_backbone = load_dino_model(paths["dino_repo"], paths["dino_weights"]).to(device)
    print("Extracting EO features for class prototypes...")
    eo_features, eo_labels = extract_features(eo_backbone, eo_loader, device)
    eo_prototypes = build_eo_prototypes(
        eo_features,
        eo_labels,
        num_classes=len(class_names),
    ).to(device)

    # The EO images and backbone are no longer needed after prototype construction.
    del eo_backbone, eo_features, eo_labels, eo_loader, train_eo
    if device.type == "cuda":
        torch.cuda.empty_cache()

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
                eo_prototypes,
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
    save_results(results, output_dir)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()

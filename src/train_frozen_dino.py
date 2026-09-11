"""Train the frozen DINOv3 linear-probe baseline used in the paper."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from evaluation import evaluate_classifier, save_predictions, seven_class_metrics
from model_utils import extract_features, load_dino_model, make_dino_transform
from utils import load_config, save_json, seed_everything, summarize_runs


# Resolve config and outputs from the repository root, regardless of where the script is launched.
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"

# Hyperparameters used for the frozen DINOv3 baseline.
SEEDS = [30, 42, 123, 777, 2025]
IMAGE_SIZE = 128
BATCH_SIZE = 256
FEATURE_BATCH_SIZE = 256
NUM_WORKERS = 4
EPOCHS = 25
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4


# Frozen baseline model: only this linear layer is trained.
class LinearProbe(nn.Module):
    """L2-normalized frozen features followed by a trainable linear head."""

    def __init__(self, feature_dim: int, num_classes: int) -> None:
        super().__init__()
        self.head = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(F.normalize(features, dim=1))


# Balance the highly imbalanced training set without discarding samples.
def make_weighted_sampler(labels: torch.Tensor, seed: int) -> WeightedRandomSampler:
    """Create the inverse-frequency sampler used by the paper experiments."""
    class_counts = torch.bincount(labels)
    sample_weights = class_counts.float().reciprocal()[labels]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )


# Train and evaluate one independent linear probe.
def train_one_seed(
    seed: int,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    test_sample_paths: list[str],
    class_names: list[str],
    output_dir: Path,
    device: torch.device,
) -> dict[str, float | int]:
    """Train one linear head and retain only its best test predictions."""
    seed_everything(seed)

    # Create a fresh head and fresh sampler for this seed.
    model = LinearProbe(
        feature_dim=train_features.shape[1],
        num_classes=len(class_names),
    ).to(device)

    train_dataset = TensorDataset(train_features, train_labels)
    test_dataset = TensorDataset(test_features, test_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=make_weighted_sampler(train_labels, seed),
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    # Only the linear head is optimized; the DINOv3 features stay fixed.
    criterion = nn.CrossEntropyLoss()
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

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            loss = criterion(model(features), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        metrics, labels, predictions = evaluate_classifier(
            model,
            test_loader,
            device,
            num_classes=len(class_names),
        )
        mean_loss = total_loss / len(train_loader)
        print(
            f"Seed {seed} | Epoch {epoch:02d}/{EPOCHS} | "
            f"Loss {mean_loss:.4f} | Accuracy {metrics['accuracy']:.4f} | "
            f"Macro-F1 {metrics['macro_f1']:.4f}"
        )

        if best_result is None or metrics["macro_f1"] > best_result["macro_f1"]:
            best_result = {
                "seed": seed,
                "best_epoch": epoch,
                **metrics,
                **seven_class_metrics(labels, predictions, class_names),
            }
            best_labels = labels.copy()
            best_predictions = predictions.copy()

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
            "method": "frozen_dino",
            "seeds": SEEDS,
            "settings": {
                "image_size": IMAGE_SIZE,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
            },
            "metrics": summarize_runs(results, metric_names),
        },
        output_dir / "summary.json",
    )


# Run feature extraction once, followed by the five linear-probe runs.
def main() -> None:
    # Load machine-specific paths from the ignored config.yaml.
    config = load_config(CONFIG_PATH, REPO_ROOT)
    paths = config["paths"]

    # Fail early if the DINOv3 or dataset paths are incorrect.
    required_paths = ["dino_repo", "dino_weights", "train_sar", "test_sar"]
    for name in required_paths:
        path = paths[name]
        if not path.exists():
            raise FileNotFoundError(f"Configured path does not exist: paths.{name}={path}")

    # Keep this method's compact outputs in a separate directory.
    output_dir = paths["output_dir"] / "frozen_dino"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the first visible GPU when available; otherwise run on CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the raw SAR images with the same preprocessing used in the experiments.
    transform = make_dino_transform(IMAGE_SIZE)
    train_images = ImageFolder(paths["train_sar"], transform=transform)
    test_images = ImageFolder(paths["test_sar"], transform=transform)
    class_names = list(train_images.classes)
    test_sample_paths = [path for path, _ in test_images.samples]

    # Feature extraction is deterministic, so neither image loader is shuffled.
    feature_loader_options = {
        "batch_size": FEATURE_BATCH_SIZE,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
    }
    train_image_loader = DataLoader(train_images, **feature_loader_options)
    test_image_loader = DataLoader(test_images, **feature_loader_options)

    # Extract frozen DINOv3 features once and reuse them for every seed.
    backbone = load_dino_model(paths["dino_repo"], paths["dino_weights"]).to(device)

    print("Extracting frozen training features...")
    train_features, train_labels = extract_features(
        backbone,
        train_image_loader,
        device,
    )
    print("Extracting frozen test features...")
    test_features, test_labels = extract_features(
        backbone,
        test_image_loader,
        device,
    )

    # The backbone is no longer needed once both feature tensors are in memory.
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Train five independent linear heads on the same frozen features.
    results = []
    for seed in SEEDS:
        results.append(
            train_one_seed(
                seed,
                train_features,
                train_labels,
                test_features,
                test_labels,
                test_sample_paths,
                class_names,
                output_dir,
                device,
            )
        )

    # Save per-seed predictions and aggregate metrics only.
    save_results(results, output_dir)
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()

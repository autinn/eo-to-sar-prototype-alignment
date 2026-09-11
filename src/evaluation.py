"""Numerical evaluation shared by the five training scripts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader


LIGHT_VEHICLE_CLASSES = {"SUV", "pickup_truck", "sedan", "van"}


# Metrics reported for every experiment.
def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    """Calculate top-1 accuracy and macro-F1 over a fixed class set."""
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=np.arange(num_classes),
                average="macro",
                zero_division=0,
            )
        ),
    }


# Shared inference loop so every method is evaluated identically.
def evaluate_classifier(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate a classifier and return metrics, labels and predictions."""
    model.eval()
    label_batches = []
    prediction_batches = []

    with torch.inference_mode():
        for inputs, labels in data_loader:
            outputs = model(inputs.to(device, non_blocking=True))
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            predictions = logits.argmax(dim=1)

            label_batches.append(labels.cpu())
            prediction_batches.append(predictions.cpu())

    labels = torch.cat(label_batches).numpy()
    predictions = torch.cat(prediction_batches).numpy()
    metrics = classification_metrics(labels, predictions, num_classes)
    return metrics, labels, predictions


# Post-hoc seven-class evaluation used in the paper.
def merge_light_vehicle_labels(
    labels: np.ndarray,
    class_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Merge SUV, pickup truck, sedan and van into one evaluation class."""
    merged_names = ["light_vehicle"] + [
        name for name in class_names if name not in LIGHT_VEHICLE_CLASSES
    ]

    old_to_new = {
        old_index: (
            0
            if class_name in LIGHT_VEHICLE_CLASSES
            else merged_names.index(class_name)
        )
        for old_index, class_name in enumerate(class_names)
    }
    merged_labels = np.asarray([old_to_new[int(label)] for label in labels])
    return merged_labels, merged_names


def seven_class_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Sequence[str],
) -> dict[str, float]:
    """Calculate metrics after the paper's light-vehicle class merge."""
    merged_labels, merged_names = merge_light_vehicle_labels(labels, class_names)
    merged_predictions, _ = merge_light_vehicle_labels(predictions, class_names)
    metrics = classification_metrics(
        merged_labels,
        merged_predictions,
        num_classes=len(merged_names),
    )
    return {
        "accuracy_7class": metrics["accuracy"],
        "macro_f1_7class": metrics["macro_f1"],
    }


# Compact saved output from the selected epoch.
def save_predictions(
    output_path: str | Path,
    sample_paths: Sequence[str],
    labels: np.ndarray,
    predictions: np.ndarray,
    class_names: Sequence[str],
) -> None:
    """Save compact best-epoch predictions without logits or checkpoints."""
    if not (len(sample_paths) == len(labels) == len(predictions)):
        raise ValueError("Sample paths, labels and predictions must have equal length.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample", "true_index", "true_class", "predicted_index", "predicted_class"]
        )
        for sample_path, label, prediction in zip(sample_paths, labels, predictions):
            writer.writerow(
                [
                    Path(sample_path).name,
                    int(label),
                    class_names[int(label)],
                    int(prediction),
                    class_names[int(prediction)],
                ]
            )

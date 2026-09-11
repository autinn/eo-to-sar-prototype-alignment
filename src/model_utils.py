"""DINOv3 loading, preprocessing and frozen feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2


DINO_MODEL_NAME = "dinov3_vits16plus"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# DINOv3 classifier shared by all LoRA fine-tuning methods.
class DINOClassifier(nn.Module):
    """L2-normalize DINOv3 features before applying a linear classifier."""

    def __init__(self, backbone: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, num_classes)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = F.normalize(self.backbone(images), dim=1)
        return self.head(features), features


# Add trainable low-rank adapters to the attention qkv layers.
def apply_lora(
    model: nn.Module,
    rank: int,
    alpha: int,
    target_modules: Sequence[str],
    dropout: float,
) -> nn.Module:
    """Apply LoRA while keeping the classification head trainable."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as error:
        raise ImportError(
            "LoRA training requires the 'peft' package. Install it with: pip install peft"
        ) from error

    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=list(target_modules),
        lora_dropout=dropout,
        bias="none",
        modules_to_save=["head"],
    )
    model = get_peft_model(model, config)

    trainable, total = model.get_nb_trainable_parameters()
    print(
        f"LoRA trainable parameters: {trainable:,}/{total:,} "
        f"({100 * trainable / total:.2f}%)"
    )
    return model


# Load the exact local DINOv3 backbone used by all methods.
def load_dino_model(repo_dir: str | Path, weights_file: str | Path) -> torch.nn.Module:
    """Load DINOv3 ViT-S/16+ from a local repository and weights file."""
    repo_dir = Path(repo_dir)
    weights_file = Path(weights_file)

    if not repo_dir.is_dir():
        raise FileNotFoundError(f"DINOv3 repository not found: {repo_dir}")
    if not weights_file.is_file():
        raise FileNotFoundError(f"DINOv3 weights not found: {weights_file}")

    return torch.hub.load(
        str(repo_dir),
        model=DINO_MODEL_NAME,
        source="local",
        weights=str(weights_file),
        trust_repo=True,
    )


# Shared image preprocessing for EO and SAR inputs.
def make_dino_transform(image_size: int = 128) -> v2.Compose:
    """Resize images and apply the normalization used by DINOv3."""
    return v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size, image_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# Frozen feature extraction used by the linear-probe baseline.
def extract_features(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract raw global DINOv3 features and labels into CPU memory."""
    model.eval()
    feature_batches = []
    label_batches = []

    with torch.inference_mode():
        for batch_index, (images, labels) in enumerate(data_loader, start=1):
            features = model(images.to(device, non_blocking=True))
            feature_batches.append(features.cpu())
            label_batches.append(labels.cpu())

            if batch_index % 100 == 0 or batch_index == len(data_loader):
                print(f"Extracted features from {batch_index}/{len(data_loader)} batches")

    if not feature_batches:
        raise ValueError("The feature DataLoader produced no batches.")

    return torch.cat(feature_batches), torch.cat(label_batches)

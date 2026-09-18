"""Extract backbone features once and persist them for reuse.

The released code computes features and discards them. ``extract_features``
returns tensors that live only inside one ``main()``; ``evaluate_classifier``
throws away both logits and features, keeping only argmax predictions; and the
saved CSVs record predicted classes without confidences. Nothing reaches disk, so
there is no path from a completed run to any embedding.

That blocks every analysis in this repository. Prototype geometry, alignment
residuals and error placement are all functions of the features, and
recomputing them means re-running feature extraction over the full training set
each time - on the real dataset, 455,635 images per modality.

This module adds the missing layer. Features are saved as ``.npz`` rather than
``.pt`` because the repository's ``.gitignore`` excludes ``*.pt`` and
``checkpoints/``, so a ``.pt`` file would be silently untracked; ``.npz`` is also
portable and compresses well. Nothing here modifies existing code - it wraps
``extract_features`` and writes the result.

Run with::

    python src/extract_and_save_features.py --split train_eo --stub
    python src/extract_and_save_features.py --split train_eo   # once DINOv3 lands
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model_utils import extract_features, load_dino_model, make_dino_transform
from utils import load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"

IMAGE_SIZE = 128
BATCH_SIZE = 256


def save_features(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_names: list[str],
    sample_paths: list[str],
    output_path: Path,
    metadata: dict[str, str] | None = None,
) -> None:
    """Write features, labels and provenance to a compressed ``.npz``.

    Provenance is stored alongside the arrays because a feature file is
    meaningless without knowing which backbone produced it. A stub-derived file
    and a DINOv3-derived file are indistinguishable by shape alone, and
    confusing them would silently invalidate any downstream result.
    """
    if features.shape[0] != labels.shape[0]:
        raise ValueError(
            f"features has {features.shape[0]} rows but labels has {labels.shape[0]}."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "features": features.detach().cpu().numpy(),
        "labels": labels.detach().cpu().numpy(),
        "class_names": np.array(class_names, dtype=object),
        "sample_paths": np.array(sample_paths, dtype=object),
    }
    for key, value in (metadata or {}).items():
        payload[f"meta_{key}"] = np.array(str(value))

    np.savez_compressed(output_path, **payload)


def load_saved_features(
    path: Path,
) -> tuple[torch.Tensor, torch.Tensor, list[str], dict[str, str]]:
    """Read back a saved feature file.

    Returns ``(features, labels, class_names, metadata)``.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Feature file not found: {path}")

    with np.load(path, allow_pickle=True) as archive:
        features = torch.from_numpy(archive["features"])
        labels = torch.from_numpy(archive["labels"])
        class_names = [str(name) for name in archive["class_names"]]
        metadata = {
            key[len("meta_") :]: str(archive[key])
            for key in archive.files
            if key.startswith("meta_")
        }

    return features, labels, class_names, metadata


def extract_and_save(
    backbone: torch.nn.Module,
    dataset: ImageFolder,
    device: torch.device,
    output_path: Path,
    batch_size: int = BATCH_SIZE,
    metadata: dict[str, str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract features for a dataset and persist them.

    Reuses ``extract_features`` from ``model_utils``, which returns raw
    un-normalized features - the form ``build_eo_prototypes`` expects, since it
    averages before normalizing.
    """
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    features, labels = extract_features(backbone.to(device), loader, device)

    save_features(
        features,
        labels,
        class_names=list(dataset.classes),
        sample_paths=[path for path, _ in dataset.samples],
        output_path=output_path,
        metadata=metadata,
    )
    return features, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("train_sar", "train_eo", "test_sar"),
        required=True,
        help="Which configured dataset path to extract.",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help=(
            "Use the stub backbone instead of DINOv3. Features carry no meaning; "
            "this only exercises the code path before weights are available."
        ),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--data-root", type=Path, default=None)
    arguments = parser.parse_args()

    # The fake dataset can be used without a config.yaml at all.
    if arguments.data_root is not None:
        subdirectory = {
            "train_sar": Path("train") / "SAR_Train",
            "train_eo": Path("train") / "EO_Train",
            "test_sar": Path("test") / "IID",
        }[arguments.split]
        dataset_path = arguments.data_root / subdirectory
        output_dir = REPO_ROOT / "outputs" / "features"
    else:
        config = load_config(CONFIG_PATH, REPO_ROOT)
        dataset_path = config["paths"][arguments.split]
        output_dir = config["paths"]["output_dir"] / "features"

    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    if arguments.stub:
        from stub_backbone import load_stub_model

        backbone = load_stub_model()
        backbone_name = "stub"
        print("WARNING: using the stub backbone. Features are meaningless.")
    else:
        config = load_config(CONFIG_PATH, REPO_ROOT)
        backbone = load_dino_model(
            config["paths"]["dino_repo"], config["paths"]["dino_weights"]
        )
        backbone_name = "dinov3_vits16plus"

    dataset = ImageFolder(dataset_path, transform=make_dino_transform(IMAGE_SIZE))
    output_path = arguments.out or (output_dir / f"{arguments.split}_{backbone_name}.npz")

    print(f"Extracting {len(dataset)} images from {dataset_path} on {device}")
    features, labels = extract_and_save(
        backbone,
        dataset,
        device,
        output_path,
        batch_size=arguments.batch_size,
        metadata={
            "backbone": backbone_name,
            "split": arguments.split,
            "image_size": str(IMAGE_SIZE),
            "source": str(dataset_path),
        },
    )

    size_mb = output_path.stat().st_size / 1e6
    print(f"Saved {tuple(features.shape)} features to {output_path} ({size_mb:.2f} MB)")
    print(f"  classes: {len(dataset.classes)}, labels: {tuple(labels.shape)}")


if __name__ == "__main__":
    main()

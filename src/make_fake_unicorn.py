"""Generate a synthetic UNICORNv2-shaped dataset for pipeline testing.

UNICORNv2 requires a Codabench account, and the real experiments additionally
require gated DINOv3 weights. Neither is needed to check that the data-loading,
training and evaluation code paths run. This script writes PNG chips into the
exact ``ImageFolder`` layout the training scripts expect, so the whole pipeline
can be exercised offline.

The layout matches ``config-example.yaml``::

    data/fake/
        train/SAR_Train/<class_name>/*.png
        train/EO_Train/<class_name>/*.png
        test/IID/<class_name>/*.png

Chip sizes follow the real dataset: SAR chips are 55x55 and EO chips 31x31, both
tiny, so a ten-image-per-class dataset is well under a megabyte. The class names
are the real ten, including the four light-vehicle classes that
``seven_class_metrics`` merges - using different names would leave that code path
untested.

Images carry a weak per-class signal rather than pure noise. A classifier cannot
learn anything useful from ten examples, and nothing here is meant to produce a
meaningful accuracy; the signal exists only so that features do not collapse to a
single point and downstream geometry measurements stay non-degenerate.

Run with::

    python src/make_fake_unicorn.py --per-class 10
    python src/make_fake_unicorn.py --per-class 50 --imbalance-ratio 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]

# The real UNICORNv2 classes. The first four are merged by seven_class_metrics
# via LIGHT_VEHICLE_CLASSES in evaluation.py, so their spelling must match.
CLASS_NAMES = [
    "SUV",
    "pickup_truck",
    "sedan",
    "van",
    "box_truck",
    "bus",
    "flatbed_truck",
    "motorcycle",
    "pickup_truck_with_trailer",
    "semi_truck_with_trailer",
]

SAR_CHIP_SIZE = 55
EO_CHIP_SIZE = 31
TEST_IMAGES_PER_CLASS = 8


def _class_counts(
    per_class: int,
    num_classes: int,
    imbalance_ratio: float,
) -> list[int]:
    """Images per class, optionally following a geometric imbalance.

    The real training set is imbalanced by roughly 1000:1 (364,291 sedans against
    353 semi trucks), which is why the training scripts use a
    ``WeightedRandomSampler``. Reproducing that skew lets the sampler be tested.
    """
    if imbalance_ratio < 1.0:
        raise ValueError("imbalance_ratio must be at least 1.0.")
    if imbalance_ratio == 1.0:
        return [per_class] * num_classes

    # Geometric spacing from per_class down to per_class / imbalance_ratio.
    factors = np.geomspace(1.0, 1.0 / imbalance_ratio, num_classes)
    return [max(1, int(round(per_class * factor))) for factor in factors]


def _make_chip(
    size: int,
    channels: int,
    class_index: int,
    num_classes: int,
    generator: np.random.Generator,
) -> np.ndarray:
    """One chip: noise plus a faint class-dependent pattern."""
    image = generator.random((size, size, channels)) * 0.5

    # A low-frequency gradient whose orientation depends on the class, so that
    # different classes are at least in principle separable.
    angle = 2.0 * np.pi * class_index / num_classes
    axis = np.linspace(-1.0, 1.0, size)
    horizontal, vertical = np.meshgrid(axis, axis)
    pattern = 0.3 * (np.cos(angle) * horizontal + np.sin(angle) * vertical)
    image += pattern[:, :, None]

    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def _write_split(
    root: Path,
    counts: list[int],
    size: int,
    channels: int,
    seed: int,
) -> int:
    """Write one split, returning the number of files created."""
    generator = np.random.default_rng(seed)
    written = 0

    for class_index, (class_name, count) in enumerate(zip(CLASS_NAMES, counts)):
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for image_index in range(count):
            array = _make_chip(size, channels, class_index, len(CLASS_NAMES), generator)
            image = Image.fromarray(
                array.squeeze(-1) if channels == 1 else array,
                mode="L" if channels == 1 else "RGB",
            )
            image.save(class_dir / f"{class_name}_{image_index:05d}.png")
            written += 1

    return written


def generate(
    output_dir: Path,
    per_class: int = 10,
    imbalance_ratio: float = 1.0,
    seed: int = 0,
) -> dict[str, int]:
    """Write a complete fake dataset and report how many files each split holds."""
    train_counts = _class_counts(per_class, len(CLASS_NAMES), imbalance_ratio)
    test_counts = [TEST_IMAGES_PER_CLASS] * len(CLASS_NAMES)

    # EO chips are RGB and SAR chips grayscale, as in the real dataset.
    sar_train = _write_split(
        output_dir / "train" / "SAR_Train", train_counts, SAR_CHIP_SIZE, 1, seed
    )
    eo_train = _write_split(
        output_dir / "train" / "EO_Train", train_counts, EO_CHIP_SIZE, 3, seed + 1
    )
    sar_test = _write_split(
        output_dir / "test" / "IID", test_counts, SAR_CHIP_SIZE, 1, seed + 2
    )

    return {"sar_train": sar_train, "eo_train": eo_train, "sar_test": sar_test}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic UNICORNv2-shaped dataset for pipeline testing."
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=10,
        help="Training images for the largest class (default: 10).",
    )
    parser.add_argument(
        "--imbalance-ratio",
        type=float,
        default=1.0,
        help=(
            "Ratio between the largest and smallest class. The real dataset is "
            "about 1000:1; use it to exercise the weighted sampler."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "fake",
        help="Output directory (default: data/fake).",
    )
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    counts = generate(
        arguments.out,
        per_class=arguments.per_class,
        imbalance_ratio=arguments.imbalance_ratio,
        seed=arguments.seed,
    )

    total_bytes = sum(path.stat().st_size for path in arguments.out.rglob("*.png"))
    print(f"Wrote fake dataset to {arguments.out}")
    for split, count in counts.items():
        print(f"  {split:12} {count:6d} images")
    print(f"  {'total size':12} {total_bytes / 1e6:6.2f} MB")
    print()
    print("Point config.yaml at it with:")
    print(f'  train_sar: "{arguments.out / "train" / "SAR_Train"}"')
    print(f'  train_eo:  "{arguments.out / "train" / "EO_Train"}"')
    print(f'  test_sar:  "{arguments.out / "test" / "IID"}"')


if __name__ == "__main__":
    main()

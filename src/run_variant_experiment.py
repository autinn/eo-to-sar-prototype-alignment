"""The corrected prototype-variant experiment.

Runs every variant in the ablation ladder under one protocol, fixing the three
confounds documented in ``analysis/CONFOUNDS.md``:

1. **One learning rate for every condition.** The published EO and synthetic runs
   used 8e-6 and 4e-6, so their 4.0 pp gap cannot be attributed to prototype type
   alone.
2. **One seed set for every condition**, which makes paired per-seed differences
   available. The published runs shared only seed 42, so only unpaired tests were
   possible - a real loss of power given seed-to-seed spread of 1.3 to 2.0 points
   against effects of 1.5 to 4.0.
3. **Model selection on a validation split**, carved from the training set. Every
   published script selects its best epoch by evaluating on the test set, and
   ``config-example.yaml`` defines no validation path.

The ladder answers a question the two-point control cannot. Real class means and
an equiangular frame differ in angular geometry, class identity and optical
origin simultaneously; the intermediate variants hold some of those fixed while
breaking others. In particular ``rotated`` keeps every pairwise angle while
destroying the correspondence to specific optical directions, so comparing it
against ``source`` isolates whether the gain comes from geometry or from the
directions themselves.

Alongside accuracy, each run records the geometry of its prototypes and the
placement of its alignment residual, so that a difference in outcome can be
related to a difference in what the prototypes actually did.

Usage::

    # Now, against the stub backbone - proves the protocol runs.
    python src/run_variant_experiment.py --stub --data-root data/fake --seeds 2 --epochs 2

    # Once DINOv3 access is granted.
    python src/run_variant_experiment.py --seeds 15

Fifteen seeds is the figure the power analysis gives for resolving the
EO-versus-MMD gap; smaller effects in the ladder may need more.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from error_placement import alignment_residual, placement_report
from evaluation import evaluate_classifier, seven_class_metrics
from feature_simulator import class_means_from_features
from geometry import geometry_report
from model_utils import (
    DINOClassifier,
    apply_lora,
    extract_features,
    load_dino_model,
    make_dino_transform,
)
from prototype_variants import build_ladder
from stats_utils import cohens_d, pooled_sd, welch_ttest
from utils import load_config, save_json, seed_everything


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.yaml"

IMAGE_SIZE = 128
BATCH_SIZE = 256
VALIDATION_FRACTION = 0.15

# One learning rate for every condition; this is the point of the rerun.
LEARNING_RATE = 8e-6
WEIGHT_DECAY = 1e-2
ALIGNMENT_WEIGHT = 0.6
ALIGNMENT_LOSS_SCALE = 2.0

LORA_RANK = 28
LORA_ALPHA = 56
LORA_TARGET_MODULES = ["qkv"]
LORA_DROPOUT = 0.05


def make_weighted_sampler(labels: list[int], seed: int) -> WeightedRandomSampler:
    """Inverse-frequency sampler, matching the published training scripts.

    Reimplemented rather than imported because the published copy lives inside a
    training script. UNICORNv2 is imbalanced roughly 1000:1, so this is load
    bearing rather than cosmetic.
    """
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


def split_train_validation(
    dataset: ImageFolder,
    fraction: float,
    seed: int,
) -> tuple[Subset, Subset]:
    """Stratified train/validation split.

    Stratified rather than random because the rarest class can hold very few
    images; a random split can drop it from validation entirely and make the
    macro-F1 used for model selection undefined for that class.
    """
    targets = torch.as_tensor(dataset.targets)
    generator = torch.Generator().manual_seed(seed)

    train_indices: list[int] = []
    validation_indices: list[int] = []
    for class_index in targets.unique().tolist():
        class_positions = (targets == class_index).nonzero(as_tuple=True)[0]
        shuffled = class_positions[torch.randperm(len(class_positions), generator=generator)]
        # Keep at least one validation example per class where possible.
        n_validation = max(1, int(round(fraction * len(shuffled)))) if len(shuffled) > 1 else 0
        validation_indices.extend(shuffled[:n_validation].tolist())
        train_indices.extend(shuffled[n_validation:].tolist())

    return Subset(dataset, train_indices), Subset(dataset, validation_indices)


def train_one_run(
    variant_name: str,
    prototypes: torch.Tensor,
    train_subset: Subset,
    validation_subset: Subset,
    test_dataset: ImageFolder,
    class_names: list[str],
    backbone_factory,
    seed: int,
    epochs: int,
    device: torch.device,
) -> dict:
    """One variant at one seed, selecting the best epoch on validation."""
    seed_everything(seed)

    model = DINOClassifier(backbone_factory(), num_classes=len(class_names))
    model = apply_lora(
        model,
        rank=LORA_RANK,
        alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        dropout=LORA_DROPOUT,
    ).to(device)

    train_labels = [train_subset.dataset.targets[index] for index in train_subset.indices]
    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        sampler=make_weighted_sampler(train_labels, seed),
        num_workers=0,
    )
    validation_loader = DataLoader(validation_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    classification_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    device_prototypes = prototypes.to(device)

    best_validation_f1 = -1.0
    best_epoch = 0
    best_state: dict | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits, features = model(images)

            cosine_distance = 1 - F.cosine_similarity(
                features, device_prototypes[labels], dim=1
            )
            loss = (1 - ALIGNMENT_WEIGHT) * classification_loss(logits, labels) + (
                ALIGNMENT_WEIGHT * cosine_distance.mean() * ALIGNMENT_LOSS_SCALE
            )
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Selection signal is validation, never test.
        validation_metrics, _, _ = evaluate_classifier(
            model, validation_loader, device, num_classes=len(class_names)
        )
        if validation_metrics["macro_f1"] > best_validation_f1:
            best_validation_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)

    # Test is touched once, after selection is complete.
    test_metrics, test_labels, test_predictions = evaluate_classifier(
        model, test_loader, device, num_classes=len(class_names)
    )
    merged = seven_class_metrics(test_labels, test_predictions, class_names)

    # Residual placement on the test set, using the head that consumed the features.
    model.eval()
    with torch.inference_mode():
        feature_batches, label_batches = [], []
        for images, labels in test_loader:
            _, features = model(images.to(device))
            feature_batches.append(features.cpu())
            label_batches.append(labels)
    residual = alignment_residual(
        torch.cat(feature_batches), prototypes, torch.cat(label_batches)
    )
    head_weight = model.base_model.model.head.modules_to_save.default.weight.detach().cpu()

    return {
        "variant": variant_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_macro_f1": float(best_validation_f1),
        **{key: float(value) for key, value in test_metrics.items()},
        **{key: float(value) for key, value in merged.items()},
        **{
            f"placement_{key}": value
            for key, value in placement_report(residual, head_weight).items()
        },
    }


def paired_comparison(
    results: list[dict],
    variant_a: str,
    variant_b: str,
    metric: str = "accuracy",
) -> dict:
    """Paired and unpaired comparison of two variants across shared seeds.

    Pairing is possible only because every variant runs on the same seed set,
    which is the second correction this experiment makes.
    """
    by_seed_a = {row["seed"]: row[metric] for row in results if row["variant"] == variant_a}
    by_seed_b = {row["seed"]: row[metric] for row in results if row["variant"] == variant_b}
    shared = sorted(set(by_seed_a) & set(by_seed_b))

    if len(shared) < 2:
        raise ValueError(
            f"Need at least two shared seeds to compare {variant_a} and {variant_b}."
        )

    values_a = torch.tensor([by_seed_a[seed] for seed in shared], dtype=torch.float64)
    values_b = torch.tensor([by_seed_b[seed] for seed in shared], dtype=torch.float64)
    differences = values_a - values_b

    mean_a, std_a = float(values_a.mean()), float(values_a.std(unbiased=True))
    mean_b, std_b = float(values_b.mean()), float(values_b.std(unbiased=True))

    result = {
        "variant_a": variant_a,
        "variant_b": variant_b,
        "metric": metric,
        "n_seeds": len(shared),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_paired_difference": float(differences.mean()),
        "std_paired_difference": float(differences.std(unbiased=True)),
    }

    # The paired test is the more sensitive one; report the unpaired test too so
    # the gain from a shared seed set is visible.
    #
    # Guard against a vanishing denominator. If every seed shows almost exactly
    # the same difference the paired t statistic diverges - a contrived example
    # with an identical +2.0 difference on all five seeds produces t ~ 3e15,
    # which is an artefact of dividing by floating-point noise rather than
    # evidence. Such a result is reported as a flag rather than a number.
    difference_std = float(differences.std(unbiased=True))
    scale = max(abs(float(differences.mean())), 1e-12)
    if difference_std > 1e-9 * scale:
        result["paired_t"] = float(differences.mean()) / (
            difference_std / len(shared) ** 0.5
        )
    else:
        result["paired_t"] = None
        result["paired_note"] = (
            "Paired differences are identical to numerical precision, so the "
            "paired t statistic is undefined. With real training runs this "
            "indicates the two conditions produced the same model."
        )

    if std_a > 0 or std_b > 0:
        unpaired = welch_ttest(mean_a, std_a, len(shared), mean_b, std_b, len(shared))
        result["unpaired_t"] = unpaired["t_statistic"]
        result["unpaired_p"] = unpaired["p_value"]
        result["cohens_d"] = cohens_d(mean_a, std_a, mean_b, std_b)
        result["pooled_sd"] = pooled_sd(std_a, std_b)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stub", action="store_true", help="Use the stub backbone.")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--seeds", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Subset of ladder variants to run (default: all).",
    )
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "analysis")
    arguments = parser.parse_args()

    if arguments.data_root is not None:
        train_sar_dir = arguments.data_root / "train" / "SAR_Train"
        train_eo_dir = arguments.data_root / "train" / "EO_Train"
        test_sar_dir = arguments.data_root / "test" / "IID"
    else:
        paths = load_config(CONFIG_PATH, REPO_ROOT)["paths"]
        train_sar_dir, train_eo_dir, test_sar_dir = (
            paths["train_sar"], paths["train_eo"], paths["test_sar"]
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    if arguments.stub:
        from stub_backbone import load_stub_model

        backbone_factory = load_stub_model
        backbone_name = "stub"
        print("WARNING: stub backbone. Results are meaningless; this checks the protocol.")
    else:
        paths = load_config(CONFIG_PATH, REPO_ROOT)["paths"]
        backbone_factory = lambda: load_dino_model(  # noqa: E731
            paths["dino_repo"], paths["dino_weights"]
        )
        backbone_name = "dinov3_vits16plus"

    transform = make_dino_transform(IMAGE_SIZE)
    train_sar = ImageFolder(train_sar_dir, transform=transform)
    train_eo = ImageFolder(train_eo_dir, transform=transform)
    test_sar = ImageFolder(test_sar_dir, transform=transform)
    class_names = list(train_sar.classes)

    print(f"Device: {device}, backbone: {backbone_name}")
    print(f"{len(train_sar)} SAR train, {len(train_eo)} EO train, {len(test_sar)} test")

    # EO prototypes come from the full EO training set, extracted once.
    eo_loader = DataLoader(train_eo, batch_size=BATCH_SIZE, shuffle=False)
    eo_features, eo_labels = extract_features(backbone_factory().to(device), eo_loader, device)
    source_prototypes = class_means_from_features(eo_features, eo_labels, len(class_names))
    del eo_features, eo_labels, eo_loader

    ladder = build_ladder(source_prototypes, seed=0)
    if arguments.variants:
        ladder = {name: ladder[name] for name in arguments.variants}

    geometry = {name: geometry_report(variant) for name, variant in ladder.items()}
    seeds = list(range(arguments.seeds))

    print()
    print(f"Running {len(ladder)} variants x {len(seeds)} seeds at lr={LEARNING_RATE}")
    print()

    results: list[dict] = []
    for variant_name, prototypes in ladder.items():
        for seed in seeds:
            row = train_one_run(
                variant_name, prototypes, *split_train_validation(
                    train_sar, VALIDATION_FRACTION, seed
                ),
                test_sar, class_names, backbone_factory, seed, arguments.epochs, device,
            )
            results.append(row)
            print(
                f"  {variant_name:<20} seed {seed:<3} "
                f"acc {row['accuracy']:.3f}  F1 {row['macro_f1']:.3f}  "
                f"concentration {row['placement_concentration']:.2f}"
            )

    # Every variant against the real optical prototypes.
    comparisons = []
    if "source" in ladder and len(seeds) >= 2:
        for variant_name in ladder:
            if variant_name != "source":
                comparisons.append(paired_comparison(results, "source", variant_name))

    print()
    print("=" * 78)
    print("Paired comparisons against the real optical prototypes")
    print("=" * 78)
    print(f"  {'variant':<22}{'diff':>8}{'paired t':>10}{'unpaired t':>12}")
    for row in comparisons:
        print(
            f"  {row['variant_b']:<22}{row['mean_paired_difference']:+8.3f}"
            f"{row.get('paired_t', float('nan')):10.2f}"
            f"{row.get('unpaired_t', float('nan')):12.2f}"
        )

    arguments.out.mkdir(parents=True, exist_ok=True)
    output_path = arguments.out / f"variant_experiment_{backbone_name}.json"
    save_json(
        {
            "backbone": backbone_name,
            "protocol": {
                "learning_rate": LEARNING_RATE,
                "seeds": seeds,
                "epochs": arguments.epochs,
                "validation_fraction": VALIDATION_FRACTION,
                "selection": "validation macro-F1",
                "note": (
                    "One learning rate and one seed set across all conditions, with "
                    "model selection on a held-out validation split. See "
                    "analysis/CONFOUNDS.md for why the published runs cannot be "
                    "compared this way."
                ),
            },
            "geometry": geometry,
            "runs": results,
            "comparisons": comparisons,
        },
        output_path,
    )
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()

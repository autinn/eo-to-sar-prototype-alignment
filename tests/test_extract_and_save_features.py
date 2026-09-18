"""Tests for feature persistence.

The reason this module exists is that the released code never writes features to
disk, so these tests mostly check round-trip fidelity and the provenance
metadata. The metadata matters more than it looks: a file of stub features and a
file of DINOv3 features have identical shape and dtype, and mistaking one for the
other would invalidate results without any visible symptom.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from extract_and_save_features import (  # noqa: E402
    extract_and_save,
    load_saved_features,
    save_features,
)


def _payload(n: int = 40, dim: int = 16, num_classes: int = 4):
    torch.manual_seed(0)
    features = torch.randn(n, dim)
    labels = torch.arange(n) % num_classes
    class_names = [f"class_{index}" for index in range(num_classes)]
    sample_paths = [f"/data/{index}.png" for index in range(n)]
    return features, labels, class_names, sample_paths


class TestRoundTrip:
    def test_features_survive_unchanged(self, tmp_path):
        features, labels, class_names, sample_paths = _payload()
        path = tmp_path / "features.npz"
        save_features(features, labels, class_names, sample_paths, path)

        loaded, loaded_labels, loaded_names, _ = load_saved_features(path)
        assert torch.allclose(features, loaded, atol=1e-6)
        assert torch.equal(labels, loaded_labels)
        assert loaded_names == class_names

    def test_preserves_dtype(self, tmp_path):
        features, labels, class_names, sample_paths = _payload()
        path = tmp_path / "features.npz"
        save_features(features, labels, class_names, sample_paths, path)

        loaded, _, _, _ = load_saved_features(path)
        assert loaded.dtype == torch.float32

    def test_creates_parent_directories(self, tmp_path):
        features, labels, class_names, sample_paths = _payload()
        path = tmp_path / "nested" / "deeper" / "features.npz"
        save_features(features, labels, class_names, sample_paths, path)
        assert path.is_file()

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_saved_features(tmp_path / "absent.npz")

    def test_rejects_mismatched_labels(self, tmp_path):
        features, _, class_names, sample_paths = _payload()
        with pytest.raises(ValueError):
            save_features(
                features, torch.arange(5), class_names, sample_paths,
                tmp_path / "features.npz",
            )


class TestProvenance:
    def test_metadata_round_trips(self, tmp_path):
        features, labels, class_names, sample_paths = _payload()
        path = tmp_path / "features.npz"
        save_features(
            features, labels, class_names, sample_paths, path,
            metadata={"backbone": "stub", "split": "train_eo"},
        )

        _, _, _, metadata = load_saved_features(path)
        assert metadata["backbone"] == "stub"
        assert metadata["split"] == "train_eo"

    def test_stub_and_real_features_are_distinguishable(self, tmp_path):
        """Shape and dtype are identical, so only the metadata separates them."""
        features, labels, class_names, sample_paths = _payload()

        stub_path = tmp_path / "stub.npz"
        real_path = tmp_path / "real.npz"
        save_features(
            features, labels, class_names, sample_paths, stub_path,
            metadata={"backbone": "stub"},
        )
        save_features(
            features, labels, class_names, sample_paths, real_path,
            metadata={"backbone": "dinov3_vits16plus"},
        )

        _, _, _, stub_meta = load_saved_features(stub_path)
        _, _, _, real_meta = load_saved_features(real_path)
        assert stub_meta["backbone"] != real_meta["backbone"]

    def test_absent_metadata_gives_an_empty_dict(self, tmp_path):
        features, labels, class_names, sample_paths = _payload()
        path = tmp_path / "features.npz"
        save_features(features, labels, class_names, sample_paths, path)
        assert load_saved_features(path)[3] == {}


class TestIntegration:
    def test_extracts_from_an_image_folder(self, tmp_path):
        """Full path: generate images, extract with the stub, save, reload."""
        from torchvision.datasets import ImageFolder

        from make_fake_unicorn import generate
        from model_utils import make_dino_transform
        from stub_backbone import load_stub_model

        data_dir = tmp_path / "fake"
        generate(data_dir, per_class=2, seed=0)

        dataset = ImageFolder(
            data_dir / "train" / "EO_Train", transform=make_dino_transform(128)
        )
        output = tmp_path / "out.npz"
        features, labels = extract_and_save(
            load_stub_model(),
            dataset,
            torch.device("cpu"),
            output,
            batch_size=8,
            metadata={"backbone": "stub"},
        )

        assert features.shape == (20, 384)
        assert labels.shape == (20,)

        loaded, loaded_labels, class_names, metadata = load_saved_features(output)
        assert torch.allclose(features, loaded, atol=1e-5)
        assert torch.equal(labels, loaded_labels)
        assert len(class_names) == 10
        assert metadata["backbone"] == "stub"

    def test_saved_features_feed_the_prototype_builder(self, tmp_path):
        """The point of persisting features: prototypes can be rebuilt without
        re-running the backbone."""
        from feature_simulator import class_means_from_features

        features, labels, class_names, sample_paths = _payload(
            n=80, dim=32, num_classes=4
        )
        path = tmp_path / "features.npz"
        save_features(features, labels, class_names, sample_paths, path)

        loaded, loaded_labels, _, _ = load_saved_features(path)
        prototypes = class_means_from_features(loaded, loaded_labels, num_classes=4)

        assert prototypes.shape == (4, 32)
        assert torch.allclose(prototypes.norm(dim=1), torch.ones(4), atol=1e-5)


class TestStorageCost:
    def test_file_size_is_proportional_to_the_feature_count(self, tmp_path):
        """Sizing matters on a machine with limited disk: caching the full
        training set costs roughly 660 MB per modality at 384 dimensions."""
        small_path = tmp_path / "small.npz"
        large_path = tmp_path / "large.npz"

        for count, path in ((100, small_path), (400, large_path)):
            features, labels, class_names, sample_paths = _payload(n=count, dim=384)
            save_features(features, labels, class_names, sample_paths, path)

        ratio = large_path.stat().st_size / small_path.stat().st_size
        assert 3.0 < ratio < 5.0

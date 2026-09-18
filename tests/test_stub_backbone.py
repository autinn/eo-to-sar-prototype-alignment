"""Tests for the DINOv3 stand-in.

These tests exist to guarantee one thing: that code developed against the stub
will run unchanged against the real backbone. Every assertion therefore checks a
property of the *contract* shared with DINOv3 rather than anything about the
stub's own behaviour.

The most important of these is ``test_lora_attaches_to_qkv_modules``. If the stub
lacked a submodule named ``qkv``, peft would attach no adapters, the training
scripts would quietly optimize only the classifier head, and every smoke test
would pass while exercising none of the code that matters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model_utils import DINOClassifier, apply_lora, make_dino_transform  # noqa: E402
from stub_backbone import StubBackbone, load_stub_model  # noqa: E402


EMBED_DIM = 384
IMAGE_SIZE = 128
NUM_CLASSES = 10


class TestBackboneContract:
    def test_exposes_embed_dim(self):
        """DINOClassifier reads backbone.embed_dim to size its head."""
        assert load_stub_model().embed_dim == EMBED_DIM

    def test_output_shape(self):
        backbone = load_stub_model()
        features = backbone(torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE))
        assert features.shape == (4, EMBED_DIM)

    def test_accepts_configurable_embed_dim(self):
        backbone = load_stub_model(embed_dim=192)
        assert backbone.embed_dim == 192
        assert backbone(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)).shape == (2, 192)

    def test_accepts_single_channel_input(self):
        """SAR chips are grayscale; the transform may hand through one channel."""
        backbone = load_stub_model()
        assert backbone(torch.randn(2, 1, IMAGE_SIZE, IMAGE_SIZE)).shape == (2, EMBED_DIM)

    def test_accepts_transformed_images(self):
        """The output of make_dino_transform must be a valid input."""
        from PIL import Image
        import numpy as np

        transform = make_dino_transform(IMAGE_SIZE)
        image = Image.fromarray(
            (np.random.default_rng(0).random((55, 55, 3)) * 255).astype("uint8")
        )
        batch = transform(image).unsqueeze(0)
        assert load_stub_model()(batch).shape == (1, EMBED_DIM)

    def test_works_under_inference_mode(self):
        """extract_features runs the backbone inside torch.inference_mode."""
        backbone = load_stub_model()
        with torch.inference_mode():
            features = backbone(torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE))
        assert features.shape == (2, EMBED_DIM)

    def test_rejects_malformed_input(self):
        with pytest.raises(ValueError):
            load_stub_model()(torch.randn(3, IMAGE_SIZE, IMAGE_SIZE))

    def test_rejects_indivisible_patch_size(self):
        with pytest.raises(ValueError):
            StubBackbone(image_size=130, patch_size=16)

    def test_rejects_indivisible_head_count(self):
        with pytest.raises(ValueError):
            StubBackbone(embed_dim=100, num_heads=6)


class TestDeterminism:
    def test_same_seed_gives_same_weights(self):
        a = load_stub_model(seed=3)
        b = load_stub_model(seed=3)
        batch = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert torch.allclose(a(batch), b(batch), atol=1e-6)

    def test_different_seeds_give_different_weights(self):
        a = load_stub_model(seed=0)
        b = load_stub_model(seed=1)
        batch = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert not torch.allclose(a(batch), b(batch), atol=1e-3)

    def test_construction_does_not_disturb_global_rng(self):
        """Seeding inside the constructor must not change the caller's RNG
        stream, or it would silently alter the seeded training runs."""
        torch.manual_seed(1234)
        expected = torch.randn(3)

        torch.manual_seed(1234)
        load_stub_model(seed=999)
        actual = torch.randn(3)

        assert torch.allclose(expected, actual)


class TestIntegrationWithRepositoryCode:
    """The stub is only useful if the repository's own code accepts it."""

    def test_works_with_dino_classifier(self):
        model = DINOClassifier(load_stub_model(), num_classes=NUM_CLASSES)
        logits, features = model(torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE))
        assert logits.shape == (4, NUM_CLASSES)
        assert features.shape == (4, EMBED_DIM)

    def test_classifier_returns_normalized_features(self):
        """DINOClassifier normalizes before the head; the alignment loss and the
        prototype builders both assume unit-norm features."""
        model = DINOClassifier(load_stub_model(), num_classes=NUM_CLASSES)
        _, features = model(torch.randn(8, 3, IMAGE_SIZE, IMAGE_SIZE))
        assert torch.allclose(features.norm(dim=1), torch.ones(8), atol=1e-5)

    def test_lora_attaches_to_qkv_modules(self):
        """peft must find real adapter targets.

        If this fails, apply_lora silently trains only the classifier head and
        every downstream smoke test becomes meaningless.
        """
        model = DINOClassifier(load_stub_model(), num_classes=NUM_CLASSES)
        adapted = apply_lora(
            model, rank=28, alpha=56, target_modules=["qkv"], dropout=0.05
        )
        trainable, total = adapted.get_nb_trainable_parameters()
        assert trainable > 0
        assert trainable < total

        adapter_names = [name for name, _ in adapted.named_parameters() if "lora" in name]
        assert adapter_names, "no LoRA parameters were created"
        assert any("qkv" in name for name in adapter_names), (
            "LoRA did not attach to the qkv projections"
        )

    def test_lora_model_runs_forward_and_backward(self):
        """A full training step must work, not just a forward pass."""
        model = DINOClassifier(load_stub_model(), num_classes=NUM_CLASSES)
        adapted = apply_lora(
            model, rank=8, alpha=16, target_modules=["qkv"], dropout=0.0
        )

        logits, features = adapted(torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE))
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 3]))
        loss.backward()

        gradients = [
            parameter.grad
            for name, parameter in adapted.named_parameters()
            if "lora" in name and parameter.grad is not None
        ]
        assert gradients, "no gradients reached the LoRA parameters"
        assert any(float(gradient.abs().sum()) > 0 for gradient in gradients)

    def test_head_remains_trainable_under_lora(self):
        """apply_lora passes modules_to_save=["head"], so the classifier head
        must still receive gradients."""
        model = DINOClassifier(load_stub_model(), num_classes=NUM_CLASSES)
        adapted = apply_lora(
            model, rank=8, alpha=16, target_modules=["qkv"], dropout=0.0
        )
        head_parameters = [
            parameter
            for name, parameter in adapted.named_parameters()
            if "head" in name and parameter.requires_grad
        ]
        assert head_parameters


class TestFeatureQuality:
    """Downstream analysis needs features that are not degenerate."""

    def test_features_are_not_constant(self):
        backbone = load_stub_model()
        features = backbone(torch.randn(32, 3, IMAGE_SIZE, IMAGE_SIZE))
        assert float(features.std()) > 1e-3

    def test_features_vary_with_input(self):
        backbone = load_stub_model()
        a = backbone(torch.randn(8, 3, IMAGE_SIZE, IMAGE_SIZE))
        b = backbone(torch.randn(8, 3, IMAGE_SIZE, IMAGE_SIZE))
        assert not torch.allclose(a, b, atol=1e-3)

    def test_identical_inputs_give_identical_features(self):
        backbone = load_stub_model()
        batch = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert torch.allclose(backbone(batch), backbone(batch), atol=1e-6)

    def test_feature_cloud_occupies_many_dimensions(self):
        """A stub that collapsed everything onto one direction would make every
        geometry measurement trivially degenerate."""
        from geometry import effective_rank

        backbone = load_stub_model()
        features = backbone(torch.randn(256, 3, IMAGE_SIZE, IMAGE_SIZE))
        assert effective_rank(features) > 2.0

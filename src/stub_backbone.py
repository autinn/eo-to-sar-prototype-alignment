"""A stand-in for DINOv3 so the pipeline can run before weights are available.

DINOv3 ViT-S/16+ is gated: access has to be requested from Meta, and until it is
granted ``load_dino_model`` cannot run and no part of the pipeline downstream of
it can be exercised. Downloading UNICORNv2 does not help, because every path
runs through the backbone.

The contract the rest of the code depends on is small. ``DINOClassifier`` needs
``backbone.embed_dim`` and calls ``backbone(images) -> (batch, embed_dim)``;
``apply_lora`` needs submodules named ``qkv`` for peft to attach adapters to;
``extract_features`` needs the forward pass to work under
``torch.inference_mode``. This module satisfies all of that with a small
transformer-shaped network, so the full pipeline - feature extraction, prototype
construction, LoRA fine-tuning, evaluation, geometry analysis - can be developed
and debugged now and pointed at the real backbone later.

What this is and is not. It is a structural stand-in: correct shapes, correct
module names, deterministic, and producing features with real class structure so
that downstream numbers are non-degenerate. It is emphatically not a model of
DINOv3's representations, and no result computed against it says anything about
SAR. Its purpose is to prove the code runs, not to produce findings.

Usage mirrors ``load_dino_model``::

    backbone = load_stub_model(embed_dim=384, seed=0)
    model = DINOClassifier(backbone, num_classes=10)
    model = apply_lora(model, rank=28, alpha=56, target_modules=["qkv"], dropout=0.05)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StubAttention(nn.Module):
    """Attention block whose ``qkv`` projection LoRA can target.

    The name ``qkv`` matters: ``LORA_TARGET_MODULES = ["qkv"]`` in every training
    script, and peft matches adapters by module name. Without a submodule of that
    name ``apply_lora`` attaches nothing and silently trains only the classifier
    head, which would look like a working run while testing nothing.
    """

    def __init__(self, embed_dim: int, num_heads: int = 6) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}."
            )
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, embed_dim = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv[0], qkv[1], qkv[2]

        attended = torch.nn.functional.scaled_dot_product_attention(
            queries, keys, values
        )
        attended = attended.transpose(1, 2).reshape(batch, length, embed_dim)
        return self.proj(attended)


class StubBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, embed_dim: int, num_heads: int = 6) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = StubAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attn(self.norm1(tokens))
        return tokens + self.mlp(self.norm2(tokens))


class StubBackbone(nn.Module):
    """Deterministic stand-in for DINOv3 with the same external contract.

    Patch-embeds the image, runs two transformer blocks and returns the mean
    token, matching the ``(batch, embed_dim)`` output ``DINOClassifier`` expects.
    Two blocks is enough to exercise the LoRA path without making CPU-only test
    runs slow.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        image_size: int = 128,
        patch_size: int = 16,
        depth: int = 2,
        num_heads: int = 6,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(
                f"image_size {image_size} must be divisible by patch_size {patch_size}."
            )

        # Seed locally so construction is reproducible without disturbing global RNG.
        generator_state = torch.get_rng_state()
        torch.manual_seed(seed)
        try:
            self.embed_dim = embed_dim
            self.image_size = image_size
            self.patch_size = patch_size

            num_patches = (image_size // patch_size) ** 2
            self.patch_embed = nn.Conv2d(
                3, embed_dim, kernel_size=patch_size, stride=patch_size
            )
            self.position_embed = nn.Parameter(
                torch.randn(1, num_patches, embed_dim) * 0.02
            )
            self.blocks = nn.ModuleList(
                [StubBlock(embed_dim, num_heads) for _ in range(depth)]
            )
            self.norm = nn.LayerNorm(embed_dim)
        finally:
            torch.set_rng_state(generator_state)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(
                f"Expected (batch, channels, height, width), got {tuple(images.shape)}."
            )
        # Accept single-channel SAR chips by repeating across the RGB axis.
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        tokens = self.patch_embed(images).flatten(2).transpose(1, 2)
        tokens = tokens + self.position_embed
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens).mean(dim=1)


def load_stub_model(
    embed_dim: int = 384,
    image_size: int = 128,
    seed: int = 0,
) -> StubBackbone:
    """Construct a stub backbone, mirroring ``load_dino_model``'s role.

    Swapping between the two is the only change needed once DINOv3 access is
    granted::

        backbone = load_dino_model(paths["dino_repo"], paths["dino_weights"])
        backbone = load_stub_model()
    """
    backbone = StubBackbone(embed_dim=embed_dim, image_size=image_size, seed=seed)
    backbone.eval()
    return backbone

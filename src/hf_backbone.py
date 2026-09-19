"""Load DINOv3 from Hugging Face behind the backbone contract this code expects.

The released ``load_dino_model`` in ``model_utils.py`` calls
``torch.hub.load(source="local")``, which needs a git clone of the DINOv3
repository plus a ``.pth`` checkpoint. Meta also publishes DINOv3 on the Hugging
Face Hub, where the same weights ship as ``model.safetensors`` for
``transformers.AutoModel`` - a different loading path entirely, and the one most
access grants actually lead to.

This module bridges the two. ``HFBackbone`` wraps a ``transformers`` DINOv3 model
so it satisfies the contract the rest of the code depends on: an ``embed_dim``
attribute and a ``forward(images) -> (batch, embed_dim)`` call. That is the same
contract ``stub_backbone.py`` implements, so every downstream path - prototype
construction, LoRA fine-tuning, evaluation, geometry and placement analysis -
works unchanged.

**Which variant.** The published results used ViT-S+/16 (``dinov3_vits16plus``,
29M parameters). The plain ViT-S/16 (21M) is a different model: same embedding
dimension of 384, fewer parameters, MLP feed-forward rather than SwiGLU. Since
only the embedding dimension is load-bearing here, either runs - but a
reproduction of the published numbers needs the variant the authors used, and
this module records which one produced a given feature file so the two can never
be confused after the fact.

**LoRA targets.** The published scripts target modules named ``qkv``. The
``transformers`` implementation names its attention projections differently, so
``attention_target_modules`` reports the correct names for whichever model was
loaded; pass them to ``apply_lora`` in place of the hardcoded ``["qkv"]``.

Authentication is required once, because the repository is gated::

    HF_HUB_DISABLE_XET=1 hf auth login

The Xet variable matters on a disk-constrained machine: Hugging Face's chunk
store otherwise keeps a second copy of every downloaded blob.

Usage::

    from hf_backbone import load_hf_dinov3, attention_target_modules

    backbone = load_hf_dinov3()                       # ViT-S/16, 86 MB
    model = DINOClassifier(backbone, num_classes=10)
    model = apply_lora(model, rank=28, alpha=56,
                       target_modules=attention_target_modules(backbone),
                       dropout=0.05)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# Hugging Face repository ids for the DINOv3 variants relevant here.
VARIANTS = {
    # The variant the published results used (29M, SwiGLU FFN).
    "vits16plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    # The plain small model (21M, MLP FFN). Same 384-dim embedding.
    "vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
}

DEFAULT_VARIANT = "vits16plus"


class HFBackbone(nn.Module):
    """A Hugging Face DINOv3 model behind this project's backbone contract.

    Exposes ``embed_dim`` and returns pooled ``(batch, embed_dim)`` features, so
    it is interchangeable with ``load_dino_model`` and ``load_stub_model``.
    """

    def __init__(self, model: nn.Module, variant: str) -> None:
        super().__init__()
        self.model = model
        self.variant = variant
        self.repo_id = VARIANTS.get(variant, variant)
        self.embed_dim = int(model.config.hidden_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(
                f"Expected (batch, channels, height, width), got {tuple(images.shape)}."
            )
        # SAR chips are grayscale; repeat to the three channels DINOv3 expects,
        # matching what the stub does so the two paths stay interchangeable.
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)

        outputs = self.model(pixel_values=images)

        # Prefer the pooled CLS representation. Some configurations return no
        # pooler, in which case the first token of the final hidden state is the
        # CLS token and serves the same role.
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = outputs.last_hidden_state[:, 0]
        return pooled


def load_hf_dinov3(
    variant: str = DEFAULT_VARIANT,
    dtype: torch.dtype | None = None,
) -> HFBackbone:
    """Load a DINOv3 backbone from the Hugging Face Hub.

    ``variant`` is a key of ``VARIANTS`` or a full repository id. The weights are
    cached under ``~/.cache/huggingface`` after the first call; ViT-S/16 is about
    86 MB.

    Raises a pointed error when the repository is gated and this machine has not
    authenticated, since that is by far the most common failure.
    """
    try:
        from transformers import AutoModel
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise ImportError(
            "Loading DINOv3 from Hugging Face requires transformers. "
            "Install it with: uv pip install transformers"
        ) from error

    repo_id = VARIANTS.get(variant, variant)

    try:
        model = AutoModel.from_pretrained(repo_id, dtype=dtype)
    except OSError as error:
        if "gated" in str(error).lower() or "401" in str(error):
            raise OSError(
                f"{repo_id} is a gated repository and this machine is not "
                f"authenticated. Request access at https://huggingface.co/{repo_id} "
                "and then run:\n\n"
                "    HF_HUB_DISABLE_XET=1 hf auth login\n\n"
                "A read token from https://huggingface.co/settings/tokens is "
                "sufficient."
            ) from error
        raise

    model.eval()
    return HFBackbone(model, variant)


def attention_target_modules(backbone: HFBackbone) -> list[str]:
    """Module names LoRA should adapt, for whichever model was loaded.

    The published scripts hardcode ``["qkv"]``, which is the name used by the
    reference DINOv3 implementation. ``transformers`` names its attention
    projections differently, and peft matches adapters by name - so passing
    ``["qkv"]`` to a Hugging Face model attaches nothing and silently trains only
    the classifier head, which looks like a working run while testing nothing.

    Returns whichever projection names are actually present, preferring a fused
    ``qkv`` when the model has one and falling back to separate query/key/value
    projections otherwise.
    """
    names = {name.rsplit(".", 1)[-1] for name, _ in backbone.named_modules()}

    if "qkv" in names:
        return ["qkv"]

    separate = [name for name in ("q_proj", "k_proj", "v_proj") if name in names]
    if separate:
        return separate

    raise RuntimeError(
        "Could not find attention projection modules to adapt. Inspect "
        "`{name for name, _ in backbone.named_modules()}` and pass the correct "
        "names to apply_lora explicitly."
    )

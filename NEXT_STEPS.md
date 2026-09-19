# Next steps: from model access to real results

What to do now that DINOv3 access is granted, in order, with what to check at
each step and roughly what it costs. Every figure below was measured on this
machine (Apple M1, 8 GB) rather than estimated.

**The short version.** Authenticate, run one smoke test, extract features, then
reproduce the published numbers. Everything up to that point runs comfortably on
the M1. The full corrected experiment does not — see §6.

---

## 0. Two things to know before starting

**Your access is for a different variant than the published results used.**
The link granted is `facebook/dinov3-vits16-pretrain-lvd1689m` — ViT-S/16, 21M
parameters. The paper used **ViT-S+/16** (`dinov3_vits16plus`), 29M parameters,
SwiGLU feed-forward. Both have embedding dimension 384, which is the only
property this code depends on, so everything runs either way.

But it matters for one thing: **a reproduction of the published accuracies needs
the variant the authors used.** If `facebook/dinov3-vits16plus-pretrain-lvd1689m`
is also granted, prefer it for §5. If not, the reproduction step becomes "does
this behave plausibly" rather than "do the exact numbers come back", and that
should be stated in the write-up rather than glossed.

**The Hugging Face weights load differently than the released code expects.**
`load_dino_model` in `model_utils.py` calls `torch.hub.load(source="local")`,
which needs a git clone of the DINOv3 repo plus a `.pth` file. The Hugging Face
repository ships `model.safetensors` for `transformers`. `src/hf_backbone.py`
bridges this — it wraps the Hugging Face model behind the same contract
(`.embed_dim`, `forward(images) -> (batch, embed_dim)`) that `stub_backbone.py`
implements, so nothing downstream changes.

---

## 1. Authenticate (5 minutes, one time)

The repository is gated, so access on the website is not enough — this machine
needs a token.

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/hf auth login
```

Paste a **read** token from https://huggingface.co/settings/tokens.

`HF_HUB_DISABLE_XET=1` matters here. Hugging Face's Xet chunk store keeps a
second copy of every downloaded blob for deduplication; on this machine it once
consumed 2.6 GB while the visible cache reported 4.3 MB. For a handful of small
models it provides no benefit and roughly doubles the cost.

**Check it worked:**

```bash
.venv/bin/python -c "from huggingface_hub import whoami; print(whoami()['name'])"
```

---

## 2. Download and verify the model (2 minutes, 86 MB)

```bash
export HF_HUB_DISABLE_XET=1
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from hf_backbone import load_hf_dinov3, attention_target_modules
import torch

backbone = load_hf_dinov3()
print('variant     :', backbone.variant)
print('embed_dim   :', backbone.embed_dim)
print('params      :', f'{sum(p.numel() for p in backbone.parameters())/1e6:.1f}M')
print('output shape:', tuple(backbone(torch.randn(2, 3, 128, 128)).shape))
print('LoRA targets:', attention_target_modules(backbone))
"
```

**Expected:** `embed_dim 384`, output shape `(2, 384)`, ~21M parameters.

**Disk budget:** 86 MB against ~16 GB free. Nothing to worry about here; §4 is
where storage actually gets spent.

**Note the LoRA targets.** The published scripts hardcode `["qkv"]`, which is
what the reference implementation calls its fused attention projection. The
`transformers` implementation may name them `q_proj`/`k_proj`/`v_proj` instead,
and peft matches adapters **by name** — so passing `["qkv"]` to a model that has
no module by that name attaches nothing and silently trains only the classifier
head. That looks like a working run while testing nothing, which is why
`attention_target_modules()` exists. Whatever it prints is what to pass.

---

## 3. Smoke test against the real backbone (5 minutes)

This is the step that matters most, and the one the docs have been pointing at
all along: **prove the pipeline runs on real weights before trusting any number
it produces.**

```bash
.venv/bin/python src/make_fake_unicorn.py --per-class 10   # if not already there
.venv/bin/python src/smoke_pipeline.py --real
```

This runs every stage — data loading, feature extraction, prototype
construction, LoRA fine-tuning, evaluation, geometry and placement analysis —
against DINOv3 on synthetic chips.

**What to check:**

- It completes without error. That is the entire point.
- Stage 2 reports `EO features: (100, 384)`.
- Stage 4 prints a non-zero LoRA trainable parameter count. **If it reports
  ~0.4% trainable or fails to find adapters, the target modules are wrong** —
  go back to §2 and pass what `attention_target_modules()` printed.
- The closing banner says it ran on `dinov3_vits16plus`, not the stub.

**Accuracies here are still meaningless** — the data is synthetic noise. What is
being verified is that the code path works with real weights.

`smoke_pipeline.py` needs `config.yaml` for `--real`. If you are using the
Hugging Face route rather than a local clone, it is simpler to point the script
at `hf_backbone` directly; see §7.

---

## 4. Get the data and extract features (1–2 hours, ~1.3 GB)

UNICORNv2 comes from the [Codabench MAVIC-C competition](https://www.codabench.org/forums/12351/)
and needs an account. **Start this now if you have not** — it has been the
critical path, not the model.

The chips are small: 55×55 SAR and 31×31 EO, about 2 GB for the full dataset.

Arrange it as `config-example.yaml` describes, copy that to `config.yaml`, fill
in the paths, then:

```bash
make check      # verifies every configured path exists
make features   # extracts and caches EO, SAR and test features
```

**Measured cost on this machine:**

| Split | Images | Time | Disk |
|---|---|---|---|
| EO train | 455,635 | ~50 min | 660 MB |
| SAR train | 455,635 | ~50 min | 660 MB |
| Test | 2,000 | ~15 s | 3 MB |
| | | **~1.7 h** | **~1.3 GB** |

Against ~16 GB free this is comfortable, but it is the largest thing the project
writes. If disk is tight, **EO train plus test is enough** for the geometry and
placement analysis — prototypes come from EO and residuals are measured on test.
That halves it to ~660 MB.

Caching matters because every later analysis reads these files instead of
re-running the backbone over 455k images.

**First real measurements.** Once features exist, two questions answer
themselves immediately and cost nothing:

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from extract_and_save_features import load_saved_features
from feature_simulator import class_means_from_features
from geometry import geometry_report
from pathlib import Path

features, labels, names, meta = load_saved_features(
    Path('outputs/features/train_eo_dinov3_vits16plus.npz'))
prototypes = class_means_from_features(features, labels, len(names))
report = geometry_report(prototypes)
print('backbone      :', meta['backbone'])
print('cosine mean   :', round(report['cosine_mean'], 4))
print('cosine range  :', round(report['cosine_range'], 4))
print('effective rank:', round(report['effective_rank'], 2))
print('isotropy      :', round(report['isotropy'], 4))
"
```

This is the first genuinely new measurement in the project: **what the real
optical class-mean geometry looks like**, against the equiangular frame the
published control substitutes for it. On stub features the cosine mean was 0.98;
if real DINOv3 prototypes are similarly correlated, the gap the ETF control
spans is as large as the write-up claims.

---

## 5. Reproduce the published results (hours to overnight)

```bash
make reproduce
```

Runs the five released training scripts unmodified, at their own settings and
seeds, and tests each reproduced accuracy against the paper under both
standard-deviation conventions. Results land in `analysis/reproduction.json`,
written **before** the summary prints, so a formatting failure cannot discard a
completed sweep.

**Why this comes before the corrected experiment.** The corrected experiment
changes the protocol. If its results differ from the paper, that difference has
two possible causes — the protocol change, or something in this environment
(library version, hardware, data revision). Reproducing first separates them: if
the published numbers come back, the environment is sound and any later
difference is attributable to the protocol.

**What counts as success:** each accuracy within seed variance of the published
figure. **A failure to reproduce is itself a finding** and belongs in the
write-up, not buried — especially given the variant mismatch in §0.

**Caveat carried forward:** the published scripts select their best epoch on the
test set, so these figures carry that optimism. That is what `run_variant_experiment.py`
corrects, not this step.

---

## 6. The corrected experiment — and why it needs a GPU

```bash
make variants-real   # 7 variants x 15 seeds, one learning rate, validation selection
make analyse         # correlates placement and geometry against outcome
```

**This will not finish on the M1.** Measured projection:

| Workload | On this M1 |
|---|---|
| One epoch (387k images, fwd+bwd) | ~2.2 hours |
| One run (25 epochs) | ~55 hours |
| **Full sweep (7 × 15 = 105 runs)** | **~240 days** |

Three ways forward, in increasing cost:

1. **Cut the ladder to four variants** — `source`, `etf`, `rotated`,
   `cosine_matched`. Still answers the geometry question (is it the angles or
   the specific directions?) at a little over half the cost. Still too slow
   locally, but it halves a rental bill.
2. **Train over cached features.** The backbone is frozen; only LoRA adapters
   and the head train. If the adapters can be applied to cached activations
   rather than re-running the backbone each epoch, the dominant cost disappears.
   This needs a code change and is worth scoping before renting anything.
3. **Rent a GPU.** At roughly $0.40/hour for a 4090 on a spot market, the full
   sweep is plausibly **$30–60**. That is the honest number for the complete
   experiment.

The staged approach: do §1–5 locally, which costs nothing and produces the first
real geometry and placement measurements. Only rent once those say the effect is
measurable and worth the full sweep.

**A useful cheap signal first.** Run a single variant at three seeds to see
whether accuracies land near the published ~33%:

```bash
.venv/bin/python src/run_variant_experiment.py --variants source etf --seeds 3 --epochs 5
```

Even at 5 epochs this takes hours locally, but it tells you whether the setup
produces sane numbers before committing to a rental.

---

## 7. If you use the Hugging Face route rather than a local clone

`config.yaml` and `load_dino_model` assume a git clone plus a `.pth`. With the
Hugging Face weights, the scripts need to build their backbone from
`hf_backbone` instead. The change is one line in each entry point — replace

```python
backbone_factory = lambda: load_dino_model(paths["dino_repo"], paths["dino_weights"])
```

with

```python
from hf_backbone import load_hf_dinov3
backbone_factory = load_hf_dinov3
```

and pass `attention_target_modules(backbone)` to `apply_lora` in place of the
hardcoded `["qkv"]`.

Alternatively, clone the DINOv3 repository and download the `.pth` so the
existing config path works unchanged. Either is fine; the Hugging Face route is
fewer steps and the one your access grant leads to.

---

## Checklist

- [ ] `hf auth login` with `HF_HUB_DISABLE_XET=1`
- [ ] Model loads, reports `embed_dim 384`
- [ ] **Note what `attention_target_modules()` prints** — LoRA silently does
      nothing if these are wrong
- [ ] `smoke_pipeline.py --real` completes, non-zero LoRA parameters
- [ ] Codabench account, UNICORNv2 downloaded, `make check` clean
- [ ] `make features` — first real geometry report
- [ ] `make reproduce` — published numbers back, or a documented failure
- [ ] Decide: cut the ladder, train over cached features, or rent
- [ ] `make variants-real` then `make analyse`

## Things that would invalidate results

- **Reporting a stub number as a finding.** Feature files record which backbone
  produced them; check `meta['backbone']` before quoting anything.
- **LoRA attached to nothing.** A run with ~0 trainable adapter parameters
  trains only the head and tells you nothing about adaptation.
- **Mixing variants.** ViT-S/16 and ViT-S+/16 features are both 384-dim and
  otherwise indistinguishable on disk. Do not compare across them.
- **Quoting accuracies from a run that warned about validation.**
  `split_train_validation` warns when a class is too rare to appear in
  validation; macro-F1 selection is blind to those classes.

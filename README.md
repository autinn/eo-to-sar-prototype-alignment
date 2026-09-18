<div align="center">

# What Transfers? Optical-to-SAR Prototype Alignment

**A capstone investigation into the mechanism behind cross-modal transfer for SAR target recognition**

Fork of [the official code](https://github.com/LucasHirsch/eo-to-sar-prototype-alignment) for
Hirsch et al., *Cross-modal learning for SAR target recognition using optical vision
foundation models* ([arXiv:2609.07753](https://arxiv.org/abs/2609.07753), SPIE Sensors + Imaging 2026)

</div>

---

## What this is

The original work trains a SAR encoder to classify radar imagery while pulling its
features toward per-class prototypes computed from a frozen optical model. It reaches
33.3% on ten vehicle classes against 26.9% for a frozen baseline, and asks the right
follow-up question: does this work because optical prototypes carry transferable
semantic structure, or because pulling features toward any well-separated targets
regularises the feature space?

**This fork investigates that question.** It adds a measurement harness, a corrected
experimental protocol, and a statistical re-analysis of the published results — all
purely additive. No upstream file is modified, so the authors' reproduction runs
unchanged and the fork stays diffable against their repository.

Two findings are already established and need no compute:

- The proposed method is **not significantly better than its own label-free baseline**
  (1.5 pp, t = 1.63, p = 0.144). That comparison had 37% power at five seeds; fifteen
  are needed.
- The published control **varies three things at once** — prototype type, learning rate,
  and seed set — and model selection reads the test set.

For the full story, see **[WRITE_UP.md](WRITE_UP.md)**. For what happens next, see
**[ROADMAP.md](ROADMAP.md)**.

## Quickstart

Everything below runs with **no model weights and no real data**:

```bash
make setup      # create .venv, install dependencies (~850 MB)
make offline    # tests, statistics, figures, end-to-end pipeline check
```

That produces the statistical analysis, three figures, and a validated pipeline run
in a few minutes. `make help` lists every target; `make check` reports what is
available and what is blocked.

## Repository layout

```
├── README.md                 you are here
├── WRITE_UP.md               the full story: motivation, gap, method, findings
├── ROADMAP.md                what remains, phased against the project calendar
├── UPSTREAM_README.md        the original authors' README, unmodified
├── Makefile                  pipeline entry points
│
├── src/
│   │  ── upstream, unmodified ──
│   ├── train_frozen_dino.py              linear probe baseline
│   ├── train_sar_only.py                 LoRA, no alignment
│   ├── train_mmd_alignment.py            global distribution matching (no labels)
│   ├── train_eo_prototype_alignment.py   the proposed method
│   ├── train_synthetic_prototype_alignment.py   the ETF control
│   ├── model_utils.py, evaluation.py, utils.py, losses.py
│   │
│   │  ── this fork: measurement ──
│   ├── geometry.py           prototype geometry descriptors
│   ├── prototype_variants.py the ablation ladder
│   ├── error_placement.py    where alignment residual lands
│   ├── stats_utils.py        significance, effect size, power
│   │
│   │  ── this fork: infrastructure ──
│   ├── feature_simulator.py  synthetic features with controllable geometry
│   ├── stub_backbone.py      stand-in for the gated DINOv3 weights
│   ├── make_fake_unicorn.py  synthetic UNICORNv2-shaped chips
│   ├── extract_and_save_features.py   feature persistence
│   │
│   │  ── this fork: pipeline ──
│   ├── run_reported_stats.py     analyse the published numbers
│   ├── run_reproduction.py       reproduce the five published methods
│   ├── run_variant_experiment.py the corrected experiment
│   ├── analyse_experiment.py     analyse its output
│   ├── smoke_pipeline.py         end-to-end check
│   └── make_plots.py             figures
│
├── tests/                    200 tests, analytic expected values
├── analysis/                 generated artifacts and written findings
│   ├── README.md             harness documentation
│   ├── CONFOUNDS.md          the design audit
│   ├── paper_reported_results.yaml   transcribed published figures
│   ├── reported_stats.{json,csv}     the statistical re-analysis
│   └── *.png                 figures
│
├── data/                     datasets (gitignored)
└── outputs/                  training outputs and cached features (gitignored)
```

## Setup

Requires Python 3.12 and [uv](https://github.com/astral-sh/uv).

```bash
make setup
```

Or manually:

```bash
export HF_HUB_DISABLE_XET=1     # Hugging Face's chunk cache doubles weight storage
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt -r requirements-analysis.txt
```

The virtualenv is about 850 MB, dominated by PyTorch. Nothing in the offline pipeline
downloads model weights.

### For the online pipeline

Two external dependencies, both gated:

| What | How | Size |
|---|---|---|
| DINOv3 ViT-S/16+ weights | Request access from Meta, then clone [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) | ~86 MB |
| UNICORNv2 EO/SAR chips | [Codabench MAVIC-C](https://www.codabench.org/forums/12351/) (account required) | ~2 GB |

Then copy `config-example.yaml` to `config.yaml` and fill in the paths. `make check`
verifies them.

Chips are small — 55×55 SAR and 31×31 EO — so the full dataset is about 2 GB and a
ten-image-per-class subset is under a megabyte.

## Usage

### Offline — no weights, no data

```bash
make test        # 200 tests
make stats       # statistical re-analysis -> analysis/reported_stats.{json,csv}
make plots       # three figures -> analysis/*.png
make smoke       # end-to-end pipeline on stub backbone + synthetic chips
make variants    # variant experiment protocol check
make offline     # all of the above
```

### Online — needs weights and data

```bash
make features       # extract and cache real features (~660 MB per split)
make reproduce      # run the five published methods, compare to the paper
make variants-real  # the corrected experiment, 15 seeds
make analyse        # correlate placement and geometry against outcome
make online         # reproduce -> experiment -> analysis
```

## Working without the gated model

DINOv3 access takes time, and the dataset alone is useless without it. Rather than
wait, `stub_backbone.py` implements the same contract the rest of the code depends on
— `.embed_dim`, a `(batch, embed_dim)` forward pass, and `qkv` submodules for LoRA to
attach to. Every code path downstream of the backbone therefore runs and is tested
today, and swapping `load_stub_model` for `load_dino_model` is the only change needed
when access arrives.

It is a structural stand-in, not a model of DINOv3. **No number computed against it is
a finding**, and saved feature files record which backbone produced them because stub
and real features are identical in shape and dtype. Re-run `make smoke` the day access
is granted, before trusting anything.

## On the tests

Numerical assertions are analytic rather than golden — the expected value is derived
from the mathematics, not recorded from a previous run:

- A simplex ETF has pairwise cosine exactly `-1/(K-1)` (uniform to 8.3e-17)
- A random rotation preserves the Gram matrix (to 7.8e-16)
- Residual confined to singular direction `i` concentrates at exactly `D·w_i`
- Residual in the classifier's nullspace perturbs the logits by ~1e-15

This caught four bugs that returned plausible numbers rather than failing — including a
concentration metric that reported 0.99 where the true value was 0.0, and an
interpretation branch that concluded "the advantage travels with the angular geometry"
from two zero-sized gaps. Both would have produced confident, wrong claims.

## Scope

The statistical findings and the design audit are claims about the published work,
verified against its reported numbers and its source code. Everything else is
instrumentation, validated on synthetic data. **No claim about SAR is supported until
the harness runs against real features.**

## Citation

The original work:

```bibtex
@misc{hirsch2026crossmodal,
  title         = {Cross-modal learning for SAR target recognition using optical vision foundation models},
  author        = {Lucas Hirsch and James R. Hopgood and Javid Khan and Yoann Altmann and Mike E. Davies},
  year          = {2026},
  eprint        = {2609.07753},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2609.07753}
}
```

The error-placement analysis adapts a concept (not code) from work on cross-model KV
cache transfer, [arXiv:2608.03893](https://arxiv.org/html/2608.03893).

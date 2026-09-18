# Analysis harness

An additive extension to this fork, asking a question the original experiments
raise but do not settle: **when prototype alignment helps, what is doing the
work?**

Nothing in the upstream code is modified. Every file here is new, so the fork
stays diffable against the authors' repository and their reproduction runs
unchanged.

## Why

The paper aligns SAR features to per-class optical prototypes and asks whether
the gain comes from optical structure or merely from having well-separated
alignment targets. Its control swaps real class means for a simplex equiangular
tight frame. Three things about that comparison motivated this work.

**The load-bearing comparison is the one that fails.** Against the paper's own
MMD baseline - which discards EO labels entirely and only matches global feature
distributions - the proposed method gains 1.5 pp, t = 1.63, p = 0.144. The
comparison that would show *class-specific* structure transfers is not
significant, and it had 37% power at five seeds, so its null result is weak
evidence rather than evidence of no effect. Fifteen seeds per arm would be needed.
Every comparison involving MMD is similarly underpowered.

**The control varies three things at once.** Prototype type, learning rate
(8e-6 against 4e-6) and seed set (`[10,40,41,42,43]` against
`[30,42,123,777,2025]`, overlapping only at 42). All three verified against the
source; see `CONFOUNDS.md`.

**Reconstruction may be the wrong objective.** The alignment loss,
`1 - cos(sar, prototype)`, asks only that the residual be small. Work on
cross-model KV cache transfer ([arXiv 2608.03893](https://arxiv.org/html/2608.03893))
found for a closely analogous problem that reconstruction quality did not predict
downstream performance (r = -0.20) while *error placement* did (r = +0.57).
`error_placement.py` ports that idea, with the classifier head playing the role
of the attention query matrix. Concept only - no code is shared with that work.

## Layout

| File | Purpose |
|---|---|
| `src/stats_utils.py` | Welch, Cohen's d, bootstrap, permutation, power analysis |
| `src/geometry.py` | Cosine spread, effective rank, participation ratio, ETF deviation |
| `src/prototype_variants.py` | The ablation ladder |
| `src/error_placement.py` | Where alignment residual lands relative to what the head reads |
| `src/feature_simulator.py` | Synthetic features with controllable geometry |
| `src/stub_backbone.py` | Stand-in for the gated DINOv3 weights |
| `src/make_fake_unicorn.py` | Synthetic UNICORNv2-shaped data |
| `src/extract_and_save_features.py` | Feature persistence, which upstream lacks |
| `src/run_reported_stats.py` | Analysis of the published numbers |
| `src/run_variant_experiment.py` | The corrected experiment |
| `src/smoke_pipeline.py` | End-to-end check |
| `src/make_plots.py` | Figures |

## The ablation ladder

Real class means and an equiangular frame differ in several properties at once,
so a gap between them identifies none of them. Each variant holds some fixed and
breaks others:

| variant | angular geometry | class identity | optical directions |
|---|---|---|---|
| `source` | kept | kept | kept |
| `rotated` | kept | kept | **destroyed** |
| `label_permuted` | kept | **destroyed** | kept |
| `cosine_matched` | kept | **destroyed** | **destroyed** |
| `mean_shuffled` | kept\* | **destroyed** | **destroyed** |
| `centered_shuffled` | **destroyed** | **destroyed** | **destroyed** |
| `gaussian` | **destroyed** | **destroyed** | **destroyed** |
| `etf` | uniform | **destroyed** | **destroyed** |

`rotated` is the variant the published control cannot express: it preserves every
pairwise angle while moving the prototypes to an arbitrary orientation. If the
gain survives rotation it comes from geometry; if it vanishes, the specific
optical directions matter.

\* `mean_shuffled` was written expecting it would destroy angular structure. It
does not - measured shifts are under 0.001, because the correlation between class
means lives in the shared component of each coordinate and permuting within a
coordinate preserves it. Kept as a control in its own right, with
`centered_shuffled` added to do the job intended.

## Running it

```bash
uv venv .venv --python 3.12
uv pip install -r requirements.txt -r requirements-analysis.txt
```

Everything below runs **without DINOv3 weights or UNICORNv2**:

```bash
.venv/bin/python -m pytest tests/ -q          # 182 tests
.venv/bin/python src/run_reported_stats.py    # the published-results analysis
.venv/bin/python src/make_plots.py            # three figures
.venv/bin/python src/make_fake_unicorn.py --per-class 10
.venv/bin/python src/smoke_pipeline.py
.venv/bin/python src/run_variant_experiment.py --stub --data-root data/fake --seeds 3 --epochs 2
```

With real weights and data, drop `--stub` and `--data-root`, and set the paths in
`config.yaml`:

```bash
.venv/bin/python src/run_variant_experiment.py --seeds 15
```

## On the stub backbone

DINOv3 ViT-S/16+ is gated. `stub_backbone.py` satisfies the same contract -
`.embed_dim`, `(batch, embed_dim)` output, `qkv` submodules for peft - so every
path downstream of the backbone can be developed and tested before access lands.
Swapping `load_stub_model` for `load_dino_model` is then the only change.

It is a structural stand-in, not a model of DINOv3. **No result computed against
it says anything about SAR.** Saved feature files record which backbone produced
them, because stub and real features are identical in shape and dtype and only
provenance separates them. Re-run the smoke test on the day access is granted,
before trusting any number.

## On the tests

The numerical assertions are analytic rather than golden: the ETF cosine is
exactly `-1/(K-1)`, rotation preserves the Gram matrix, residual confined to
singular direction `i` concentrates at exactly `D * w_i`, and residual in the
head's nullspace perturbs the logits by ~1e-15.

That mattered. Three bugs were caught this way, each of which returned a
plausible number rather than failing:

- `residual_concentration` normalized over the read subspace alone, returning
  ~0.99 for nullspace residual whose true value is 0.0.
- The forest plot drew normal 1.96 intervals against a Welch *t* test, so three
  comparisons appeared to clear zero while being marked non-significant.
- The geometry figure drew near-orthogonal source prototypes, understating the
  span the ladder covers; real class means from a shared encoder sit near 0.94.

## Scope

The statistical findings and the design confounds are claims about the published
work, verified against its numbers and its source. Everything else is
instrumentation, validated on synthetic data. No claim about SAR is supported
until the harness runs against real features.

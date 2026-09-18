# What Transfers? Isolating the Mechanism of Optical-to-SAR Prototype Alignment

**Capstone project proposal — Satellite Data Analysis and Applications track**
Autin Au-Yeung · 18 September 2026

---

## 1. Motivation

Synthetic aperture radar sees what optical sensors cannot. It penetrates cloud,
works at night, and responds to surface roughness and dielectric properties
rather than reflected light. For the applications that matter most urgently —
flood mapping during a storm, damage assessment on the night of an earthquake,
detecting vessels that have switched off their transponders — SAR is often the
only modality returning usable imagery at the moment the imagery is needed.

The obstacle is labels. A radar image of a truck does not look like a truck; it
is a pattern of bright and dark returns that a trained analyst interprets. Optical
imagery is labelled at enormous scale because anyone can annotate a photograph.
The result is a field with abundant optical supervision and comparatively little
SAR supervision, and a natural response: borrow what large vision models have
learned from photographs and point it at radar.

That response does not work directly. Models pretrained on optical imagery
degrade sharply on SAR, and the reasons are physical rather than incidental.
Coherent imaging produces multiplicative speckle that corrupts the local texture
statistics convolutional and attention features rely on. Side-looking geometry
introduces layover, foreshortening and shadow, which alter object topology in
ways no optical training distribution contains. And backscatter is semantically
ambiguous in a way brightness is not: two different materials can return
identically dark, while one material can look entirely different depending on
moisture content or row orientation. The DisasterM3 benchmark authors name
"significant optical-SAR performance degradation" among their open problems, and
the OmniEarth benchmark finds remote-sensing-specialised vision-language models
scoring 25–45% at the task of identifying whether an image is optical or SAR at
all — below general-purpose models, and near chance.

Against that background, Hirsch et al. (arXiv:2609.07753, SPIE Sensors + Imaging
2026) propose something appealingly simple. Take a frozen optical foundation model
(DINOv3 ViT-S/16+), compute the average optical feature for each vehicle class to
form a set of *class prototypes*, then train a SAR encoder with LoRA to classify
SAR chips while also pulling each SAR feature toward its class's optical
prototype. At inference the model sees only SAR. On the UNICORNv2 ten-class
vehicle dataset this reaches 33.3% accuracy against 26.9% for a frozen baseline.

The authors then ask the right follow-up question, and it is the question this
capstone takes up. Alignment forces features toward ten well-separated target
points. **Pushing classes apart in feature space improves classification whether
or not those points mean anything.** So does the method work because optical
prototypes carry transferable semantic structure, or because the alignment term
acts as a geometric regulariser that any separated targets would provide?

Their control replaces optical prototypes with a simplex equiangular tight frame
— ten maximally-separated points carrying no optical information whatsoever. The
frame reaches 29.3% against the optical prototypes' 33.3%, and they conclude the
improvement "is not explained only by latent space regularization, but is related
to structure encoded in the EO prototypes."

The gap this capstone addresses is that the released experiments do not establish
that conclusion, and the reasons are specific and fixable.

## 2. Literature review and gap analysis

### 2.1 What the reported numbers support

Working from the paper's reported means and standard deviations over five seeds,
I computed all ten pairwise comparisons (Welch's unequal-variance t-test,
Cohen's d, achieved power; `src/run_reported_stats.py`, output in
`analysis/reported_stats.json`). Conclusions are unchanged whether the published
standard deviations are read as sample or population values, so nothing below
depends on that ambiguity.

| Comparison | Δ (pp) | t | p | power at n=5 | seeds for 80% |
|---|---|---|---|---|---|
| EO prototype − frozen | +6.40 | 10.88 | 0.0003 | 1.00 | 1 |
| EO prototype − ETF control | +4.00 | 3.75 | 0.0074 | 0.96 | 3 |
| EO prototype − SAR-only | +3.50 | 4.10 | 0.0035 | 0.98 | 3 |
| **EO prototype − MMD** | **+1.50** | **1.63** | **0.144** | **0.37** | **15** |
| ETF control − SAR-only | +0.50 | 0.46 | 0.661 | 0.07 | 188 |

Two rows carry the argument.

**The fourth row is the one that matters.** The paper's own MMD baseline aligns
the *global* distribution of SAR features to the global distribution of optical
features. It explicitly discards optical labels — in the released code the EO
label is bound to `_` and never used. It therefore cannot transfer any
class-specific information, only a generic sense of what optical features look
like. It reaches 31.8%. The proposed class-conditional method reaches 33.3%, and
that 1.5-point difference does not reach significance.

This is the comparison that would demonstrate *class* structure transfers. The
ETF control rules out pure geometry; it cannot rule out generic optical-ness.
Both remaining explanations — class semantics, and modality-generic optical
structure — predict beating the ETF. Only one predicts beating MMD.

**The comparison was also badly underpowered.** At five seeds it had 37% power
against an effect of the observed size: more likely to miss the effect than find
it. Its null result is therefore weak evidence rather than evidence of no effect.
Fifteen seeds per arm would be needed for a conventional 80% chance. Notably,
*every* comparison involving MMD sits between 0.37 and 0.59 power, so MMD's
position in the ranking is unresolved in both directions — it is statistically
indistinguishable from the method above it and from the baseline below it.

The last row is a genuine and useful negative result the paper does not emphasise:
prototype geometry alone buys nothing (+0.5 pp, p = 0.661). Well-separated targets
by themselves do not help.

### 2.2 Three design confounds

Reading the released code against the reported procedure surfaced three issues,
each verified against source and documented in `analysis/CONFOUNDS.md`.

**The control varies three things at once.** Comparing
`train_eo_prototype_alignment.py` against `train_synthetic_prototype_alignment.py`,
the training loops are byte-identical except for a single identifier
(`eo_prototypes[labels]` becomes `synthetic_prototypes[labels]`). But the scripts
differ in learning rate (8e-6 against 4e-6) and in seed set
(`[10,40,41,42,43]` against `[30,42,123,777,2025]`, overlapping only at 42). The
4.0-point gap cannot be attributed to prototype type alone. The learning-rate
difference is not cosmetic: an ETF places targets at maximum mutual separation
while real class means are highly correlated, so the alignment gradient has a
different scale in the two conditions, and step size interacts with it.

**Non-overlapping seeds forfeit statistical power.** Because the two conditions
share only one seed, per-seed pairing is impossible and only unpaired tests are
available. Given seed-to-seed standard deviations of 1.3 to 2.0 points against
effects of 1.5 to 4.0, this is a substantial loss at no compute saving.

**Model selection reads the test set.** Every training script evaluates on
`test_loader` each epoch and retains the best macro-F1; `config-example.yaml`
defines only `train_sar`, `train_eo` and `test_sar`, with no validation path. The
paper describes retaining "the checkpoint with the highest validation macro-F1",
suggesting a held-out split was intended. As released, reported accuracies are
optimistic by an unknown margin. The effect is likely modest over 25 epochs on a
balanced 770-image test set, and applies to all methods equally so the ranking is
probably unaffected — but the absolute numbers are best-epoch-on-test.

### 2.3 A second framing, from an unrelated literature

Recent work on cross-model KV cache transfer (arXiv:2608.03893) addresses a
structurally identical problem: making one model's internal representations
usable by another that was never trained to share its space. The relevant finding
is counter-intuitive. Reconstruction quality — how close the mapped
representation lands to the target — **did not** predict downstream performance
(r = −0.20). What did predict it (r = +0.57) was *where the residual error
landed*: error falling in directions the downstream computation reads is
damaging, while error of identical magnitude in directions it ignores is
harmless. Their successful fix worked by relocating error, not shrinking it.

The prototype alignment objective, `1 − cos(sar_feature, prototype)`, is a pure
reconstruction loss — precisely the quantity that work found to be the wrong
target. The analogy transfers cleanly: there the reference geometry came from the
attention query matrix, here it comes from the classifier head, which is the only
consumer of the aligned features. A head of shape (10, 384) reads at most ten
directions and is blind to the other 374.

This yields a third hypothesis neither paper considers: **alignment may help
because optical prototypes happen to place residual error where the classifier
does not look.** The ETF control cannot distinguish this, because random
separated points differ from optical prototypes in both semantics *and* error
geometry.

Preliminary measurements on the validated pipeline (stub backbone, so
illustrative only) show two variants with nearly identical residual magnitude
(0.0740 against 0.0721 RMS) but different placement concentration (1.020 against
1.314), and roughly 97% of alignment residual landing in the classifier's
nullspace. The cosine objective sees only the magnitude.

### 2.4 Positioning

The SAR foundation-model field has moved quickly — CrossEarth-SAR, SARATR-X,
SARCLIP and SARVLM all postdate the framing of this problem, and the general
approach of adapting optical models to SAR with parameter-efficient fine-tuning
is now well populated. This capstone deliberately does not compete there. Adding
another adaptation method to a crowded field is less valuable than establishing
whether the mechanism the field assumes is actually operating. The published
best result is 33.3% on ten classes; understanding why a method barely works is
more tractable and more useful than marginally improving it.

### 2.5 Research question

> **When optical-to-SAR prototype alignment improves SAR classification, what
> property of the optical prototypes is responsible — their class-specific
> semantic content, their angular geometry, or the placement of the residual
> error they induce relative to the classifier's read subspace?**

Three sub-questions, each separately answerable:

1. Does the advantage over label-free distribution matching survive adequate
   statistical power?
2. Does the advantage survive rotation of the prototypes — which preserves every
   pairwise angle while destroying correspondence to specific optical directions?
3. Does residual placement predict downstream accuracy better than residual
   magnitude, as it does in the cross-model transfer setting?

## 3. Planned deliverables

**D1 — Statistical re-analysis of the published results.** All pairwise
comparisons with effect sizes, power, and the seed counts required to resolve
each. *Complete*: `analysis/reported_stats.json`, `analysis/reported_stats.csv`,
`analysis/power_analysis.png`, `analysis/comparison_forest.png`.

**D2 — Design audit.** The three confounds, verified against source, with a
corrected protocol. *Complete*: `analysis/CONFOUNDS.md`.

**D3 — A tested measurement harness.** Prototype geometry descriptors, the
ablation ladder, the error-placement diagnostic, a controllable feature
simulator, and feature persistence (which the upstream code lacks entirely).
*Complete*: 182 tests with analytic expected values.

**D4 — Reproduction.** The five published methods at published settings, with
reported accuracies confirmed to fall within seed variance. *Blocked on model and
data access.*

**D5 — The corrected experiment.** The full ablation ladder under one learning
rate, one seed set, validation-based model selection, and 15 seeds per arm, with
paired per-seed comparisons. *Protocol complete and tested
(`src/run_variant_experiment.py`); execution blocked.*

**D6 — Mechanism analysis.** Residual placement reported alongside accuracy for
every variant, testing whether placement predicts outcome. *Spring.*

**D7 — Written thesis and defense.**

## 4. Methodology and key variables

### 4.1 The ablation ladder

The central methodological contribution. Real class means and an equiangular
frame differ in several properties simultaneously, so a gap between them
identifies none of them. Each variant holds some properties fixed and breaks
others (`src/prototype_variants.py`):

| Variant | Angular geometry | Class identity | Optical directions |
|---|---|---|---|
| `source` (real EO means) | kept | kept | kept |
| `rotated` | kept | kept | **destroyed** |
| `label_permuted` | kept | **destroyed** | kept |
| `cosine_matched` | kept | **destroyed** | **destroyed** |
| `centered_shuffled` | **destroyed** | **destroyed** | **destroyed** |
| `gaussian` | **destroyed** | **destroyed** | **destroyed** |
| `etf` (published control) | uniform | **destroyed** | **destroyed** |

`rotated` is the variant the published control cannot express. A random rotation
is an isometry: it preserves the Gram matrix exactly (verified to 7.8e-16) while
moving every prototype to an arbitrary orientation. If the advantage survives
rotation, it comes from geometry. If it vanishes, the specific optical directions
matter — which is the strong form of the semantic-transfer claim.

### 4.2 Error placement

Given a classifier head `W` of shape (K, D), take its singular value
decomposition to obtain right singular vectors `V` and singular values `s`. For
alignment residual `E = f_sar − p_label`, project onto the read directions,
measure energy per direction `p_i`, and compute

> `concentration = Σ_i w_i p_i / (E_total / D)`, with `w_i = s_i² / Σ_j s_j²`

The denominator is the per-direction energy of an isotropic residual of the same
total magnitude, which makes the measure scale-free — it reports placement
independent of size. Reference values are analytic and asserted in tests:
isotropic residual gives 1.0, residual in the head's nullspace gives 0.0, and
residual confined to singular direction `i` gives exactly `D·w_i`.

### 4.3 Key variables

**Independent.** Prototype variant (7 levels); random seed (15 levels);
alignment weight; modality gap and shared structure in simulation.

**Dependent.** Ten-class and seven-class accuracy and macro-F1 (the seven-class
metric merges the four light-vehicle classes, which have similar SAR signatures);
residual concentration; nullspace energy fraction; residual RMS; logit
perturbation.

**Controlled.** Learning rate (8e-6, fixed across all conditions — correcting
confound 1); seed set (shared — correcting confound 2); LoRA rank 28, alpha 56,
`qkv` targets, dropout 0.05; 25 epochs; batch size 256; inverse-frequency
weighted sampling against the ~1000:1 class imbalance; model selection on a
stratified validation split (correcting confound 3).

**Measured but not controlled.** Prototype geometry (pairwise cosine mean and
spread, effective rank, participation ratio, isotropy, ETF deviation) — reported
per variant so outcome differences can be related to geometric differences.

## 5. Traceability analysis

Following the NASA science traceability matrix structure, which flows from goal
through objective and measurement requirement to instrument and data product.

### Goal

**G.** Determine what property of optical class prototypes is responsible when
optical-to-SAR alignment improves SAR target recognition.

| # | Objective | Measurement objective | Measurement requirement | Instrument | Data product |
|---|---|---|---|---|---|
| **O1** | Establish what the published evidence supports | All pairwise comparisons with effect size and power | Welch t-test; power to resolve a 1.5 pp gap at pooled sd 1.46; conclusions stable across both std conventions | `stats_utils.py`, `run_reported_stats.py` | `reported_stats.json`, `power_analysis.png`, `comparison_forest.png` |
| **O2** | Determine whether the control isolates prototype type | Compare the two scripts line by line for every differing constant | Every difference identified and verified against source, not inferred | Source reading; `diff` | `CONFOUNDS.md` |
| **O3** | Establish that measurements are trustworthy before use | Each numerical function checked against an analytically known value | ETF cosine = −1/(K−1) to 1e-6; rotation preserves Gram to 1e-6; nullspace residual gives 0.0 and perturbs logits < 1e-10; single-direction residual = D·w_i to 1e-6 | pytest suite | 182 passing tests |
| **O4** | Reproduce the published results | Five methods at published settings, 5 seeds | Reported accuracies within seed variance | Upstream training scripts, unmodified | Reproduction table |
| **O5** | Determine whether the advantage over label-free alignment is real | EO prototypes against MMD, adequately powered | ≥15 seeds per arm; one learning rate; shared seed set; paired per-seed test | `run_variant_experiment.py` | `variant_experiment_dinov3.json` |
| **O6** | Determine whether angular geometry alone suffices | `source` against `rotated` | Rotation verified isometric; identical protocol; paired per-seed test | `prototype_variants.rotated`, `geometry.py` | Paired comparison table |
| **O7** | Determine whether class identity is required | `source` against `label_permuted` and `cosine_matched` | Multiset of pairwise angles preserved; only assignment broken | `prototype_variants` | Paired comparison table |
| **O8** | Determine whether placement predicts outcome better than magnitude | Concentration and residual RMS against accuracy across all variant-seed runs | Correlation across ≥105 runs; placement measured on the test set after selection | `error_placement.py` | Placement-outcome correlation |
| **O9** | Characterise method behaviour under known geometry | Sweep separation, modality gap and shared structure in simulation | Each control verified to move the corresponding measured property | `feature_simulator.py` | Simulation sweeps |

### Requirements flowing from this

- **R1 (from O5).** 15 seeds × 7 variants × 25 epochs. Exceeds the M1's capacity;
  requires either training over cached features or rented GPU (~$30–60).
- **R2 (from O4, O5).** DINOv3 ViT-S/16+ weights. Mitigated for development by
  `stub_backbone.py`; **not** mitigable for results.
- **R3 (from O4–O8).** UNICORNv2 via Codabench (~2 GB; chips are 55×55 SAR and
  31×31 EO). Mitigated for development by `make_fake_unicorn.py`.
- **R4 (from O5, O8).** Feature persistence, since re-extracting over 455,635
  images per analysis is infeasible. ~660 MB per split.
- **R5 (from O3).** Every reported number traceable to a tested function. No
  result computed against the stub backbone may be reported as a finding.

## 6. Timeline

Fall dates are fixed by the track; spring is provisional.

| Week | Dates | Milestone | Status |
|---|---|---|---|
| 1–3 | 1–18 Sep | Literature review, statistical analysis, harness | **Complete** |
| 3 | 18–25 Sep | Codabench registration, UNICORNv2 download | **Critical path** |
| 4 | **1 Oct** | **Detailed Project Proposal** | This document |
| 4–5 | Oct | Reproduction of published results (D4) | Blocked on R2 |
| 5–6 | Oct | Feature extraction and caching; first real geometry report | Blocked on R2, R3 |
| **7** | Oct | **Fall Committee Meeting** | |
| 7–10 | Oct–Nov | Corrected experiment, 15 seeds (D5) | Needs R1 |
| 10–11 | Nov | Analysis, figures, draft | |
| **11** | **20 Nov** | **Rough Draft** | |
| 12–13 | Nov–Dec | Placement-outcome analysis (D6) | |
| **14** | **10 Dec** | **Fall video (~5 min)** | |
| **15** | Dec | **Fall Oral Defense** | |
| Spring | Jan–Apr | Mechanism work; placement-aware objective if warranted | Provisional |
| Spring | Apr | Final thesis and defense | |

The fall arc is diagnosis; the spring arc attempts a fix. This split is
deliberate: the diagnosis is valuable whatever it finds, so the project does not
depend on the intervention succeeding.

### Contingencies

Detailed in `ROADMAP.md`. The main ones: if DINOv3 access is denied, substitute
an ungated backbone (DINOv2, CLIP ViT-B/16) — the method is not DINOv3-specific,
only the reproduction is. If compute is short, cut the ladder to four variants,
which still answers the geometry question at a quarter the cost. If the effect
vanishes entirely under the corrected protocol, that is the strongest available
result, not a failure.

## 7. Why this is feasible

Three months of fall term is short for an empirical project, and the honest
argument for feasibility rests on what is already done rather than what is
planned.

The measurement infrastructure exists and is tested (182 tests, analytic expected
values). The experimental protocol is written and runs end to end. Two findings
are established and need no further compute. The remaining work is execution
against real data, not construction.

The data is small — 55×55 and 31×31 chips, about 2 GB for the full dataset, which
is unusual in remote sensing and makes this tractable on modest hardware. The
model is 21M parameters with a frozen backbone and LoRA adapters.

The principal risk is access rather than difficulty, and both blockers have
development mitigations in place: the pipeline has been validated end to end
against a stub backbone and synthetic data, so the day access arrives the work is
execution rather than development.

---

## Appendix A — PLO/HC/LO mapping

*To be completed with the specific learning outcomes for the concentration.*
Anticipated: `#professionalism` and `#composition` throughout; `#evidencebased`
and `#studyreplication` in §2; `#descriptivestats`, `#significance` and
`#confidenceintervals` in §2.1; `#designthinking` and `#constraints` in §4.1 and
§5; `#algorithms` and `#optimization` in §4.2; `#modeling` in §4.3;
`#sourcequality` in §2.4; `#gapanalysis` in §2.2; `#audience` in §7.

## Appendix B — Second reader

Suitable expertise spans three areas, in rough priority order: (i) experimental
methodology and statistical power in machine learning, which is where the fall
contribution sits; (ii) SAR or remote sensing physics, for the modality-gap
framing in §1; (iii) representation learning or model interpretability, for the
error-placement analysis in §4.2. A reader with (i) would be most valuable, since
the central claims are methodological.

Contacting the source paper's authors is also worth considering — the analysis is
a constructive critique of their released work, and they flagged the underlying
question themselves.

## Appendix C — Reproducibility

All code, data generation and analysis are in this repository and run without the
gated model or dataset:

```bash
uv venv .venv --python 3.12
uv pip install -r requirements.txt -r requirements-analysis.txt
.venv/bin/python -m pytest tests/ -q                       # 182 tests
.venv/bin/python src/run_reported_stats.py                 # §2.1 table
.venv/bin/python src/make_plots.py                         # figures
.venv/bin/python src/make_fake_unicorn.py --per-class 10
.venv/bin/python src/smoke_pipeline.py                     # end-to-end
```

The work is purely additive: no upstream file is modified, so the authors'
reproduction is unaffected and the fork remains diffable against their
repository.

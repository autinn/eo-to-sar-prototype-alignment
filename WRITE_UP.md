# What Transfers?

*The story of this repository — why it exists, what it found, and what it does.*

---

## 1. The problem SAR creates

Radar sees through cloud and works at night. For flood mapping during a storm,
damage assessment on the night of an earthquake, or finding vessels that have
switched off their transponders, synthetic aperture radar is often the only
sensor returning usable imagery at the moment it is needed.

The difficulty is labels. A radar image of a truck does not look like a truck.
It is a pattern of bright and dark returns produced by surface roughness and
dielectric properties, and reading it takes training. Photographs, by contrast,
are labelled at enormous scale because anyone can annotate one. The field
therefore has abundant optical supervision and comparatively little SAR
supervision.

The obvious response is to borrow. Large vision models have absorbed a great deal
from photographs; point that knowledge at radar and the labelling problem
softens. Except it does not work directly, and the reasons are physical rather
than incidental. Coherent imaging produces multiplicative speckle that corrupts
the local texture statistics learned features rely on. Side-looking geometry
introduces layover, foreshortening and shadow, changing object topology in ways
no optical training distribution contains. And backscatter is ambiguous where
brightness is not — two materials can return identically dark, and one material
can look entirely different depending on moisture or row orientation.

The scale of the gap is easy to underestimate. The OmniEarth benchmark finds that
remote-sensing-specialised vision-language models score 25–45% at simply
identifying whether an image is optical or SAR — worse than general-purpose
models, and near chance. The DisasterM3 authors list "significant optical-SAR
performance degradation" among their open problems.

## 2. An appealing idea, and the question it raises

Hirsch et al. propose something simple. Take a frozen optical foundation model,
DINOv3 ViT-S/16+. Compute the average optical feature for each vehicle class —
one *prototype* per class. Then train a SAR encoder with LoRA to classify SAR
chips while also pulling each SAR feature toward its class's optical prototype.
At inference the model sees only radar.

On UNICORNv2, ten classes of vehicle, this reaches 33.3% against 26.9% for a
frozen baseline. Modest in absolute terms — SAR vehicle classification is hard —
but a real improvement.

The authors then ask the question that makes this interesting. The alignment term
pushes features toward ten well-separated points. **Separating classes in feature
space improves classification whether or not those points mean anything.** So is
the method transferring optical semantics, or is it a geometric regulariser
wearing a semantic costume?

They run the right control: replace optical prototypes with a simplex equiangular
tight frame — ten maximally separated points containing no optical information at
all. The frame gets 29.3%, the optical prototypes 33.3%, and they conclude the
improvement "is not explained only by latent space regularization, but is related
to structure encoded in the EO prototypes."

That conclusion is plausible. This repository exists because the released
experiments do not establish it.

## 3. What the numbers actually support

Working from the paper's own reported means and standard deviations, I computed
every pairwise comparison — Welch's t-test, effect size, and crucially the
achieved power.

| Comparison | Δ (pp) | t | p | power at n=5 | seeds for 80% |
|---|---|---|---|---|---|
| EO prototype − frozen | +6.40 | 10.88 | 0.0003 | 1.00 | 1 |
| EO prototype − ETF control | +4.00 | 3.75 | 0.0074 | 0.96 | 3 |
| EO prototype − SAR-only | +3.50 | 4.10 | 0.0035 | 0.98 | 3 |
| **EO prototype − MMD** | **+1.50** | **1.63** | **0.144** | **0.37** | **15** |
| ETF control − SAR-only | +0.50 | 0.46 | 0.661 | 0.07 | 188 |

The fourth row is where the argument turns.

The paper includes an **MMD baseline** that aligns the *global* distribution of
SAR features to the global distribution of optical features. It throws optical
labels away entirely — in the code the label is bound to `_` and never used. It
can carry no class-specific information whatsoever, only a generic sense of what
optical features look like. It reaches 31.8%.

The proposed class-conditional method reaches 33.3%. **That difference is not
significant.**

This matters because it is the comparison that would demonstrate *class*
structure transfers. The ETF control rules out pure geometry, but it cannot rule
out generic optical-ness: both surviving explanations — class semantics, and
modality-generic optical structure — predict beating the frame. Only one predicts
beating MMD.

And the comparison had **37% power**. At five seeds it was more likely to miss an
effect of the observed size than to find it, so its null result is not evidence of
no difference; it is barely evidence of anything. Every comparison involving MMD
sits between 0.37 and 0.59 power, leaving MMD statistically indistinguishable
both from the method above it and the baseline below it.

The last row is a real negative result the paper does not dwell on: prototype
geometry alone buys nothing (+0.5 pp, p = 0.661). Well-separated targets by
themselves do not help.

## 4. Three things the released code reveals

Reading the scripts against the described procedure surfaced problems that the
reported numbers alone would not show. Each is verified against source in
[`analysis/CONFOUNDS.md`](analysis/CONFOUNDS.md).

**The control varies three things at once.** The EO and synthetic training loops
are byte-identical except for a single identifier. But the scripts differ in
learning rate (8e-6 against 4e-6) and seed set (`[10,40,41,42,43]` against
`[30,42,123,777,2025]`, overlapping only at 42). The 4-point gap cannot be
attributed to prototype type alone. The learning-rate difference is not cosmetic:
an ETF places targets at maximum separation while real class means are highly
correlated, so the alignment gradient has a different scale in each condition.

**Non-overlapping seeds throw away power.** Sharing only one seed makes per-seed
pairing impossible, leaving only unpaired tests. With seed-to-seed standard
deviations of 1.3–2.0 against effects of 1.5–4.0, that is a substantial loss for
no compute saving.

**Model selection reads the test set.** Every script evaluates on `test_loader`
each epoch and keeps the best macro-F1. `config-example.yaml` defines no
validation path. The paper describes selecting on "validation macro-F1", so a
held-out split was clearly intended. As released, the reported accuracies are
best-epoch-on-test.

## 5. A second idea, from an unrelated field

While reading about cross-model KV cache transfer — making one language model's
internal state usable by another that never shared its representation space — I
found a result that reframes this problem.

The intuitive way to judge such a mapping is reconstruction: how close does the
mapped representation land to the target? That work measured it, and found
reconstruction quality **did not** predict downstream performance (r = −0.20).
What did predict it (r = +0.57) was *where the leftover error landed*. Error
falling in directions the downstream computation reads is damaging; error of
identical magnitude in directions it ignores is harmless. Their successful fix
worked by relocating error, not by shrinking it.

Now look at the alignment objective here: `1 − cos(sar_feature, prototype)`. That
is a pure reconstruction loss — precisely the quantity found to be the wrong
target.

The analogy transfers cleanly. There the reference geometry came from the
attention query matrix; here it comes from the classifier head, the only consumer
of the aligned features. A head of shape (10, 384) reads at most ten directions
and is blind to the other 374.

This yields a third hypothesis neither paper considers: **alignment may help
because optical prototypes happen to place residual error where the classifier
does not look.** The ETF control cannot distinguish this, because random
separated points differ from optical prototypes in both semantics and error
geometry simultaneously.

## 6. The research question

> **When optical-to-SAR prototype alignment improves SAR classification, what
> property of the optical prototypes is responsible — their class-specific
> semantic content, their angular geometry, or the placement of the residual
> error they induce relative to the classifier's read subspace?**

Three sub-questions, separately answerable:

1. Does the advantage over label-free distribution matching survive adequate power?
2. Does it survive *rotating* the prototypes — preserving every pairwise angle
   while destroying correspondence to specific optical directions?
3. Does residual placement predict accuracy better than residual magnitude?

## 7. How the repository answers them

### The ablation ladder

The central methodological move. Real class means and an equiangular frame differ
in several properties at once, so a gap between them identifies none of them.
Each variant holds some fixed and breaks others:

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
is an isometry — it preserves the Gram matrix exactly while moving every
prototype to an arbitrary orientation. If the advantage survives rotation it is
geometric. If it vanishes, the specific optical directions carry something, which
is the strong form of the semantic-transfer claim.

### The error-placement diagnostic

Take the classifier head `W` of shape (K, D) and decompose it. Project the
alignment residual onto its right singular vectors, weight each direction by its
squared singular value, and normalise by what an isotropic residual of the same
total size would place there:

> `concentration = Σᵢ wᵢ pᵢ / (E_total / D)`,  `wᵢ = sᵢ² / Σⱼ sⱼ²`

The normalisation makes it scale-free: it measures *placement* independent of
size. Isotropic residual gives 1.0, residual in the head's nullspace gives 0.0,
and residual on singular direction `i` gives exactly `D·wᵢ`. All three are
asserted in tests.

### The corrected protocol

One learning rate across every condition. One seed set, enabling paired per-seed
comparisons. Model selection on a stratified validation split carved from
training, with the test set touched once. Fifteen seeds per arm, the figure the
power analysis demands.

## 8. Building it without the model

DINOv3 weights are gated behind a Meta access request, and the dataset is useless
without them. Rather than wait, I built a structural stand-in: `stub_backbone.py`
implements the contract the rest of the code needs — `.embed_dim`, a
`(batch, embed_dim)` forward pass, `qkv` submodules for LoRA to attach to. Every
path downstream of the backbone therefore runs and is tested today.

Paired with `make_fake_unicorn.py`, which writes synthetic chips at the real
dimensions (55×55 SAR, 31×31 EO) into the exact directory layout, the entire
pipeline — feature extraction, prototype construction, LoRA training, evaluation,
geometry and placement analysis — was developed and debugged before any gated
resource arrived. Swapping `load_stub_model` for `load_dino_model` is the only
change needed.

The stub is a structural stand-in, not a model of DINOv3. No number computed
against it is a finding, and feature files record which backbone produced them
because stub and real features are identical in shape and dtype.

## 9. What testing caught

The tests assert analytic values — derived from the mathematics, not recorded
from a previous run. An ETF's pairwise cosine must be exactly `−1/(K−1)`. A
rotation must preserve the Gram matrix. Nullspace residual must perturb the
logits by nothing.

This was not ceremony. Four bugs were caught, and every one returned a plausible
number rather than failing:

**The concentration metric normalised over the wrong subspace.** Dividing by the
mean energy across only the ten read directions meant that residual living
entirely in the nullspace produced 0/0, and floating-point noise returned **0.992**
— for a case whose true value is exactly 0.0. That would have silently corrupted
every result built on it.

**`mean_shuffled` did not do what I documented.** I claimed it destroyed angular
structure; measurement showed it moves pairwise cosines by ±0.0004. The
correlation between class means lives in the shared component of each coordinate,
and permuting within a coordinate preserves it exactly. Documented honestly and
`centered_shuffled` added to do the intended job.

**A figure disagreed with its own test.** The forest plot drew normal 1.96
intervals while the reported test used Welch's *t*; at ~7 degrees of freedom the
critical value is nearer 2.4, so three comparisons appeared to clear zero while
being marked non-significant.

**The interpretation logic drew a conclusion from nothing.** Asked to explain a
run where both the rotated and ETF gaps were zero, it reported "the advantage
travels with the angular geometry" — a confident claim from no evidence, caused
by a guard added to prevent division by zero. It now checks there is an advantage
to explain before attributing one.

The last is the one worth dwelling on. A project whose thesis is that a field drew
a conclusion its evidence did not support is exactly the project that must not do
the same thing.

## 10. What the repository does

```
make setup      # environment
make offline    # everything that needs no gated resources
make online     # reproduction, corrected experiment, analysis
```

The offline pipeline runs the test suite, produces the statistical re-analysis and
figures, generates synthetic data, and exercises the full training and analysis
path end to end. It finishes in minutes and needs nothing external.

The online pipeline reproduces the five published methods and checks them against
the paper (so that any later difference is attributable to the protocol change
rather than the environment), runs the corrected experiment, and correlates
placement and geometry against outcome.

## 11. Where this stands

**Established, needing no compute:** the statistical re-analysis and the design
audit. Both are claims about the published work, verified against its numbers and
its source.

**Built and tested, awaiting access:** everything else. 200 tests, a validated
pipeline, and a corrected protocol ready to run.

**Not yet supported:** any claim about SAR. The instruments are validated on
synthetic data; they have not yet touched a real radar image.

The project is designed to survive its own likeliest outcome. If the effect
vanishes under the corrected protocol, that is the strongest available result —
it would mean optical prototypes carry no class-specific advantage over
label-free distribution matching, and a premise the field is building on needs
revisiting. If it survives, the ladder says *which* property carries it. Either
way the corrected protocol is the contribution, and it does not depend on the
sign of the answer.

---

*For the formal proposal with traceability matrix, see [WRITEUP.md](WRITEUP.md).
For remaining work, see [ROADMAP.md](ROADMAP.md). For navigating the code, see
[README.md](README.md).*

# Roadmap

Status as of **18 September 2026**. Dates follow the Satellite Data Analysis
track calendar; the fall deliverables are fixed, the spring ones are provisional.

## Where the project stands

The analysis harness is built and tested (182 tests). Two findings about the
published work are established and need no further compute:

- The proposed method is not significantly better than its own label-free MMD
  baseline (1.5 pp, t = 1.63, p = 0.144), and that comparison had 37% power at
  five seeds. Fifteen per arm are needed.
- The control experiment varies prototype type, learning rate and seed set
  simultaneously, and model selection reads the test set. All verified against
  source.

Everything else is instrumentation waiting on two external unblocks.

## Blockers

| Blocker | Status | Blocks | Mitigation |
|---|---|---|---|
| DINOv3 ViT-S/16+ weights (Meta gate) | Requested 18 Sep, pending | All real-feature work | `stub_backbone.py` — every code path already runs and is tested |
| UNICORNv2 via Codabench | **Not yet started** | All real-data work | `make_fake_unicorn.py` — pipeline validated on synthetic chips |

Neither blocker stops proposal work. **Register for Codabench this week** so both
clocks run in parallel; it is currently the critical path, not the model.

---

## Phase 1 — Proposal (now → 1 October)

The deliverable is the Detailed Project Proposal: literature review and gap
analysis, deliverable structure, traceability matrix, and preliminary satellite
data work.

- [x] Statistical analysis of published results (`analysis/reported_stats.json`)
- [x] Confound documentation (`analysis/CONFOUNDS.md`)
- [x] Geometry, variants, error-placement, simulator modules
- [x] Stub backbone and synthetic data, full pipeline validated
- [x] Corrected experiment protocol (`src/run_variant_experiment.py`)
- [x] Figures
- [x] Traceability matrix and proposal (`WRITEUP.md`)
- [ ] **Register for Codabench, download UNICORNv2** — critical path
- [ ] Extend the literature review: SAR foundation models (CrossEarth-SAR,
      SARATR-X, SARCLIP, SARVLM) to position this work against a field that has
      moved since the source paper
- [ ] Identify a second reader
- [ ] PLO/HC/LO appendix

**Preliminary data work for the proposal.** Once UNICORNv2 is downloaded, run
`extract_and_save_features.py --stub` on real chips. This exercises real data end
to end without the gated weights and produces genuine class-count statistics,
chip dimensions and imbalance figures for the proposal — satisfying the
"preliminary satellite data work" requirement even if the model has not landed.

## Phase 2 — Reproduction (1 October → committee meeting, week 7)

Establishes that measurements against the real system can be trusted.

- [ ] Re-run `smoke_pipeline.py` with real weights **before trusting any number**
- [ ] Reproduce the five published methods at their published settings; confirm
      the reported accuracies fall within seed variance
- [ ] Extract and cache EO and SAR features (≈660 MB per split)
- [ ] First real geometry report: pairwise cosine structure of genuine DINOv3
      optical class means, against the ETF the control uses
- [ ] First real placement measurement: does the ~97% nullspace fraction seen on
      stub features hold for DINOv3?

A failure to reproduce is itself a finding and should be reported, not buried.

## Phase 3 — The corrected experiment (weeks 7 → 11)

The fall contribution. Rough draft due **20 November**.

- [ ] Run `run_variant_experiment.py --seeds 15` across the ladder
- [ ] Paired per-seed comparisons of every variant against real prototypes
- [ ] Settle whether the EO-versus-MMD gap survives adequate power
- [ ] Settle whether `rotated` matches `source` — geometry or specific directions
- [ ] Report placement alongside accuracy for every variant

**Compute estimate.** Eight variants × 15 seeds × 25 epochs = 3,000 epochs over
~387k training images. Not feasible on the M1. Two options: cache features and
train only the LoRA adapter over cached activations where the architecture
allows, or rent a GPU. At roughly $0.40/hour for a 4090 on a spot market this is
plausibly $30–60 for the full sweep — worth running the local diagnostic phase
first and renting only if Phase 2 says the effect is measurable.

## Phase 4 — Fall close-out (weeks 11 → 15)

- [ ] Rough draft (20 Nov)
- [ ] Video, ~5 minutes (10 Dec)
- [ ] Oral defense (week 15)
- [ ] Peer feedback on another presentation

## Phase 5 — Spring: the mechanism (provisional)

Fall diagnoses; spring attempts a fix. Direction depends on the fall result:

**If placement predicts outcome** — the interesting case. The alignment objective
is a reconstruction loss, and the KV-transfer work found reconstruction is the
wrong target. A placement-aware objective that weights residual by how strongly
the classifier reads each direction is a concrete, testable intervention.

**If placement does not predict outcome** — a clean negative result, and the
finding transfers: the analogy from attention to classification does not hold,
which is worth reporting. Pivot to the geometry question, characterising which
prototype property drives the gain.

**If nothing beats MMD at 15 seeds** — the strongest result available. Optical
prototypes would carry no class-specific advantage over label-free distribution
matching, and the field's premise needs revisiting.

---

## Risks

| Risk | Likelihood | Response |
|---|---|---|
| DINOv3 access denied or slow | Medium | Substitute an ungated backbone (DINOv2, CLIP ViT-B/16). The method is not DINOv3-specific; only the reproduction is. |
| Codabench closed or data unavailable | Low–Medium | SSDD, HRSID or OpenSARShip as substitute SAR datasets. Loses direct comparability with the published numbers — treat as a fallback, not a plan. |
| Full sweep exceeds compute budget | Medium | Cut the ladder to four variants (`source`, `etf`, `rotated`, `cosine_matched`) — still answers the geometry question, quarter the cost. |
| Reproduction fails | Low | Report it. A documented failure to reproduce is a legitimate contribution. |
| Effect vanishes under the corrected protocol | Medium | This is the likeliest interesting outcome and the project is designed to survive it: the corrected protocol is the contribution regardless of sign. |

## Standing commitments

- Additive only. No upstream file is modified, so the fork stays diffable.
- Every numerical claim carries a test with an analytic expected value.
- Results computed against the stub backbone are never reported as findings.
- Test-set evaluation happens once, after model selection on validation.

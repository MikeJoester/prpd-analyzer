# PatchCore PD-vs-Noise error analysis — paper section + investigation plan

Merges `my-error-analysis-plan.md` (the paper-section skeleton, §1–§5 below) with the investigation needed to
fill it. Written 2026-09-24 against `Results/patchcore/final_k8_s42..44` (v2 data: 229 test files, 75 PD / 154 Noise).

**Part A** is the write-up structure, with the numbers already in hand and the gaps marked.
**Part B** is the analysis work that closes those gaps.

---

## How the generic outline was adapted to this dataset

| Outline item | Problem here | Replacement |
|---|---|---|
| "FPs = normal flagged anomalous, FNs = anomalies missed" | This is **not** normal-vs-anomaly. Two banks compete, and neither class is "normal". | Define once: **PD is the event of interest**, so FN = missed PD, FP = noise called PD. State it in the caption; the code's positive class is Noise, so the confusion matrices must be read accordingly. |
| §2 causes: lighting, shadows, surface marks | No such factors in PRPD. | Real candidate causes: unfamiliar **noise environment** on specific dates, **bursty PD** diluted by window averaging, **Lab-dominated PD bank**, **amplitude/attenuation** differences, label noise. |
| §2 Category 3 + §3 ground-truth mask column | **There is no pixel-level ground truth**, and none can be made without a physics-based annotation of pulse locations. | Report window-level attribution instead: the 28 per-window scores as a strip, and whether the top-scoring window coincides with visible discharge activity. Qualitative, and stated as such. |
| §3 "3×3 grid: image / mask / heatmap" | Only two of three columns exist, **and the raw 128×3600 view is a pulse-sequence (PRPS) plot, not the 2D PRPD pattern PD papers show**. | Columns: **φ-q-n PRPD pattern** (phase × amplitude, colour = pulse count — the conventional 2D view), **per-window score strip**, **nearest memory-bank neighbour**. See §3 and Analysis J. |
| §4 "coreset discarded rare normal variations" | Testable here, and partly tested already. | Keep, with our measured evidence: coreset 0.001 → 0.748, 0.01 → 0.954, 0.05 → 0.957 val AUROC (saturating, so the coreset is **not** the bottleneck). |
| §5 "fine-tune the feature extractor" | PatchCore has **no trainable parameters** — fine-tuning means replacing the backbone or its weights, which changes the method. | Frame as: swap backbone, or train a small projection on Field data; and note that the honest finding may be that ImageNet features are the ceiling. |

---

# Part A — the paper section

## §1 Quantitative breakdown of errors

**Already measurable** (seed 42, threshold chosen on val):

| | Value |
|---|---|
| Errors | 61 of 229 files |
| **Noise → PD (false PD alarms)** | **43** |
| PD → Noise (missed PD) | 18 |
| At the F1-optimal threshold instead | 36 errors: 32 missed PD, 4 false alarms |

The direction **flips with the threshold**, which is itself the story: at the val threshold the model over-calls PD;
at the F1-optimal point it becomes conservative and misses PD instead. Report both, and say which threshold each
number belongs to.

*Drafting line:* "PatchCore reached an image-level AUROC of 0.853 [0.816, 0.872] over three seeds. At the
validation-selected threshold, 43 of 61 errors were noise files called PD, while at the F1-optimal threshold the
balance inverts to 32 missed PD against 4 false alarms — the ranking is stable, the operating point is not."

Needs: **Analysis A, C, I** (error table, margin/calibration, confidence interval).

## §2 Categorization of failure modes

**Category 1 — over-sensitivity (noise called PD).** Measured: errors concentrate on a few field dates. Three dates
hold **64%** of all errors; `20250217` (20 of 32 files wrong) and `20250213` (12 of 24) are both Field Noise dates.
Candidate cause: a noise environment the Noise bank never saw (H3).

**Category 2 — under-sensitivity (missed PD).** Measured: for **all 18** missed PD files, at least one window scores
on the PD side. The `mean` aggregation over 28 windows dilutes bursty discharge (H1). By fault type at the optimal
threshold, Corona is missed 6/6 and Void 23/51, while Floating is caught 15/17.

**Category 3 — attribution (replaces "localization").** Whether the highest-scoring windows coincide with visible
discharge. No ground truth, so this is illustrative, not a metric.

Needs: **Analysis B, D, E, F**.

## §3 Visual analysis — in the 2D PRPD representation

**Supervisor note (2026-09-24): error cases must be shown as 2D PRPD patterns of the raw data, the way the
literature presents them.** Our stored matrix is `128 phase × 3600 cycles` of peak amplitudes — a *pulse
sequence* (PRPS) view. The conventional PD figure is the **φ-q-n plot**: phase (0–360°) on x, apparent charge
/ amplitude on y, colour = number of pulses. It is obtained by histogramming amplitudes over the cycle axis
(verified: one Lab Corona file gives 75,615 pulses in a 128 × 256 phase-amplitude histogram).

Figure, ~8 rows: 4 noise-called-PD and 4 missed-PD, each with a **correct file from the same date** beside it
for contrast, so the reader sees what separates them rather than one pattern alone.

| Column | Content |
|---|---|
| 1 | **φ-q-n PRPD pattern** (phase × amplitude, log-count colour) — the view a PD engineer reads |
| 2 | PRPS view (`128 × 3600`) — what the model actually consumed |
| 3 | Per-window score strip (28 windows), threshold marked |
| 4 | **Nearest memory-bank neighbour** as a φ-q-n plot, labelled with its group/label/date |

Caption must state that no pixel ground truth exists and that column 4 is the matched training window.

Needs: **Analysis J (new), E, F**.

## §4 Root-cause hypotheses (PatchCore mechanics)

| Mechanism | Status | Evidence |
|---|---|---|
| **Aggregation dilutes bursty PD** (H1) | strongest lead, untested | E5: every missed PD file has PD-like windows |
| **The two banks' distances are on different scales** (H2) | untested | val→test threshold transfer is poor: macro F1 0.727 at val threshold vs 0.799 at test-optimal |
| **Unfamiliar noise environment** (H3) | untested | 2 dates dominate the false alarms |
| **PD bank is Lab-dominated** (H4) | untested | 1,435 Lab vs 348 Field PD files in the bank; all evaluation is Field |
| **Backbone unsuited to PRPD** (§4 of your outline) | partly tested | WideResNet-50 0.954 vs ResNet-50 0.881 val; and PatchCore only **ties** a kNN on the 256-D hand feature (0.853 vs 0.860 test), which is the real indictment of ImageNet features |
| **Coreset discarded rare normals** | **tested, rejected** | 0.001 → 0.748, 0.01 → 0.954, 0.05 → 0.957: saturating |
| **Label noise** (H5) | untested | 23 files already relabelled by the 260909 change log |
| **Small-sample noise** (H6) | partly measured | only 10 of 229 files are wrong in all three seeds; 53 are wrong in exactly one |

## §5 Implications and solutions

Order by what the evidence supports:

1. **Change the window aggregation** (count-of-PD-like-windows instead of mean) — targets the measured Category 2.
2. **Rank-normalize each bank's distances** — targets the threshold instability.
3. **Rebalance the PD bank toward Field** (`--k-field`) — targets the domain gap.
4. **Recover the 690 undated files** so val/test stop being this thin — the largest lever, and a data fix.
5. **Backbone alternatives**, stated honestly: if PatchCore cannot beat a 256-D kNN, the finding to report is that
   ImageNet features add nothing on PRPD, not that PatchCore needs more tuning.

---

# Part B — the investigation

## Hypotheses (test → confirming evidence → action)

| # | Hypothesis | Test | Confirms if | Action |
|---|---|---|---|---|
| **H1** | Aggregation dilutes bursty PD | Re-decide every file under `mean`/`min`/`p10`/"k-of-28", chosen on val | ≥ half of the 18 missed PD flip, without many new false alarms | Adopt the count rule |
| **H2** | Bank distance scales differ | Rank-normalize each `d` against its own bank's train scores, re-evaluate at the val threshold | The 0.727 → 0.799 gap shrinks | Adopt in `evaluate_two_model.py` |
| **H3** | Novel noise environment on 2 dates | Compare `d_noise` per date against other noise dates and the train distribution | Those dates are far from **both** banks | Report as coverage limit; test adding field-noise variety |
| **H4** | PD bank is Lab-dominated | For each test file's nearest entries, record Lab vs Field origin | Missed PD match Lab windows less / at larger distance | Raise `--k-field` |
| **H5** | Some errors are label errors | Cross-reference persistent errors with the Analyzer `suspect_score`, **and judge their φ-q-n pattern against the canonical signature of their labelled class (Analysis M)** | Persistent errors rank high on suspicion and their 2D pattern does not match the labelled fault type | Send for expert review with the φ-q-n figure — never relabel silently |
| **H6** | Much of the error is sampling noise | Bootstrap over **dates**, count errors inside the seed band | Interval ≈ ±0.05 AUROC | State the resolution limit of the test set |

## Analyses and deliverables (all under `Results/patchcore/error_analysis/`)

| ID | Output | Content | Cost |
|---|---|---|---|
| **A** | `error_table.csv` | One row per test file: score, margin to τ, prediction, error flag per seed, error flags for SVM and EfficientAD | 2 h, local |
| **B** | `error_by_date.csv`, `error_by_label.csv`, `error_rate_by_date.png` | Where the errors live | 1 h, local |
| **C** | `margin_hist.png`, `calibration.json` | Error rate by margin decile; H2 rank-normalization result | 2 h, local |
| **D** | `aggregation_study.csv` | Each aggregation rule × val-chosen τ × resulting val/test confusion (H1) | 2 h, local |
| **E** | `nn_audit.csv` | Top-5 bank neighbours of each error's worst window, mapped through `train_windows.csv` to `sample_id`/domain/label | 1 day, **server** (faiss) |
| **F** | `cases/<sample_id>.png` | The §3 figure rows | 3 h, after E |
| **G** | `suspect_cross_check.csv` | Persistent errors × Analyzer suspect score (H5) | 2 h, local |
| **H** | `method_overlap.csv` | Cross-method error overlap. Measured already: PatchCore∩SVM 29, PatchCore∩EfficientAD 16, **all three only 6** | done in draft, 1 h to finalize |
| **I** | `bootstrap.json` | Date-level bootstrap CI for every method (H6) | 1 h, local |
| **J** | `phi_q_n/<sample_id>.png` + `prpd_pattern.py` | φ-q-n renderer (phase × amplitude × count) used by every figure; the supervisor's 2D view | 3 h, local |
| **K** | `tsne_errors.png` | Test files on the Analyzer's 2D t-SNE of the 256-D feature, coloured by class, errors marked — shows whether errors sit between the clusters or inside the wrong one | 2 h, local |
| **L** | `phi_q_n_stats.csv` | Classic PD descriptors per file (per half-cycle: pulse count, mean/max amplitude, skewness, kurtosis, phase asymmetry, cross-correlation), compared error vs correct | 3 h, local |
| **M** | `literature_patterns.md` | Canonical φ-q-n signatures per fault type from the PD literature, with each error case judged against its labelled class | 3 h + reading |

## Order

1. A, B, H → substrate and the §1/§2 numbers (2–3 h)
2. I → which differences are even real (1 h)
3. D → biggest identified mechanism (2 h)
4. C → threshold transfer (2 h)
5. **J, K → the 2D views the supervisor asked for; K is cheap and may explain the error geometry immediately** (5 h, local)
6. E, F → the "why" and the §3 figure (1 day, server)
7. L, M → PD-domain descriptors and the literature comparison, for the discussion section (1 day)
8. G → feeds the data-quality track (2 h)

## Pitfalls

- **Never tune on test.** Rule changes in C and D are chosen on val, applied once to test.
- **Dates, not files, are the sampling unit.** 229 files come from 35 dates; one date can be 32 files.
- **Seed noise is large.** 61 / 30 / 54 errors across seeds; quote the seed or report all three.
- **Tiny cells.** Corona n=6, Particle n=1 in test — anecdote, not rate.
- **`unknown_date` files (≈45% of Field data) are all in train**, so they never appear in this analysis.
- **Label noise is real** (23 files relabelled) — H5 is a genuine possibility, not an excuse.

## What "done" looks like

Able to state, with numbers: which error type dominates and at which threshold; which mechanism causes it;
whether it is fixable inside PatchCore (fix chosen on val, verified once on test); and how much residual error is
irreducible at this test-set size.

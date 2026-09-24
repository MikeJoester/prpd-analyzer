# PatchCore PD-vs-Noise error analysis — results

Executed 2026-09-24 following `PatchCore_error_analysis_plan.md`. All numbers come from the seed-42 run
(`Results/patchcore/final_k8_s42`) unless a seed range is given. Outputs: `Results/patchcore/error_analysis/`.

Convention: PD is the event of interest. **missed PD** = PD file called Noise; **false alarm** = Noise file
called PD. The decision threshold is chosen on val.

---

## Headline

1. **The test set cannot separate PatchCore from a plain kNN on the 256-D hand feature.** Date-level
   bootstrap: the paired difference is −0.007 AUROC, 95% CI [−0.127, +0.151], P(PatchCore better) = 0.46.
2. **PatchCore does beat the two deep/classical detectors**: vs EfficientAD P = 0.91, vs One-Class SVM
   P = 0.98 (CI [0.006, 0.485], excludes zero).
3. **The aggregation hypothesis (H1) is wrong.** Making the decision sensitive to individual PD-like windows
   destroys performance.
4. **Errors are quiet, borderline files.** Both error types have far fewer pulses than correct files, and sit
   close to the threshold.
5. **The 10 persistent errors split into two distinct causes** — 3 atypical Corona (label-quality candidates)
   and 7 textbook Void files the model simply misses.
6. **The mechanism is bank ambiguity on quiet files.** Missed PD are not far from the PD bank; the *difference*
   between the two banks collapses (gap 0.925 → 0.152). Missed PD also match Lab rather than Field training
   windows (0.8 vs 0.2), confirming the Lab-dominated bank as a contributing cause.

---

## §1 Quantitative breakdown (A, I)

| | seed 42 | seed 43 | seed 44 |
|---|---:|---:|---:|
| Errors of 229 | 61 | 30 | 54 |
| false alarms / missed PD | 43 / 18 | – | – |
| Wrong in **all three** seeds | **10** | | |
| Wrong in exactly one seed | 53 | | |

**Date-level bootstrap** (2000 resamples of the 35 test dates, seed-42 runs):

| Method | test AUROC | 95% CI |
|---|---:|---|
| PatchCore | 0.864 | [0.730, 0.965] |
| kNN-5 (256-D) | 0.867 | [0.750, 0.948] |
| EfficientAD | 0.737 | [0.552, 0.869] |
| One-Class SVM | 0.611 | [0.379, 0.864] |

Paired differences vs PatchCore: **kNN-5 −0.007 [−0.127, +0.151] (P = 0.46)**, EfficientAD +0.128 (P = 0.91),
SVM +0.243 [0.006, 0.485] (P = 0.98).

> The intervals are ±0.12 wide. **Any tuning gain below ~0.1 AUROC is invisible on this test set.**

## §2 Where the errors are (B, H)

**By date** — three dates hold 64% of the errors, and the two worst are pure Field-Noise dates:

| date | files | PD | errors | missed PD | false alarms |
|---|---:|---:|---:|---:|---:|
| 20250217 | 32 | 0 | 20 | 0 | 20 |
| 20250213 | 24 | 0 | 12 | 0 | 12 |
| 20220427 | 17 | 17 | 7 | 7 | 0 |
| 20231023 | 6 | 6 | 5 | 5 | 0 |

**By method** (seed 42, files wrong): PatchCore 61, EfficientAD 68, SVM 95, kNN-5 81. Overlap is small —
PatchCore ∩ EfficientAD 16, PatchCore ∩ SVM 29, PatchCore ∩ kNN-5 29. The methods fail on largely
**different** files, so "these files are intrinsically hard" is not the explanation.

## §3 What the errors look like (J, K, L)

**Activity level.** Median total pulses per file (φ-q-n histogram):

| | correct | error |
|---|---:|---:|
| Noise files | 18,747 | 4,334 (false alarms) |
| PD files | 34,086 | 3,787 (missed PD) |

Error rate by pulse-count quartile: Q1 (quietest, median 194 pulses) 28%, **Q2 (median 4.3k) 51%**, Q3 (33k)
14%, Q4 (461k) 14%. Busy PD files are never missed (Q4 PD error rate 0%). Correlation between log pulse count
and distance from the threshold is +0.52: **the more active the file, the more confident the model**.

**Position in feature space** (`tsne_errors.png`). Median fraction of a file's 15 nearest train/val neighbours
that share its own class: correct 0.73, missed PD 0.70, **false alarms 0.53**. 21 of 43 false alarms sit in a
neighbourhood that is majority PD — they are genuinely PD-like in the 256-D feature, not merely borderline.

**Margin.** Median |score − threshold| is 0.035 for errors vs 0.131 for correct files; the half of files nearest
the threshold carries a 40% error rate against 13% for the far half.

## §3b Which training windows the model matched (E)

The memory banks were rebuilt recording, for every bank entry, the training window it came from
(`bank_entry_source.csv`). For each test file the window that drove its decision was then queried against both
banks and its top-5 neighbours traced back.

**PD files** (their most PD-like window):

| | files | Lab share of neighbours | dist to PD bank | dist to Noise bank | **gap** |
|---|---:|---:|---:|---:|---:|
| caught | 57 | 0.2 | 1.787 | 2.828 | **0.925** |
| missed | 18 | **0.8** | 1.628 | 1.733 | **0.152** |

**Noise files** (their most Noise-like window):

| | files | dist to PD bank | dist to Noise bank | gap |
|---|---:|---:|---:|---:|
| correct | 111 | 1.861 | 2.064 | +0.124 |
| false alarm | 43 | 1.992 | 1.772 | −0.166 |

Two findings:

1. **H4 confirmed.** Missed PD files match Lab training windows (mean Lab share 0.70) while caught PD files
   match Field windows (0.31), p = 6e-5. The Lab-heavy PD bank (1,435 Lab vs 348 Field files) is the wrong
   reference for the field measurements it is asked to recognise.
2. **The real mechanism is bank ambiguity, not distance.** Missed PD files are *not* far from the PD bank —
   they are slightly **closer** to it than caught files (1.628 vs 1.787). What collapses is the *difference*
   between the two banks: the gap falls from 0.925 to 0.152 (p = 1e-6). Noise false alarms show the mirror
   image, with the gap going slightly negative.

This ties the whole analysis together with the activity finding: **quiet files produce near-empty windows, and
near-empty windows are well covered by both banks.** Both distances shrink, their difference collapses toward
zero, the file lands next to the threshold, and the seed or the threshold decides the label. That is why errors
are quiet (§3), near the threshold (§3), seed-unstable (§1), and method-specific (§2).

> Caveat: the window chosen per file depends on its predicted class (most Noise-like for false alarms, most
> PD-like otherwise), so the "which bank is closer" column is partly determined by that choice. The Lab/Field
> neighbour share and the gap comparison are made **within** the same selection rule and are not affected.

## §4 Hypotheses — what survived

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| **H1** | Aggregation dilutes bursty PD | **Rejected** | Every alternative rule is worse on val. `mean` 0.959 val AUROC; `min` 0.916; `max` 0.907; "PD if ≥3 of 28 windows look PD" 0.752 and 124 false alarms. Noise files contain PD-like windows too, so sensitivity to any single window is fatal. |
| **H2** | Bank distance scales differ | **Not supported** | Rank-normalizing each bank against its val distribution changes AUROC by −0.044 / +0.023 / +0.028 across seeds (median 0.853 → 0.844). It does lift macro F1 at the val threshold for 2 of 3 seeds (0.718 → 0.755, 0.727 → 0.738), so it helps threshold transfer slightly, not ranking. |
| **H3** | Unfamiliar noise environment | **Supported** | 32 of 61 errors come from two Field-Noise dates never seen in training, with 63% and 50% error rates on those dates. |
| **H4** | PD bank is Lab-dominated | **Supported** | Missed PD files match **Lab** training windows (median 0.8 of their top-5 PD-bank neighbours) while correctly-caught PD files match **Field** windows (0.2); Mann-Whitney p = 6e-5. See §3b. |
| **H5** | Some errors are label errors | **Partly supported** | See below. |
| **H6** | Much of the error is sampling noise | **Confirmed** | Only 10 of 229 files are wrong in all three seeds; 53 are wrong in exactly one. Bootstrap CI ±0.12. |

## §5 The 10 persistent errors (G)

All 10 are **missed PD**, from just two dates, and they split cleanly:

| Group | n | Analyzer suspect score | Reading |
|---|---:|---|---|
| **Corona, 20231023** | 3 | 434 / 375 / 358 — all above the p95 cut (210) | The Analyzer independently finds these atypical for Corona. **Label-quality candidates**: send to expert review with their φ-q-n plots. |
| **Void, 20220427** | 7 | 7.8 – 17.2, cell percentile 0.6–20% | Textbook Void by the Analyzer's own metric, yet PatchCore misses them. **A genuine model failure**, not label noise. |

## §6 Revised recommendations

The plan's priority list changes, because H1 is dead and H2 is weak:

1. **Stop tuning against this test set.** The bootstrap CI (±0.12) is wider than every effect measured so far.
   Move to date-grouped cross-validation before any further model work.
2. **Report PatchCore ≈ kNN-5 honestly.** On this data an ImageNet memory bank does not beat a 256-D
   hand-crafted feature with 5-NN. That is the paper's most defensible finding.
3. **Fix the data, not the model.** The dominant error source is two unseen field-noise environments, and 690
   Field files are excluded from val/test only because their dates cannot be parsed. Recovering those dates
   (their filenames carry epoch-ms timestamps) is worth more than any hyperparameter.
4. **Rebalance the PD bank toward Field** (`--k-field` in the fine-tuning plan). This is now the best-supported
   model-side change: missed PD match Lab windows 0.8 of the time, caught PD only 0.2.
5. **Keep rank normalization as an operating-point fix only**, with the caveat that it does not improve ranking.
6. **Send the 3 Corona files to review**, and treat the 7 Void misses as the model-failure case study for §4.

## Reproduce

```bash
python Comparison/error_analysis.py     # A, B, C, D, H, I
python Comparison/error_figures.py      # J, K, L  (t-SNE, phi-q-n stats, case figures)
# on the GPU server, after rebuilding banks with coreset provenance:
python Comparison/nn_audit.py --run Results/patchcore/nnaudit_k8_s42   # E
```

## Engineering note

Rebuilding both banks in parallel at K=8 needs 45.8 GB (PD) + 22.5 GB (Noise) of feature pool and was
OOM-killed once on the 125 GB server; the earlier run had fit only marginally (47.9 + 24.8 GB peak RSS).
**At K=8 with layer2+layer3, build the banks with `SEQUENTIAL=1`.** The rebuilt banks reproduced the original
sizes exactly (111,829 PD / 54,942 Noise entries), so the coreset is deterministic given the seed.

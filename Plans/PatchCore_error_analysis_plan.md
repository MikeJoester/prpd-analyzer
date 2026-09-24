# PatchCore PD-vs-Noise error analysis plan

Purpose: find out **why** the two-bank PatchCore is wrong on the files it gets wrong, and turn that into
either a fix or a documented limitation for the paper. Written 2026-09-24 against
`Results/patchcore/final_k8_s42..44` (v2 data, 229 test files: 75 PD, 154 Noise).

---

## 1. What the errors already look like (measured, not assumed)

These five facts come from the existing run artifacts and should shape everything below.

| # | Finding | Numbers (seed 42, val threshold) |
|---|---|---|
| E1 | **Most errors are false "PD" calls on noise**, not missed PD | 43 of 61 errors are Noise→PD; 18 are PD→Noise |
| E2 | **Errors concentrate on a few field dates** | 3 dates hold 64% of all errors; `20250217` (20/32 wrong) and `20250213` (12/24) are both Field Noise |
| E3 | **Errors sit near the decision threshold** | median \|score − τ\| is 0.035 for errors vs 0.131 for correct files; the half nearest τ holds a 40% error rate, the far half 13% |
| E4 | **Errors are seed-unstable** | 61 / 30 / 54 errors for seeds 42/43/44; only **10 files are wrong in all three**, while 53 are wrong in exactly one |
| E5 | **Every missed PD file has PD-looking windows** | for all 18 PD→Noise files, at least one window scores on the PD side; the `mean` aggregation dilutes them |

Read together: this looks much more like a **calibration/aggregation problem on a couple of unfamiliar
noise environments** than like a feature extractor that cannot see PD. The analysis below is designed
to confirm or kill that reading.

---

## 2. Hypotheses to test, in priority order

Each has a test, the evidence that would confirm it, and what to do if it holds.

| # | Hypothesis | Test | Confirming evidence | Action if true |
|---|---|---|---|---|
| **H1** | **Aggregation dilutes bursty PD.** PD activity occupies a few of the 28 windows; averaging buries it. | For each test file, compare the label decision under `mean`, `min`, `p10`, and "k-of-28 windows below τ". Measure how many of the 18 missed PD files flip. | ≥ half the missed PD files flip to correct without adding many new Noise→PD errors | Replace the aggregation with a count-based rule (e.g. "PD if ≥ 3 windows are PD-like"), chosen on val |
| **H2** | **The two banks' distances are on different scales**, so `d_pd − d_noise` is biased and τ does not transfer. | Rank-normalize each `d` against that bank's own train-window score distribution, then re-evaluate with the val-chosen τ. | val→test threshold gap shrinks; macro F1 at the val threshold moves toward the optimal-threshold F1 (0.727 → 0.799 gap) | Adopt rank normalization in `evaluate_two_model.py` |
| **H3** | **Two noise dates are a novel environment** the Noise bank does not cover. | For every test window, record `d_noise` and group by date. Compare the two bad dates against the other noise dates and against the train-window `d_noise` distribution. | `d_noise` is systematically higher on `20250213` / `20250217` — they are far from *both* banks, not "PD-like" | Report as a coverage limitation; test whether adding those dates' *train-split* neighbours (or more Field noise) closes it |
| **H4** | **The PD bank is Lab-dominated** (1,435 Lab vs 348 Field files), so Field PD is far from it. | For each test file's nearest bank entries, record whether they come from Lab or Field train windows, and compare error vs correct files. | Missed PD files match Lab windows far less often / at larger distance than correct ones | Raise `--k-field` (fine-tuning plan #9), or build the PD bank from Field only and compare |
| **H5** | **Some "errors" are label errors.** The 260909 change log already moved 23 files. | Cross-reference persistent errors (E4's 10 files) with the Analyzer `suspect_score` from `prpd_analyzer.quality`, and inspect their PRPD images. | Persistent errors score in the top suspect percentiles, and look like the class the model predicted | Send that list to review (do not silently relabel — CLAUDE.md forbids it) |
| **H6** | **Most errors are borderline noise, not systematic failure** (E3, E4). | Bootstrap the test metric over dates and report a confidence interval; count how many errors lie within the seed-to-seed score spread. | The interval is wide (roughly ±0.05 AUROC) and most errors are inside the seed band | State in the paper that the test set cannot separate methods below that margin; push for date-grouped CV |

---

## 3. Analyses to run (with deliverables)

### A. Error taxonomy — `Results/patchcore/error_analysis/error_table.csv`
One row per test file: `sample_id`, date, group, label, score, margin to τ, prediction, error flag per seed,
error flag for SVM and EfficientAD. This is the substrate for everything else.
*Cheap, local, no GPU.*

### B. Where the errors live — `error_by_date.csv`, `error_by_label.csv`, plus a date × error-rate figure
Error rate per date with file counts, and per fault type. Include noise dates: E2 says that's where the mass is.
*Cheap, local.*

### C. Margin and calibration — `margin_hist.png`, `calibration.json`
Score histograms for correct vs wrong files, error rate in margin deciles, and the H2 rank-normalization
re-evaluation. Report macro F1 at the val threshold before and after.
*Cheap, local.*

### D. Aggregation study — `aggregation_study.csv`
For each candidate rule (`mean`, `median`, `min`, `p10`, `p90`, "k-of-28"), the val-chosen τ and the resulting
val and test confusion. Tests H1 directly. **Choose on val only.**
*Cheap, local — window scores are already saved.*

### E. Nearest-neighbour introspection — `nn_audit.csv` (the "why" tool)
For each error file's worst window, query both faiss indexes for its top-5 neighbours and map those indexes back
to rows of `bank_*/train_windows.csv`, which record `sample_id`, `domain` and `label`. That turns "distance 0.31"
into "its nearest match is a Lab Corona window from file X". Tests H4, and often explains H3 too.
*Needs the saved banks → run on the server (faiss). Half a day of coding.*

### F. Visual case review — `cases/<sample_id>.png`
For ~15 files (the 10 persistent errors, plus the largest-margin mistakes in each direction): the full
128×3600 PRPD, the 28 window scores as a strip, and the nearest bank neighbour from E. This is what a
reviewer will want to see, and it feeds H5.
*Cheap once E exists.*

### G. Label-quality cross-check — `suspect_cross_check.csv`
Join persistent errors against `build_category_report` (pooled shrinkage Mahalanobis, `quality.py`).
Report overlap, don't change labels.
*Cheap, local.*

### H. Cross-method error overlap — `method_overlap.csv` + Venn-style counts
Already partly measured: at seed 42 PatchCore and SVM share 29 errors, PatchCore and EfficientAD 16, and only
**6 files are wrong for all three**. Files all three miss are candidates for "genuinely ambiguous"; files only
PatchCore misses point at something specific to its features.
*Cheap, local.*

### I. Uncertainty — `bootstrap.json`
Bootstrap AUROC and macro F1 by **resampling dates** (not files), since files within a date are correlated.
Report a 95% interval for every method. Tests H6 and tells you which differences in the comparison table are real.
*Cheap, local.*

---

## 4. Order of work

| Step | Analyses | Why here | Effort |
|---|---|---|---|
| 1 | A, B, H | Builds the substrate and confirms E1–E2 for all three seeds | 2–3 h |
| 2 | I | Tells you which differences are even worth explaining | 1 h |
| 3 | D | Directly targets the biggest identified mechanism (E5/H1) | 2 h |
| 4 | C | Threshold transfer is the other big mechanism (E3/H2) | 2 h |
| 5 | E, F | The "why", and the figures a reviewer asks for | 1 day |
| 6 | G | Feeds the data-quality track, not the model | 2 h |

Steps 1–4 are local and need no GPU. Only E needs the server.

---

## 5. Pitfalls specific to this dataset

- **Never tune on test.** Every rule change in D or C is chosen on val, then applied once to test. The plan's
  numbers above are diagnostic, not selection criteria.
- **Dates, not files, are the unit.** 229 test files come from 35 dates, and one date can be 32 files. Any
  per-file statistic or bootstrap that ignores this overstates significance.
- **Seed noise is large** (E4). Report every error statistic across all three seeds, or say which seed it is.
- **Small cells.** Corona n=6 and Particle n=1 in test. A per-fault-type error rate on those is anecdote.
- **`unknown_date` files are all in train**, so they never appear in this analysis, but they are ~45% of Field
  data. Fixing the date parser (results plan §6) would change the error picture more than any model change.
- **Label noise is real** (23 files already relabelled). Treat H5 as a genuine possibility, not an excuse.

---

## 6. What a good outcome looks like

By the end you should be able to state, with numbers:

1. **What kind of error dominates** — currently false-PD calls on two unfamiliar noise dates.
2. **Which mechanism causes it** — aggregation (H1), threshold scale (H2), bank coverage (H3/H4), or labels (H5).
3. **Whether it is fixable within PatchCore** — and if yes, the fix was chosen on val and verified once on test.
4. **How much of the remaining error is irreducible noise** at this test-set size (H6/I).

That is enough for an honest error-analysis section, whether or not the fixes land.

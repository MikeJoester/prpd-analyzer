# PatchCore fine-tuning plan (manual)

For tuning the two-bank PD-vs-Noise PatchCore by hand on the GPU server.
Starting point: the current best config, measured in `PatchCore_v2_results.md`.

```text
wideresnet50 | layer2+layer3 | image 224 | patchsize 3 | dim 1024/1024 | coreset 0.01 | num_nn 1 | K=8
test: image AUROC 0.853 [0.816, 0.872]   optimal F1 0.799   |   baseline 256-D kNN-5: AUROC 0.860, F1 0.796
```

**The goal of tuning is to beat the kNN-5 baseline (AUROC 0.860), not just to improve on 0.853.**

---

## Data setting (fixed during tuning)

`PatchCore/patchcore_data/prpd_v2`, built from `artifacts/ai_data_20260911_011704` (`raw_v2`, Synthetic excluded), split seed 42.
**3,118 files in total**, each tiled into 28 non-overlapping 128×128 windows (87,304 images).

| Split | Lab PD | Lab Noise | Field PD | Field Noise | PD total | Noise total | Files | Dates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **train** | 1,435 | 156 | 348 | 720 | 1,783 | 876 | **2,659** | 46 |
| **val** | – | – | 75 | 155 | 75 | 155 | **230** | 33 |
| **test** | – | – | 75 | 154 | 75 | 154 | **229** | 35 |
| Total | 1,435 | 156 | 498 | 1,029 | 1,933 | 1,185 | **3,118** | 114* |

\* 114 distinct dates in total: 105 Field dates and 10 Lab dates, which share the `unknown_date` bucket. Each date belongs to exactly one split (46 + 33 + 35 = 114).

- **All Lab files are in train**, so val and test are Field only: every number you tune on is a field result.
- **Field PD is 70/15/15 by date** (348/75/75) and Field Noise likewise (720/155/154). No date appears in two splits.
- **Windows actually used at K=8:** 14,264 PD + 7,008 Noise for the banks; 6,440 val and 6,412 test windows are scored in full (28 per file).
- **Field PD labels** (the limit on per-label conclusions):

| Split | Corona | Floating | Particle | Void |
|---|---:|---:|---:|---:|
| train | 26 | 99 | 2 | 221 |
| val | 1 | 9 | 0 | 65 |
| test | 6 | 17 | 1 | 51 |

  Val PD is dominated by Void (65 of 75), and Corona/Particle are too rare in val to steer tuning. A change that only helps Corona will not show up in val AUROC.
- **Noise outnumbers PD about 2:1** in val and test, which is why macro (not plain) averages are used.
- Changing any of this (rebuilding the data, `--split-seed`, `--exclude-groups`) invalidates comparisons against every earlier run.

## 0. Rules to follow while tuning (they decide whether a result means anything)

| Rule | Why |
|---|---|
| **Change one parameter per run.** | Two changes at once can cancel out or mask each other. |
| **Run 3 seeds (42/43/44) before believing any gain.** | Seed noise on val is about **±0.03 AUROC**. K=8 looked like +0.03 on seed 42 alone, but was +0.005 over 3 seeds. |
| **Select on val only. Never look at test while tuning.** | `VAL_ONLY=1` does this for you: it scores val only and writes no test numbers. |
| **Keep the split fixed** (`split_seed=42`, the `prpd_v2` build). | Rebuilding data changes val/test, and results stop being comparable. |
| **Adopt a change only if median val AUROC gains more than the [min, max] seed spread.** | Otherwise you are fitting noise. |
| **Record every run**, including failures. | `python PatchCore/summarize_runs.py Results/patchcore/<run>... --out table.csv` collects them. |

**Execution protocol used (2026-09-29).** Running 3 seeds for every candidate would be ~27 runs and about 18 h of
server time, so candidates are **screened at seed 42 first** and only those that beat the current config on val
go on to the full 3 seeds. A screen result is never an adoption decision by itself — the seed spread is ±0.03
val AUROC, which is larger than most effects being tested.

`--num-nn` needs **no rebuild**: it only changes scoring, so those variants reuse the existing `final_k8_s*`
banks (`score_two_bank.py --bank-run ... --num-nn k`). Note that PatchCore's score with k > 1 is the *mean*
distance to the k nearest bank entries, not the distance to the k-th.

### How to run one candidate

```bash
# on the server, from ~/DR_12
VAL_ONLY=1 bash PatchCore/run_two_bank.sh <run_name> <train args>
# example: one parameter changed, three seeds
for s in 42 43 44; do VAL_ONLY=1 bash PatchCore/run_two_bank.sh nn3_s$s --num-nn 3 --seed $s; done
python PatchCore/summarize_runs.py Results/patchcore/nn3_s4? --out /tmp/nn3.csv
```

Add `SEQUENTIAL=1` if a setting needs more host RAM than two banks at once allow (see the memory column).
Only when tuning is finished: rerun the winner without `VAL_ONLY`, 3 seeds, and evaluate test **once**.

---

## 1. Model parameters (already command-line flags, no code change)

Sorted by how much they are worth trying first. "Cost" is PD-bank build time at K=8; the Noise bank is about ¼ of that.

| # | Parameter | Flag | Current | Values to try | What it changes | Cost per run | Evidence so far |
|---|---|---|---|---|---|---|---|
| 1 | **Neighbours in scoring** | `--num-nn` | 1 | **3**, 5, 9 | Score = distance to the k-th nearest bank entry. Larger k smooths the score and resists a single odd bank entry. | 33 min (scoring only, no rebuild if you reuse a bank) | Untested. Your old v1 used 9. Cheapest real knob. |
| 2 | **Train windows per file** | `--k` | 8 | **12**, 16, 28 (all) | How much of each file enters the bank. Directly sets bank coverage and RAM. | K=12 ≈ 50 min, K=16 ≈ 70 min, K=28 ≈ 2 h | K 1→8 gave 0.953→0.959 (median). Trend still rising at the grid edge, so K=12/16 is the obvious next step. **K≥16 needs `SEQUENTIAL=1`** (~90 GB host RAM at K=16). |
| 3 | **Coreset ratio** | `--coreset` | 0.01 | 0.02, **0.05**, 0.1 | Fraction of patch features kept in the bank. Higher = finer memory, slower build and search. | 0.05 ≈ 48 min, 0.1 ≈ 1.5 h | 0.001→0.01 gave 0.748→0.954 (big). 0.01→0.05 gave +0.003 (small). Likely near saturation; low priority. |
| 4 | **Feature layers** | `--layers` | layer2 layer3 | `layer2` alone, `layer3` alone, `layer3 layer4` | Which depth the patch features come from. Deeper = more semantic, coarser grid. | ~33 min | layer1+2 was clearly worse (0.887 vs 0.953). Not yet tried: layer3 alone, or layer3+4. **layer1 needs `--batch-size 16`** (4× the patches; OOM at 64). |
| 5 | **Input size** | `--image-size` | 224 | 128 (native), 160, 320 | The 128×128 window is resized to this. 128 = no resampling; larger = finer patch grid, more patches. | 128 ≈ 12 min, 320 ≈ 70 min | Untested. 128 is much cheaper and avoids upsampling artifacts; 320 is the opposite bet. Worth one run each. |
| 6 | **Backbone** | `--backbone` | wideresnet50 | `resnet101`, `resnext101`, `wideresnet101`, `vgg19_bn` | Feature extractor. Valid names: alexnet, resnet50, resnet101, resnext101, vgg11, vgg19, vgg19_bn, wideresnet50, wideresnet101. | ~30–50 min | wideresnet50 (0.954) beat resnet50 (0.881). A bigger backbone is the next thing to test. |
| 7 | **Embedding dimensions** | `--pretrain-dim` / `--target-dim` | 1024 / 1024 | 512/512, 256/256, 2048/1024 | Size of each patch feature before it enters the bank. Smaller = less RAM and faster search, some information lost. | dim 512 ≈ 25 min | Untested at v2 scale. Your old v1 used 256. Mainly a cost lever; try it if you want K=28 to fit. |
| 8 | **Patch size** | `--patchsize` | 3 | 1, 5 | Neighbourhood (in feature cells) pooled into one patch feature. Larger = more context per patch, blurrier localization. | ~33 min | Untested. |
| 9 | **Per-domain windows** | `--k-field` | = `--k` | `--k 8 --k-field 16` or 24 | More windows from **Field** files than Lab files, so the bank represents field variety better. | between K=8 and K=16 | Untested, and it targets the main known weakness: Lab files are about 80% of the PD bank while all testing is on Field. **High value, try early.** |

## 2. Data parameters (need a dataset rebuild, `PatchCore/build_prpd_v2.py`)

A rebuild changes the windows, so rebuild into a **new folder** and never compare across builds without re-running every candidate.

| # | Parameter | Flag | Current | Values to try | What it changes | Notes |
|---|---|---|---|---|---|---|
| 10 | **Window width** | `--window` | 128 (28 per file) | 64 (56/file), 256 (14/file) | Cycles of context per window. Narrow = more, simpler windows; wide = more temporal context, fewer windows. | 256 is no longer square, so `--image-size` will stretch it. Rebuild: `--out .../prpd_w256`. |
| 11 | **Window overlap** | *not implemented* | none | stride ½ window | Overlapping windows give more training windows and smoother coverage. | Needs a small change in `tile()` in the build script. |
| 12 | **Groups included** | `--exclude-groups` | Synthetic excluded | also exclude `Lab Noise`, or include Synthetic | Which groups form the two banks. | Excluding Lab Noise makes the Noise bank purely field-like — matching the test domain. **Worth one test.** |
| 13 | **Split seed** | `--split-seed` | 42 | 43, 44 | A different date assignment. | **Do not use for tuning.** Only to check how much your conclusions depend on one split. |

## 3. Scoring and decision parameters (no retraining — cheapest of all)

These are applied after scoring, so you can try them on an existing run in seconds.

| # | Parameter | Where | Current | Values to try | What it changes |
|---|---|---|---|---|---|
| 14 | **Window→file aggregation** | automatic | best of 5 on val | mean / median / max / min / p90 | How 28 window scores become one file score. All five are computed for every run: see `val_selection.json`. **Nothing to run.** |
| 15 | **Threshold** | automatic | macro-F1 best on val | fixed recall, or per-class percentile | The PD/Noise cut-off. Known weak point: val→test transfer is poor. |
| 16 | **Score normalization** | `Comparison/evaluate_two_model.py` | raw `d_pd − d_noise` | rank-normalize each `d` against its own bank's train scores | The two banks have different distance scales, so their difference is biased. **Small code change, potentially real gain.** |
| 17 | **Patch score pooling** | `patchcore.py` `PatchMaker.score` | max over patches | mean of top-5 patches | One window's score from its patch scores. `max` is sensitive to a single outlier patch. |

## 4. Suggested order

Work down this list, 3 seeds each, and stop when runs stop paying for themselves.

| Step | What | Why first | Expected time |
|---|---|---|---|
| 1 | `--num-nn 3` and `5` | Cheapest, untested, your old config used 9 | ~1 h total if you reuse banks |
| 2 | `--k-field 16` (with `--k 8`) | Directly targets the Lab/Field imbalance | ~2.5 h |
| 3 | `--k 12` then `16` | The one lever with a proven upward trend | ~4 h |
| 4 | `--image-size 128` | Removes the 128→224 upsampling; also very cheap | ~1 h |
| 5 | Rank normalization (#16) | Fixes a known bias between banks, no retraining | ~30 min of coding |
| 6 | Backbone `resnet101` / `resnext101` | Next-largest lever after the above | ~2 h |
| 7 | `--layers layer3` alone, `layer3 layer4` | Depth is untested above layer3 | ~2 h |
| 8 | Exclude Lab Noise (#12) | Makes the Noise bank match the test domain | rebuild + ~1.5 h |

## 5. Worksheet (fill in as you go)

Val AUROC, 3 seeds, median [min, max]. Adopt only if the gain clears the spread.

**Results so far (2026-09-29/10-01).** Tuning was stopped early; what was measured:

| Run | Parameter changed | K | val AUROC s42 | s43 | s44 | Verdict |
|---|---|---:|---:|---:|---:|---|
| (current) | – (K=8 config) | 8 | 0.987 | 0.958 | 0.959 | baseline, median 0.959 |
| nn3 | `--num-nn 3` | 8 | 0.954 | 0.973 | 0.986 | **rejected** — paired change +0.003, sign inconsistent |
| nn5 | `--num-nn 5` | 8 | 0.916 | 0.970 | 0.985 | **rejected** — paired −0.011 |
| nn9 | `--num-nn 9` | 8 | 0.896 | 0.970 | 0.986 | **rejected** — paired −0.017 |
| kfield16 | `--k 8 --k-field 16` | 8 | 0.989 | – | – | +0.002 vs K=8 — inside noise, needs 3 seeds |
| k12 | `--k 12` | 12 | – | – | – | cancelled mid-run |

> **Flaw in the screening queue (`queue_tune_screen.sh`): the runs below omitted `--k 8`, so they fell back to
> the default K=4 and changed two things at once.** They are comparable only to the **K=4** baseline
> `v2_start_s42` (val 0.9541), not to the K=8 config. Re-run with `--k 8` before drawing any conclusion.

| Run | Parameter changed | K | val AUROC s42 | vs K=4 baseline (0.954) |
|---|---|---:|---:|---|
| l34 | `--layers layer3 layer4` | 4 | 0.984 | **+0.029** — the only promising lever; re-run at K=8, 3 seeds |
| bb_rn101 | `--backbone resnet101` | 4 | 0.937 | −0.017 — no gain over wideresnet50 |
| img128 | `--image-size 128` | 4 | 0.901 | −0.053 — upsampling to 224 helps, keep it |
| l3 | `--layers layer3` | 4 | 0.891 | −0.063 — layer2 contributes |

Remaining worksheet:

| Run name | Parameter changed | Val AUROC s42 | s43 | s44 | Median [min, max] | Keep? |
|---|---|---|---|---|---|---|
| l34_k8 | `--layers layer3 layer4 --k 8` | | | | | |
| kfield16 | `--k 8 --k-field 16` (seeds 43/44) | 0.989 | | | | |
| k12 | `--k 12` | | | | | |
| k16 | `--k 16` (SEQUENTIAL=1) | | | | | |

## 6. Practical limits (measured on the server)

| Setting | Host RAM (PD bank) | GPU | Note |
|---|---|---|---|
| K=8, dim 1024 (current) | 48 GB (PD) + 25 GB (Noise) | 6 GB | **Use `SEQUENTIAL=1`** — parallel builds were OOM-killed once at this size |
| K=16, dim 1024 | ~90 GB | 6 GB | Use `SEQUENTIAL=1` |
| K=28, dim 1024 | ~160 GB | 6 GB | **Won't fit.** Needs `--target-dim 512` or `SEQUENTIAL=1` + dim 512 |
| layer1 included, batch 64 | – | OOM | Use `--batch-size 16` |
| layer3+layer4 | lower than now | low | Coarser grid, so fewer patches: fast |

The server has 125 GB RAM and 2× 24 GB GPUs. Watch a run with `free -g` and `nvidia-smi`.

## 7. When tuning is done

1. Pick the winner on **val** only.
2. Rerun it with 3 seeds **without** `VAL_ONLY`, so val and test are both scored.
3. `python Comparison/evaluate_two_model.py --run <run> --title PatchCore` for each seed.
4. `python Comparison/compare_v2.py --out Results/Model_Comparison_v2` to rebuild the table.
5. Update `Plans/PatchCore_v2_results.md` with the new config and numbers.
6. **Judge against the kNN-5 baseline** (AUROC 0.860). If PatchCore still ties it, the honest conclusion is that ImageNet features add nothing here over the 256-D feature, and that is a result worth reporting.

> A caution about the test set: it holds 75 PD files from 17 dates, and single dates visibly move the metrics. A test gain under about 0.03 AUROC is not distinguishable from date noise. If tuning produces only small gains, the date-grouped cross-validation in `PatchCore_v2_results.md` §6 is the better next move.

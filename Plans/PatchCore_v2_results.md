# PD vs Noise on v2 data: PatchCore, EfficientAD and One-Class SVM results

Run 2026-09-18 on the GPU server (2× RTX 4090, conda `danenv`). Plans: `PatchCore_data_plan.md`, `PatchCore_training_plan.md`.
Raw outputs: `Results/patchcore/`, `Results/efficientad/`, `Results/svm/`. Final table: `Results/Model_Comparison_v2/`.

## 1. Setup (same for all three methods)

- **Data:** `PatchCore/patchcore_data/prpd_v2`, built from `artifacts/ai_data_20260911_011704` (`raw_v2`), Synthetic excluded.
- **Split:**
  - Train = all Lab PD + Lab Noise + 70% of Field PD and Field Noise.
  - Val and test = 15% of Field each.
  - Field files are split **by date**, with no date shared between splits.
- **File counts** (3,118 files total; 28 windows each = 87,304 images):

| Split | Lab PD | Lab Noise | Field PD | Field Noise | PD | Noise | Files | Dates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 1,435 | 156 | 348 | 720 | 1,783 | 876 | 2,659 | 46 |
| val | – | – | 75 | 155 | 75 | 155 | 230 | 33 |
| test | – | – | 75 | 154 | 75 | 154 | 229 | 35 |

  Val and test are Field only, from dates never seen in training. At K=8 the banks hold 14,264 PD and 7,008 Noise windows.
- **Input:** 128×128 windows (28 per file), raw uint8 values.
- **Training windows:** every method uses **K = 8** seeded windows per train file (14,264 PD / 7,008 Noise), the identical list for the same seed. This was verified run-by-run from the saved window lists and data checksums.
  - Correction (2026-09-19): SVM and EfficientAD were first run at K=4 by mistake. Those runs are kept as `*_K4_MISMATCH_s*` and are **not** used here.
- **Two models per method:** one fit on PD, one on Noise. Window score `s = anomaly(PD model) − anomaly(Noise model)`, where > 0 means the window looks like Noise.
- **Protocol:**
  - The window→file aggregation and the threshold are chosen on val, then applied unchanged to test.
  - Test was scored once per final model.
  - Each method was trained with 3 seeds (42/43/44). Cells show the median [min, max].
- **Pixel-wise AUROC is not reported.** The data has no pixel-level ground truth. **Window-level AUROC** replaces it: every test window is scored and labelled with its file's class.
  - The old "Pixel AUROC 0.64" came from all-anomalous dummy masks on Noise images. It was the image score repeated per pixel, not a localization measure.

## 2. Final comparison (test set)

**Optimal F1**, **Precision** and **Recall** are **macro** averages over the two classes (PD, Noise), measured at the threshold that maximizes macro F1 **on the test set itself**. This is the same style as the old v1 "Optimal F1".

- **These three are optimistic upper bounds**, because the threshold is tuned on test.
- **AUROC is threshold-free** and unaffected.
- The honest val-threshold F1/P/R are still computed and saved in `Results/Model_Comparison_v2/per_seed.csv` (columns `f1`, `precision`, `recall`).

| Method | Image AUROC | Window AUROC | Optimal F1 | Precision | Recall |
|---|---|---|---|---|---|
| **PatchCore** (WRN-50, L2+L3, K=8) | **0.853** [0.816, 0.872] | **0.811** [0.797, 0.823] | **0.799** [0.774, 0.863] | **0.870** [0.828, 0.896] | **0.774** [0.754, 0.844] |
| EfficientAD (K=8) | 0.724 [0.604, 0.736] | 0.691 [0.500, 0.747] | 0.672 [0.652, 0.697] | 0.693 [0.668, 0.694] | 0.714 [0.645, 0.719] |
| One-Class SVM (K=8) | 0.598 [0.597, 0.616] | 0.671 [0.667, 0.677] | 0.594 [0.590, 0.626] | 0.626 [0.608, 0.637] | 0.640 [0.623, 0.655] |
| *Baseline: 256-D kNN-5* | *0.860* | n/a | *0.796* | *0.797* | *0.795* |
| *Baseline: 256-D Mahalanobis* | *0.494* | n/a | *0.565* | *0.768* | *0.584* |
| *Baseline: always Noise* | *0.500* | *0.500* | *0.402* | *0.336* | *0.500* |

How to read it:

- **Seed ranges:** cells are median [min, max] over 3 seeds, and each column's median is taken separately. `per_seed.csv` has the matched values for each seed.
- **Macro F1 is the average of the two per-class F1 scores**, not the harmonic mean of macro P and R.
- **Pixel AUROC is N/A** because there is no pixel ground truth. Window AUROC replaces it.

## 3. What the results say

1. **PatchCore is clearly the best of the three detectors**: file AUROC 0.853 vs EfficientAD 0.724 and SVM 0.598.
2. **PatchCore does not beat the simple 256-D kNN-5 baseline on AUROC**: 0.853 [0.816, 0.872] against 0.860.
   - By the adoption rule in training plan §6.1, PatchCore is **not yet shown to be worth its cost over the hand-made feature**.
   - At the test-optimal threshold the two are also tied on F1 (0.799 vs 0.796).
3. **There is a large val→test drop** for every learned method (PatchCore 0.96 → 0.85, EfficientAD 0.80 → 0.72, SVM 0.90 → 0.60). The test set is small and clustered by date:
   - The 75 PD files come from 17 dates, and two dates hold most of the PatchCore errors.
     - 20231023: 6 PD files, 17% correct.
     - 20220427: 17 PD files, 59% correct.
   - Field PD Corona: 1 of 6 correct.
   - One odd date moves the metrics noticeably, so the confidence interval on the test numbers is wide.
4. **The threshold transfers poorly from val to test.** With the threshold chosen on val, PatchCore macro F1 is 0.727 [0.718, 0.846], against 0.799 at the test-optimal threshold. The score ranking is stable, but the val-chosen cut-off is not, so the optimal-F1 numbers above overstate deployable performance.
5. **EfficientAD's weakness is not the old teacher bug.** The v2 code seeds and saves the teacher, and it is still well below PatchCore.
   - Correction: at K=8 EfficientAD reaches 0.724 test AUROC (0.583 at K=4), so it is weak rather than at chance, and it is sensitive to training-data size.
   - Possible causes (untested): the model is a tiny random-teacher PDN trained for 5 epochs, about 2 minutes of training.
   - The published EfficientAD distils its teacher from a pretrained WideResNet. This code uses a **random** teacher, so the numbers say little about EfficientAD as published.

## 4. Model selection (step 6, val only)

| Run | Change from start config | Val AUROC | Val macro F1 |
|---|---|---:|---:|
| e4_k8 (seed 42) | K = 8 | 0.987 | 0.961 |
| final_k8 seeds 43 / 44 | K = 8 | 0.958 / 0.959 | – |
| e3_cs005 | coreset 5% | 0.957 | 0.903 |
| v2_start_s42 | start: WRN-50, layer2+3, coreset 1%, K=4 | 0.954 | 0.882 |
| e1_l23_k1 | K = 1 (control for E1) | 0.953 | 0.904 |
| e4_k2 | K = 2 | 0.900 | 0.792 |
| e1_l12_k1 | layer1+2, K = 1 | 0.887 | 0.799 |
| e2_resnet50 | ResNet-50 backbone | 0.881 | 0.791 |
| e3_cs0001 | coreset 0.1% | 0.748 | 0.669 |

**Lessons:**
- **K=8 was chosen on one seed, and that seed was lucky.** Seed 42 gave 0.987, while seeds 43/44 give 0.958–0.959, about the same as K=4 (0.954) and K=1 (0.953).
  - The K effect is small compared with seed noise, which is roughly ±0.03 (K=2 at 0.900 is probably partly noise too).
  - This repeats DIFFUSION.md's lesson: **repeat seeds before adopting.** Future selection runs should use 3 seeds from the start.
- **Settings that did clearly matter:**
  - layer2+3 beats layer1+2: 0.953 vs 0.887 at equal patch budget (K=1). The old layer1+2 choice was worse.
  - WideResNet-50 beats ResNet-50: 0.954 vs 0.881.
  - Coreset 1% beats 0.1%: 0.954 vs 0.748. Going to 5% adds little.
- `e1_l123_k1` (layer1+2+3) ran out of GPU memory at batch 64. It was rerun at batch 16 as `e1_l123_k1_bs16`; see that folder.

## 5. Engineering notes

- **Two memory fixes in `PatchCore/two_bank_common.py`:**
  - Chunked projection in the coreset step: the library otherwise puts the whole N×1024 feature pool on the GPU.
  - A preallocated feature pool instead of the library's list-then-concatenate, which halves peak host RAM.
  - With both, the K=8 PD bank peaks at 48 GB of host RAM and 6 GB of GPU memory.
- **Timing on 2× RTX 4090 (per seed):**
  - PatchCore: PD bank 33 min, Noise bank 9 min, scoring about 5 min.
  - EfficientAD: about 1 min per model.
  - SVM: about 10 s.
- **Server environment:** `danenv` has Python 3.11, torch 2.5.1+cu121 and faiss-gpu-cu12 1.10 (pip). The conda faiss 1.9 build needs glibc ≥ 2.32, but the server runs Ubuntu 20.04 with glibc 2.31.

## 6. Open issues / next steps

1. **Recover real dates for undated files.** 690 Field files (509 of them Noise) have `unknown_date` and were forced into train. Many filenames contain epoch-ms timestamps, so their dates can be recovered. That needs a fix in the Analyzer's date parser and a new `ai_data` build. It would give a larger and more representative val/test set.
2. **Make the test estimate more reliable.** Use date-grouped cross-validation over the Field dates instead of a single 15% test split. With only 17 PD test dates, the single split is noisy.
3. **Fix the threshold instability.** Options: calibrate `s` per bank (e.g. rank-normalize against each bank's train scores, training plan §9), or report at a fixed recall.
4. **Test whether PatchCore adds anything over kNN-5.** Run both inside the same date-grouped CV. If they stay tied, the 256-D kNN is far cheaper.
5. **Rebuild EfficientAD properly** (optional): a teacher distilled from a pretrained WideResNet, with more training. Only worth doing if a third learned detector is still needed.

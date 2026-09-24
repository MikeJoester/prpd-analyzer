# PatchCore v2: Training Plan (PD vs Noise, Two Memory Banks)

Plan written 2026-09-18. It trains on the data built by `PatchCore_data_plan.md` (`PatchCore/patchcore_data/prpd_v2/`).

## 1. Task

**Tell PD from Noise on Field data**, using the train/val/test split from the data plan:

- **Train:** 100% Lab PD + 100% Lab Noise + ~70% Field PD + ~70% Field Noise
- **Val:** ~15% Field PD + ~15% Field Noise
- **Test:** the remaining Field PD + Field Noise

PatchCore on its own only measures "distance from what it has seen". Two memory banks turn it into a PD/Noise decision:

```text
Bank_PD    = coreset of patch features from all train PD windows
Bank_Noise = coreset of patch features from all train Noise windows

For a test window w:
  d_PD(w)    = PatchCore anomaly score of w against Bank_PD
  d_Noise(w) = PatchCore anomaly score of w against Bank_Noise
  s(w)       = d_PD(w) − d_Noise(w)   # > 0: w looks more like Noise
```

The model's output is a **PD-vs-Noise evidence score**, not a fault diagnosis. Report it as such (CLAUDE.md, Phase 8).

## 2. Where it runs (CLAUDE.md Section 0)

- **Training runs on the GPU server only.** Get the SSH target from `.env` and run `conda activate danenv`.
- The server has 2× RTX 4090 (24 GB each). The two banks are independent, so **build them in parallel**:
  - `CUDA_VISIBLE_DEVICES=0` builds Bank_PD
  - `CUDA_VISIBLE_DEVICES=1` builds Bank_Noise
- Before the first run, check the environment on the server:
  - `torch.cuda.device_count() == 2` and a real kernel runs (`torch.zeros(1).cuda()`).
  - `faiss` has GPU support (`faiss.get_num_gpus() >= 1`). Otherwise the coreset and kNN steps fall back to CPU and are very slow.
  - `patchcore` imports with `PYTHONPATH=PatchCore/patchcore-inspection/src`.
- Copy `prpd_v2/` to the server (`rsync -a`) and check its `manifest.json` checksums before training.

## 3. Model configuration

### 3.1 Starting configuration

This is a starting point, not a final choice. Candidates are chosen on **val** (Section 5).

| Setting | Start value | Note |
|---|---|---|
| Backbone | `wideresnet50` (ImageNet) | Canonical PatchCore backbone; fits easily on a 24 GB card. Also try `resnet50` (the previous setting) as an ablation. |
| Feature layers | `layer2` + `layer3` | Canonical PatchCore choice. The previous `layer1+layer2` is an ablation candidate: PRPD blobs may favor shallow features, but that claim was never tested. |
| Input size | 224 × 224, resized from the 128×128 window | Resize = image size, **no center-crop**, so no phase bins are lost |
| Patch size | 3 | Library default |
| Pretrain / target embed dim | 1024 / 1024 | Library defaults. The old 256 was chosen for an 11 GB 2080 Ti, which no longer applies. |
| Coreset | `approx_greedy_coreset`, ratio **0.01** | 4090 memory allows 10× the old 0.001 |
| kNN for scoring | 1 | Canonical PatchCore |
| Train windows per file (K) | 4, seeded | Memory limit (3.2) |

### 3.2 Memory budget (why K = 4)

The library (`PatchCore._fill_memory_bank`) collects every patch feature before running the coreset. At 224 input with layer2 features, that is 28 × 28 = 784 patches per window.

| Bank | Train files | Windows (K = 4) | Patches | Raw features (1024-d fp32) |
|---|---:|---:|---:|---:|
| PD | ~1,435 + ~350 | ~7.1k | ~5.6M | ~23 GB → **too much** |
| Noise | ~156 + ~720 | ~3.5k | ~2.7M | ~11 GB |

Target dim 1024 with K = 4 is too big for the PD bank. Pick one of these, in order of preference:

1. **Target embed dim 512.** This halves the memory to about 11.5 GB for the PD bank. A small-scale val run must confirm the AUROC hit is under 0.01 before keeping it.
2. Use K = 2 for the PD bank only. Lab PD dominates it and its windows repeat heavily.
3. Build the bank in chunks: run a per-chunk coreset, then a final coreset over the union. This needs a small wrapper around `ApproximateGreedyCoresetSampler`.

Record which option was used in `run_config.json`.

### 3.3 Class balance

The two banks see different amounts of data (PD ≈ 2× Noise files). kNN distance barely depends on bank size once the coreset covers the space, but check this:

- Compare the val score distribution of `d_PD` on PD vs `d_Noise` on Noise.
- If one bank is systematically "tighter", adding a constant offset to `s` is enough. That is the threshold in Section 4.4, so no re-weighting is needed.

## 4. Pipeline

### 4.1 Scripts (new, in `PatchCore/`)

| Script | Does |
|---|---|
| `train_two_bank.py` | Reads `prpd_v2/manifest.csv`, builds one bank (`--cls pd` or `--cls noise`) with `patchcore.PatchCore.fit()`, and saves it with `save_to_path()` |
| `score_two_bank.py` | Loads both banks, scores every val/test window against each, and writes `window_scores.csv` + `file_scores.csv` |
| `evaluate_two_bank.py` | Picks the aggregation and threshold on **val**, applies them unchanged to **test**, and writes metrics + figures |

- These use the library API directly. The MVTec `run_patchcore.py` CLI assumes one "good" class and fits this task poorly.
- Each script takes `--data-root prpd_v2` and **checks that `ai_data_root` / `raw_data_version` in its `run_config.json` match the data folder** before running.

### 4.2 Commands (on the server, from the repo root)

```bash
conda activate danenv
export PYTHONPATH=PatchCore/patchcore-inspection/src:$PYTHONPATH
RUN=Results/patchcore/v2_$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=0 python PatchCore/train_two_bank.py --cls pd    --data-root PatchCore/patchcore_data/prpd_v2 --out $RUN --seed 42 &
CUDA_VISIBLE_DEVICES=1 python PatchCore/train_two_bank.py --cls noise --data-root PatchCore/patchcore_data/prpd_v2 --out $RUN --seed 42 &
wait
python PatchCore/score_two_bank.py    --run $RUN --splits val test
python PatchCore/evaluate_two_bank.py --run $RUN
```

Run long jobs under `tmux` or `nohup` so an SSH drop doesn't kill them.

### 4.3 Window → file aggregation

Each file has 28 window scores `s(w)`. Candidates for the file score `S`:

- `mean(s)`
- `max(s)`, i.e. "any Noise-like window"
- `min(s)`, i.e. "any PD-like window"
- **`median(s)`**
- 90th percentile

Choose on **val** by AUROC. Keep all candidates in `file_scores.csv` so the choice can be audited.

- Physical prior to check, not assume: PD often appears in bursts, so a file may be PD even if most windows look like noise. That would favor `min(s)`.

### 4.4 Decision threshold

- `S > τ` → Noise, otherwise PD.
- Pick `τ` on **val** by maximizing macro F1. Record balanced accuracy at that τ as well.
- **Never tune on test.** Test is scored once per final configuration.

## 5. Model selection on val

Everything below is chosen **on val only**. Change one thing at a time from the Section 3.1 start config.

| Experiment | Values |
|---|---|
| E1: layers | `layer2+layer3` (start) / `layer1+layer2` / `layer1+layer2+layer3` |
| E2: backbone | `wideresnet50` / `resnet50` |
| E3: coreset ratio | 0.001 / 0.01 / 0.05 (if memory allows) |
| E4: K (train windows per file) | 2 / 4 / 8 (with dim 512) |
| E5: aggregation | Section 4.3 candidates |

Adoption rules (same conventions as DIFFUSION.md §18.1):

- Keep the split fixed (`split_seed = 42`). **Repeat the final candidate with 3 training seeds** (42/43/44). The seed changes only the window subsample and the coreset init.
- Report the median and [min, max]. Don't adopt a change whose improvement is inside the seed range.
- Keep a results table per experiment with the val metrics, even for configurations that are not adopted.

## 6. Evaluation (test, once per final config)

Positive class = Noise (to match `s > 0` = Noise-like). Report:

| Metric | Why |
|---|---|
| **AUROC** (file level) | Threshold-free separability |
| **Macro F1**, **balanced accuracy** at val-chosen τ | Test has about 2× more Noise than PD, so plain accuracy and single-class F1 are inflated (as with the old 0.93–0.96 F1) |
| Per-class precision / recall | |
| Confusion matrix | `Results/patchcore/<run>/confusion_matrix.png` |
| **Breakdown by Field PD label** | Recall for Void and Floating separately. Report Corona with its tiny n; don't interpret Particle (n ≤ 2). |
| **Per-date results** | Detects whether one date drives the score |
| Relabelled files (13 Field PD → Noise) | Show their scores separately, as a sanity check on the change log |

### 6.1 Baselines (same split, same files, same metrics)

A PatchCore number means little without a cheap reference:

1. **Mahalanobis / kNN on the Analyzer 256-D feature** (mean + max per phase), with two class centers fit on train. This needs no GPU and can run locally.
2. **Majority class** (always "Noise"), to show what the imbalance alone gives.

PatchCore is only worth keeping if it beats baseline 1 on test AUROC and macro F1, by more than the seed range.

## 7. Outputs

```text
Results/patchcore/v2_YYYYMMDD_HHMMSS/
├── run_config.json        # data run_config hash, ai_data_root, raw_data_version, all model settings, seeds, K, dim option, git state
├── bank_pd/  bank_noise/  # save_to_path() output (faiss index + params)
├── window_scores.csv      # sample_id, window_idx, split, d_pd, d_noise, s
├── file_scores.csv        # sample_id, split, cls, label, date, S_mean, S_max, S_min, S_median, S_p90
├── val_selection.json     # chosen aggregation + τ + val metrics
├── metrics_test.json      # Section 6 metrics
├── per_date.csv  per_label.csv
├── confusion_matrix.png  score_hist_val.png  score_hist_test.png  roc_test.png
└── baselines.json
```

Memory banks go in the run folder, following the earlier refactor decision that PatchCore outputs live in `Results/patchcore/`. If the faiss files grow large, move `bank_*` to `artifacts/patchcore_banks/<run>/` and store the path in `run_config.json`.

## 8. Order of work

1. **Data**: run the data plan (build, split report, spot-check). *Local.*
2. **Environment check** on the server (Section 2).
3. **Smoke run**: 5% of train windows, dim 512, both banks, score 20 val files. Checks memory and timing and that the pipeline runs end to end. *Server.*
4. **Baselines** (Section 6.1) on val and test. *Local.* They set the bar before any PatchCore tuning.
5. **Start config** (Section 3.1) on val. *Server.*
6. **E1–E5** on val, one change at a time. *Server.*
7. **Final config × 3 seeds**, then test once, and write the report. *Server.*
8. Update `Plans/PatchCore.md` and `Results/Model_Comparison_Metrics.csv`. Mark the old rows as "v1 (random split, not comparable)".

## 9. Known risks

- **Small, uneven Field val/test.** About 75 PD files per split, spread over only a few dates. One odd date can swing the metrics, which is why per-date results are required (Section 6).
- **Lab dominates Bank_PD** (~80% of PD train files). The bank may represent Lab PD well and Field PD poorly, which is the Lab→Field gap already seen in earlier results.
  - If Field PD recall on val is low, try giving Field windows a larger K than Lab windows, so the coreset sees more Field variety.
  - Record this as an experiment, not a silent default.
- **The two banks measure distances on different scales.** This is handled by the val-chosen τ (Section 4.4). If the val histograms show heavy overlap, compare against a rank-normalized `s`, where each `d` is converted to a percentile against its own bank's train scores.
- **ImageNet features on PRPD images.** This is the core open question. The baselines in Section 6.1 are there to show whether a pretrained CNN adds anything over the hand-made 256-D feature.

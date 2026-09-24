# PRPD Denoising Diffusion (`prpd_diffusion`) Development Plan

This document is the design, development, and validation plan for the `Diffusion/prpd_diffusion` package — **a conditional diffusion model that removes noise from field PRPDs**. It inherits the data rules of the Analyzer (`claude.md`), and in case of conflicting rules, `claude.md` takes precedence.

Document baseline: 2026-08-21. The figures below are values actually executed and verified at this point in time.

---

## 1. Goal

Reduce the noise mixed in Field measurement PRPDs to restore a `128 × 3600` matrix close to a Lab-level PD pattern.

The final judgment criterion is not "the picture looks clean," but the following two quantitative conditions:

1. **paired**: In synthetic test pairs, the restoration error is smaller than the baseline where the input is left as is (`mae < baseline_mae`).
2. **unpaired**: When restoring actual Field PDs, the 256-dimensional feature distribution becomes closer to the Lab PD distribution
   (`frechet_after < frechet_before`, `mmd_after < mmd_before`).

The output of this model represents the **amount of noise reduction** and is not a physical failure judgment. It should be indicated as such in the UI and reports.

### What this model does not do

- Failure type classification (Role of Phase 7 classification model)
- Anomaly detection (Role of Phase 8 anomaly detection)
- Reinterpreting original `.dat` — It only consumes raw tensors verified and saved by the Analyzer.
- Modifying original data — Input folders are treated as read-only.

---

## 2. Input Data Contract

### 2.1 Consumed Structure

The original `.dat` is not reinterpreted. Only the execution folders verified and saved by the Analyzer are used.

```text
artifacts/ai_data_YYYYMMDD_HHMMSS/
├── metadata.parquet              # sample_id, group, label, date, raw_tensor_path, ...
├── raw_tensors_v1/{group}.npz    # key "raw", (number of group files, 128, 3600), uint8
├── run_config.json
└── manifest.json
```

`AiDataset` in `Diffusion/prpd_diffusion/contract.py` reads this folder and ensures the following:

- Validates metadata rows ↔ npz array index correspondence (`groupby("group").cumcount()`)
- Validates `sample_id` uniqueness (detects mixed dates/types)
- Validates that the number of metadata rows per group matches the raw tensor count
- Optional checksum re-verification via `verify_manifest()`
- Compressed npz → per-group `.npy` cache generation via `materialize_memmap()` (mmap reading in the training loop)

`tests/test_contract.py` directly compares the first/last sample of each group with the original `.dat` to verify if this correspondence is actually correct. **The original is read only for verification purposes and is never read in the training path.**

**Axis Convention (`CLAUDE.md` Section 2.1).** The `(128, 3600)` of the raw tensor is `(phase, time)`.
The `.dat` payload itself is stored in the order of `3600 cycle × 128 phase bin` (phase is the continuous axis), and the Analyzer's `read_dat` transposes it to match this convention. Thus, `matrix.mean(axis=1)` / `matrix.max(axis=1)` is the profile per phase, and `matrix[:, a:b]` is the time crop.

**`ai_data_*` execution folders generated before 2026-08-25 (`raw_v1`) have their axes reversed.** This is because the parser immediately did `reshape(128, 3600)` on the payload. See `STALE_AXIS_BUG.md` in each folder. New executions have `raw_data_version: raw_v2` and `payload_layout` items in `run_config.json`.

### 2.2 Which dataset to use — Independent selection per track

Do not scan the original `Data/` folder directly. **Use only the ai_data execution folders generated and verified by the Analyzer.**

**The diffusion track selects the dataset to use independently of the anomaly detection track.**
There is no constraint that both tracks must point to the same execution. The config of each track specifies its `ai_data_root`, and records that value in its run output (Section 11 of `claude.md`).

There are four execution folders under `artifacts/`, but **only one is usable.**

| Folder | Root | File Count | Groups | Remarks |
|---|---|---:|---|---|
| `ai_data_20260825_101830` | By Type | 3,172 | PD + Noise + Synthetic | **The only valid dataset.** `raw_v2`. Default for both tracks. Based on the figures in Sections 2.3 and 5 of this document |
| ~~`ai_data_20260820_172229`~~ | By Type | 2,021 | PD + Noise | **Invalid — axis inversion bug (Section 2.1)** |
| ~~`ai_data_20260820_170628`~~ | By Type | 3,172 | PD + Noise + Synthetic | **Invalid — axis inversion bug** |
| ~~`ai_data_20260821_160405`~~ | By Type | 1,384 | PD only | **Invalid — axis inversion bug** |

Invalid folders each have a `STALE_AXIS_BUG.md`. The files are kept only for historical preservation and are not used as training/evaluation input.

While both tracks may point to the same folder, **the pool is divided by the group selection in the config** — the 2D baseline uses 4 groups of PD + Noise (3,121 items), and ARDD 1D uses Lab PD + Field PD (1,958 items). This does not conflict with the rule of using only one execution per run. It is choosing a subset from one folder, not mixing folders.

Selection Rules:

- **Only one ai_data execution is used within a single run.** Do not merge files from multiple executions (Analyzer rules "use only one root", "no duplicate aggregation" in `claude.md`).
- If `ai_data_root` is left empty, `latest_run_folder()` picks the latest by name. As execution folders increase, the target changes, so **always specify it in the config.**
  `tests/test_contract.py` also targets the latest folder, so they look at the same folder.
- Record the selected value and that execution's `raw_data_version` in `run_config.json` and `dataset_description.txt`.
- If the dataset is changed, the composition and date splits/pool sizes in Section 2.3 also change. Re-run `prepare` on the changed execution to update that section, and record the comparison of metrics with the previous D4 results.

**What is valid only if the dataset for both tracks is the same** — Agreement is required only below.

- `s_noise`/`s_novel` decomposition in anomaly detection (`ANOMALY.md` Section 3.2, Phase A5): Valid only under the same `sample_id` set and the same date split. Anomaly detection `decompose` asserts the match of `ai_data_root` and `raw_data_version` between the two runs, and refuses execution if they differ.
- When placing restoration results and anomaly scores side-by-side in the same table or figure.

Other independent diffusion tasks (D1~D6 prepare, training, evaluation) are executed regardless of which dataset the anomaly detection track uses.

As of 2026-08-21, the anomaly detection track is unstarted with only the `EfficientAD/` folder created, so there is no target run to compare yet. Therefore, currently the diffusion track can freely choose the dataset without constraints, and when combination becomes necessary (Phase A5 in `ANOMALY.md`), the values of both runs can be matched.

Reference — The current default dataset is the entire original `Data/by_type` amount, so there is no difference by group.

| group | Original Folder | Default Dataset |
|---|---:|---:|
| Lab PD | 1,445 | 1,445 |
| Field PD | 513 | 513 |
| Lab Noise | 146 | 146 |
| Field Noise | 1,017 | 1,017 |
| Synthetic | 51 | 51 |
| Total | 3,172 | 3,172 |

The old default (2,021) was a filtered subset in the UI, so only half of Field Noise (507/1,017) was included. Since the diversity of the noise pool determines the generalization of this model, replacing it with the full execution is advantageous. If the scope needs to be narrowed or conditions changed, **do not mix folders within one run**, but generate a new ai_data execution in the Analyzer (`PRPD_Analyzer/scripts/build_ai_dataset.py`) and put that path in the config. This replacement can be decided independently by the diffusion track.

### 2.3 Current Default `ai_data_20260825_101830` (3,172) Composition — Measured on 2026-08-25

The figures below are the values when this execution is selected. If another generated dataset is selected, recalculate and update this table and the date split/pool sizes.

```text
group        label     files  dates
Field Noise  Noise      1017     57
Field PD     Corona       33      4
Field PD     Floating    130     32
Field PD     Particle      3      2
Field PD     Void        347     29
Lab Noise    Noise       146      7
Lab PD       Corona      286      8
Lab PD       Floating    286      8
Lab PD       Particle    311      9
Lab PD       Void        562      9
Synthetic    Noise        51      1
                        3172    114 (Total unique dates)
```

The `Lab PD` which will be the clean pool has 1,445 items, and the noise pool is `Lab Noise` 146 + `Field Noise` 1,017 = 1,163 items.

**Changes from the old default.** The previous default `ai_data_20260820_172229` (2,021) was not only invalidated by the axis inversion bug but was also a filtered subset in the UI. Since the new execution includes all 5 groups in their entirety, the clean pool grew from 1,062 → 1,445, and the noise pool from 637 → 1,163. All figures depending on this dataset, including the actual split values in Section 5 below, must be recalculated upon re-execution.

The 51 `Synthetic` items are included in this execution, but one of them is 0 across the entire range and has an abnormally high compression ratio (23 KB / 23.5 MB), meaning its properties differ from the measured data. It is not put into the clean/noise/evaluation pools, and if it is to be used, a separate quality check is done first.

---

## 3. Method: Creating paired data through measured noise synthesis

Since clean/noisy pairs from the same point in time do not exist, they are composed as follows.

```text
clean x0 = Lab PD raw matrix
noise n  = Lab Noise / Field Noise raw matrix   (Measured noise, not artificial Gaussian)
noisy y  = mix(x0, n)                           Default mixing = elementwise maximum
Training : model(cat[x_t, y], t) -> eps_hat     (Conditional diffusion in Palette / SR3 style)
```

Reason for keeping `maximum` as the default: One PRPS bin is the **peak amplitude** of that phase window, so when a PD pulse and noise occur in the same bin, the measured value is generally closer to the larger one.
If the detector characteristics differ, it can be changed to `additive` (255 clip) or `quadrature` (RMS) for comparison.

Noise augmentation (`data/noise_model.py`):

| Item | Default | Purpose |
|---|---|---|
| `gain_min` ~ `gain_max` | 0.6 ~ 1.4 | Diversify noise intensity |
| `phase_roll` | true | Mitigate fixed-phase bias of noise |
| `time_roll` | true | Mitigate time-axis alignment bias |
| `dropout_probability` | 0.05 | Train some batches with clean as-is → preserve identity |

The definition of the noise itself is also a subject of comparison. The `cin_noise` candidate using parametric noise (white + pink + power narrowband) instead of the measured pool, and missing injection `burst_missing` are in 18.3~18.4.
**Experiments that changed the noise model are not mixed with the algorithm comparison table** (18.1).

### Premises to confirm early on without fail

- **Lab PD is also not perfectly clean.** Since it contains its own noise floor, the training goal is not "remove all noise" but "reduce Field level → Lab level."
- First, confirm **whether the synthetic noisy resembles actual Field PD.** Measure the distance between the `mix(Lab PD, Field Noise)` distribution and the actual `Field PD` distribution in the 256 feature space, and if they differ significantly, adjust the gain range/mixing mode before starting training. Training without this verification easily leads to a "model that only erases synthetic noise well." → **Termination condition of Phase D1**

---

## 4. Representation

Do not input the entire `128 × 3600` into U-Net (460,800 pixels per sample, aspect ratio 28:1).

```text
crop_v1 (Default): Training uses (128, 256) random time window
                   Inference tiles the entire file by window → restores → concatenates again (no information loss)
```

- Value convention: Model input/output float32 `[-1, 1]`, storage uint8 `[0, 255]`. Round-trip conversion is lossless (`to_model_input` / `from_model_output`, verified in `tests/test_representation.py`).
- Since tiling is `3600 = 14 × 256 + 16`, 16 columns of `edge` padding are attached to the last window, and then cropped out when concatenating.
- Since noise reduction requires output of the same resolution as the input, downsampled representations are not used as training input.
  `time_pool` and `prpd_histogram` are auxiliary representations for visualization and distribution comparison.
- Since the phase axis (128) has a cyclic structure, the conv padding is set to `circular` only in the phase direction, and `zeros` in the time direction (`PhaseCircularConv2d` in `models/unet.py`).

---

## 5. Split Rules

- A single date belongs to **exactly one split** across all groups (`scope="global"`).
- The noise pool also follows the same rule. If a noise date used in train reappears in test, it is a leak.
- File-level random splitting is not used (Rule in `claude.md`).
- `data/splits.py` performs a **deterministic greedy** assignment considering (group, label) coverage, and `check_no_date_leakage` catches violations. The CLI throws an exception without starting training if a leak is found.

### Actual split results from `ai_data_20260825_101830` (3,172, seed=42)

```text
Dates: train 36 / val 45 / test 33        (Total 114 days, 0 leaks)

pool            clean   noise   real_noisy
train             982     820          356
val               256     162           77
test              207     181           80
```

It is normal for the ratio of the number of dates to the number of samples to be mismatched. This is because greedy assignment is based on (group, label) cell coverage and sample count, not the number of dates — large dates go to train, and several small dates go to val.

Known limitations — Must be explicitly stated in reports:

- There are only 3 `Field PD / Particle` in total, so it is split into train 1 / test 2. No statistical significance.
- `Field PD / Corona` (33 items) is assigned as train 21 / val 6 / test 6. The sample size is small.
- → Actual Field PD evaluation is only valid centering around `Void` (243/52/52) and `Floating` (91/19/20).

These limitations stem from the scope of the selected generated dataset. To mitigate this, specify a broader execution as `ai_data_root` or **generate a new ai_data execution** with broader conditions in the Analyzer (Section 2.2).
Since the above split figures depend on the current default execution, if the dataset is changed, re-run `prepare` to update them.

**Compared to the old default (2,021).** As the pool grew (clean 1,062 → 1,445, noise 637 → 1,163), `Field PD / Corona` appeared in all three splits, and `Particle` appeared in test.
However, the above figures are the first calculation after the axis fix, and **training has not been re-executed yet.**
Note that there are only 3 `Field PD / Particle` in the entire original folder, so it cannot be evaluated in any execution.

---

## 6. Model and Diffusion Design

### 6.1 Conditional U-Net (`models/unet.py`)

```text
Input : cat[x_t, y]  (B, 2, 128, W)
Output: eps_hat      (B, 1, 128, W)
```

| Element | Content |
|---|---|
| Channel | `base_channels=64`, multipliers `(1, 2, 4)` → 64/128/256 |
| Block | 2 ResidualBlocks per stage (+1 for up), GroupNorm + SiLU |
| Attention | 4-head self-attention in stage 2 and middle (operates on `128×256` → `32×64`) |
| Padding | Circular on phase axis / zero on time axis |
| Timestep | DDPM standard sinusoidal embedding → 2-layer MLP → bias injection per block |
| Output init | Initialize last conv weight/bias to 0 (stabilizes initial training) |

The number of parameters is recorded in the console and `dataset_description.txt` at the start of training.

### 6.2 Diffusion (`diffusion/schedule.py`, `gaussian.py`)

- Schedule: `cosine` default (Nichol & Dhariwal), `linear` selectable. `T=1000`.
- Objective function: epsilon prediction, `l2` default (`l1`, `huber` selectable).
- Since schedule coefficients are calculated with numpy, **unit testing is possible without torch** (`tests/test_schedule.py`).

### 6.3 Sampling (`diffusion/sampler.py`)

- DDIM abbreviated sampling (default 50 steps, `eta=0` → deterministic), DDPM full steps are also provided.
- `clip_denoised` clips x0 to `[-1, 1]` and then **readjusts eps as well** (error accumulates on mismatch).
- `denoise_full_file()` restores one file in window units and concatenates it as uint8.

This setup (Conditional DDPM + DDIM) is the **baseline for comparison**. The comparison experiment protocol and candidate list with other series techniques are in Section 18.

---

## 7. Training Procedure (`training/trainer.py`)

- optimizer: AdamW (`lr=1e-4`, `weight_decay=0.01`), grad clip 1.0
- EMA (`decay=0.999`) weights are saved together, and **EMA weights are used for sampling**
- Validation loss draws `t` and `noise` with a fixed-seed generator to enable comparison across epochs
- Validation pairs are also fixed with `set_epoch(0)`
- `last.pt` is saved every epoch, and `best.pt` is updated when val loss is minimal
- epoch/step/train_loss/val_loss/elapsed time are recorded in `train_log.jsonl`
- `num_workers=0` is safe on Windows (default)

---

## 8. Evaluation Metrics (`evaluation/metrics.py`)

| Category | Metric | Judgment Criterion |
|---|---|---|
| paired (synthetic test) | `mae`, `psnr`, `mean_profile_mae`, `max_profile_mae` | `mae < baseline_mae` (= `mae_improvement > 0`) |
| unpaired (actual Field PD) | Fréchet distance, MMD (256 feature) | `frechet_after < frechet_before`, `mmd_after < mmd_before` |

- `baseline_mae` is the error "when doing nothing" (input as is). If it's not lower than this, the model is meaningless.
- The 256-dimensional mean/max feature uses the Analyzer's `make_features` as is (definitions must be the same for comparison).
  This feature is for **evaluation and statistics** and is not used as deep learning input.
- Standardized statistics are always calculated only on the reference set (train Lab PD).
- If the sample size is smaller than the feature dimension (256), Fréchet distance is unstable, so MMD is examined together.
  The current defaults are `sample.max_files=32` and `--reference-files 64`, so **both metrics must be read together.**

`evaluation/report.py` creates a self-contained HTML containing metrics and figures (PRPD heatmap, phase profile comparison).

Evaluation axes that do not exist yet — **Performance curve by noise intensity** (SNR sweep) and **tolerance to missing injection**. Both protocols are brought from `ARDD-2025` and use the above table's metrics as is (Section 18.4). Regardless of the algorithm candidate, apply it to the baseline and record the point of collapse.

---

## 9. Execution

Below is the **2D baseline** execution procedure. The ARDD 1D track uses a separate CLI (Section 19.9).

```powershell
# 0) Dependencies (Install CUDA build first for GPU)
python -m pip install -r requirements.txt
python -m pip install -r requirements-diffusion.txt
# GPU: pip install torch --index-url https://download.pytorch.org/whl/cu128
#   Reason for cu128: The RTX 5080 on this PC is Blackwell (sm_120), so the cu124 build has no kernel.
#   After installation, do not just look at successful import, check actual kernel execution:
#     python -c "import torch; print(torch.cuda.get_device_capability()); print(torch.zeros(1).cuda())"
#   → (12, 0). Confirmed combination: torch 2.11.0+cu128 / RTX 5080 (2026-08-24)

# 1) Check data, split, and pool (torch not required)
#    The generated dataset to use is specified as data.ai_data_root in config (independent of anomaly detection setting).
#    To use another execution, create a config copy with only that value changed and specify it.
python -m Diffusion.prpd_diffusion.cli prepare --config Diffusion/configs/denoise_base.json --cache --verify-manifest

# 2) Confirm pipeline connection (a few minutes)
python -m Diffusion.prpd_diffusion.cli train --config Diffusion/configs/denoise_smoke.json

# 3) Main training
python -m Diffusion.prpd_diffusion.cli train --config Diffusion/configs/denoise_base.json --cache

# 4) Generate restoration results / evaluate
python -m Diffusion.prpd_diffusion.cli sample   --run Results/diffusion_runs/denoise_YYYYMMDD_HHMMSS
python -m Diffusion.prpd_diffusion.cli evaluate --run Results/diffusion_runs/denoise_YYYYMMDD_HHMMSS

# Tests
python Diffusion/tests/run_tests.py          # Or python -m pytest tests
```

`prepare` and `evaluate` work without torch. Only `train` and `sample` require torch.

Long training is executed as a separate process so it runs regardless of VS Code closing.

```powershell
Start-Process python `
    -ArgumentList "-m Diffusion.prpd_diffusion.cli train --config Diffusion/configs/denoise_base.json --cache" `
    -WorkingDirectory "C:\Users\USER\Documents\Yong\202608_PRPD" `
    -WindowStyle Hidden
```

---

## 10. Configuration Reference (`configs/denoise_base.json`)

This is the **2D baseline configuration**. The ARDD 1D track (`configs/ardd1d_*.json`) uses a separate set of dataclasses and does not have keys like `data.crop_width` — see Section 19 and `ardd1d/config.py`.
The reason they don't share configurations is that adding a key in one shakes the "unknown key" verification of the other.

| section | key | Default | Description |
|---|---|---|---|
| (root) | `seed` | 42 | Split, augmentation, sampling seed |
| | `run_prefix` | `denoise` | Result folder prefix |
| data | `ai_data_root` | `artifacts/ai_data_20260825_101830` | **Generated dataset this track will use. Specify independently of the anomaly detection setting.** If left empty, the latest folder by name is selected, changing the target whenever executions increase |
| | `memmap_cache` | `artifacts/repr_cache` | npz → npy cache. The actual path is `{cache}/{ai_data execution name}/{group}.npy` and is **separated per execution** (Modified 2026-08-24). Changing the dataset does not reuse the cache from the previous execution |
| | `crop_width` | 256 | Time-axis window width |
| | `samples_per_file` | 4 | Number of crops sampled from a file per epoch |
| | `clean_groups` | `["Lab PD"]` | Training target x0 |
| | `noise_groups` | `["Lab Noise", "Field Noise"]` | Measured noise for synthesis |
| | `real_noisy_groups` | `["Field PD"]` | Unlabeled for evaluation only |
| split | `ratios` | 0.7 / 0.15 / 0.15 | Target ratio by date unit |
| noise | `mode` | `maximum` | Comparison target for `additive`, `quadrature` |
| model | `base_channels` / `channel_multipliers` | 64 / `[1,2,4]` | |
| diffusion | `schedule` / `timesteps` / `loss_type` | cosine / 1000 / l2 | |
| train | `epochs` / `batch_size` / `learning_rate` | 100 / 16 / 1e-4 | |
| sample | `steps` / `eta` / `max_files` | 50 / 0.0 / 32 | DDIM setting, max files for evaluation |

`config_hash()` is the first 12 characters of the sha256 of the entire configuration. Used for execution identification and cache keys.
If there's an unknown key, it throws an exception immediately upon loading (to prevent ignoring typos).

---

## 11. File Structure

```text
202608_PRPD/
├── DIFFUSION.md                     # This document
├── claude.md                        # Analyzer plan (higher-level rules)
├── requirements-diffusion.txt       # torch, tqdm, pytest
├── configs/
│   ├── denoise_base.json            # 2D baseline training configuration
│   ├── denoise_smoke.json           # Minimum configuration for checking 2D pipeline connection
│   └── ardd1d_*.json                # Section 19 track: smoke / base(E1) / e2_linear / e3_morph
│                                    #                   / e4_resid / ardd(E5)
├── src/
│   ├── prpd_analyzer/               # Existing Analyzer (reused as read-only)
│   └── prpd_diffusion/
│       ├── contract.py              # ai_data_* loader/validation/memmap cache (separated per execution)
│       ├── cli.py                   # [2D] prepare / train / sample / evaluate
│       ├── data/
│       │   ├── representation.py    # crop/tiling/value conversion/auxiliary representation
│       │   ├── noise_model.py       # mixing operator, NoiseBank, augmentation
│       │   ├── splits.py            # Date unit split/coverage/leakage check
│       │   ├── pairs.py             # clean / noise / real_noisy pool composition
│       │   ├── torch_dataset.py     # PairedCropDataset, FullFileWindowDataset
│       │   ├── profile.py           # [Section 19] profile_v1 representation + Analyzer feature compatibility
│       │   └── cin_noise.py         # [Section 19] CIN generation/SNR calibration/burst missing (A4)
│       ├── diffusion/               # 1D/2D common — Section 19 track also uses it as is without modification
│       │   ├── schedule.py          # beta/alpha schedule (numpy, torch not required)
│       │   ├── gaussian.py          # q_sample, posterior, training loss
│       │   └── sampler.py           # DDPM/DDIM, denoise_full_file
│       ├── models/
│       │   ├── common.py            # group_norm(), timestep_embedding() — 1D/2D common
│       │   ├── unet.py              # Conditional 2D U-Net (phase-axis circular padding)
│       │   └── unet1d.py            # [Section 19] Conditional 1D U-Net + components on/off (A5)
│       ├── components/              # [Section 19] Paper technique unit (parameter-free)
│       │   ├── entropy.py           #   Local information entropy H(x,t) — Eq.(7)
│       │   ├── ardd_resid.py        #   Adaptive residual gate — Eq.(5)(6), A2
│       │   └── morph_attn.py        #   Morphological gradient attention — Eq.(9)~(15), A3
│       ├── ardd1d/                  # [Section 19] 1D track package
│       │   ├── config.py            #   Ardd1dConfig (seed / train_seed separated)
│       │   ├── pairs.py             #   PairFactory, profile pair cache
│       │   ├── dataset.py           #   torch Dataset, denoise_profiles
│       │   ├── metrics.py           #   Paired metrics by channel (mean/max separated)
│       │   ├── artifacts.py         #   run_config/description (D1 judgment auto-recorded)
│       │   ├── report.py            #   profile comparison/SNR curve figure
│       │   └── cli.py               #   prepare / synth-check / train / sample
│       │                            #   / evaluate / sweep / compare
│       ├── algorithms/              # Section 18 comparison candidates (traditional techniques, etc.) — **Unimplemented**
│       ├── ref/                     # Original paper PDF (Section 18.2 registry)
│       ├── training/trainer.py      # EMA, checkpoint, logs, deterministic val loss (1D/2D common)
│       ├── evaluation/
│       │   ├── metrics.py           # MAE/PSNR/profile, Fréchet/MMD (256 feature)
│       │   └── report.py            # Self-contained HTML report
│       └── runs/
│           ├── config.py            # dataclass configuration, config_hash
│           └── artifacts.py         # run folder, description, manifest
├── tests/                           # 128 passed / 0 failed / 0 skipped
│   ├── run_tests.py                 # Runner executable without pytest
│   ├── test_contract.py             # Actual artifacts + original .dat matching, cache separated per execution
│   ├── test_representation.py  test_noise_model.py  test_splits.py
│   ├── test_schedule.py  test_metrics.py  test_config.py  test_model.py
│   └── test_profile.py  test_cin_noise.py  test_components.py           # [Section 19]
│       test_model1d.py  test_ardd1d_config.py  test_ardd1d_pairs.py
└── artifacts/
    ├── ai_data_YYYYMMDD_HHMMSS/     # Input candidates (Do not modify). Select one with config
    ├── repr_cache/{ai_data execution name}/  # npz → npy memmap cache (separated per execution)
    ├── profile_cache/{ai_data execution name}/{noise setting hash}/   # [Section 19] profile pair cache
    └── diffusion_runs/
        ├── denoise_YYYYMMDD_HHMMSS/     # 2D baseline run
        ├── ardd1d_{e1..e5}[s{seed}]_YYYYMMDD_HHMMSS/   # [Section 19] run
        │   ├── run_config.json  split_config.json  dataset_description.txt
        │   ├── checkpoints/  samples/  train_log.jsonl
        │   ├── metrics.json  evaluation_report.html  manifest.json
        │   └── robustness.json  robustness_report.html   # upon sweep execution
        ├── ardd1d_synth_check.{csv,json,html}            # [Section 19] D1 validation artifacts
        └── comparison_YYYYMMDD_HHMMSS/   # 18.1 format candidate comparison results
```

---

## 12. Reproducibility Rules

- All executions are saved in the `denoise_YYYYMMDD_HHMMSS` folder and do not overwrite existing executions.
- The following 4 files must be kept:
  - `run_config.json` — The entire configuration + the finalized absolute path of the selected `ai_data_root` and that execution's `raw_data_version`
  - `split_config.json` — Actual date list, quantity per split, coverage table
  - `dataset_description.txt` — Human-readable description (input path, pool composition, mixing, environment, seed)
  - `manifest.json` — Artifact path/size/checksum (`.pt` checksum is omitted)
- `environment_info()` records the python/platform/numpy/pandas/torch versions and CUDA availability.
- Since sampling is probabilistic, record the number of DDIM steps/eta/seed/checkpoint name in `samples/sample_config.json`.
- Use **only one root** among by-date/by-type, and do not merge the two results.
- The generated dataset to use is selected independently by this track, but only one is used within a single run, and the selected value is recorded.
  Without this record, it cannot be known later which data the result came from.
- When combining or comparing side-by-side with anomaly detection results, first check if the `ai_data_root` and `raw_data_version` of the two runs are the same. If they differ, do not compare, and write down the fact that they differ in the report.

---

## 13. Test Plan

### Current Status (2026-08-25 Execution)

```text
136 passed, 0 failed, 0 skipped
```

Installed torch (2.11.0+cu128) to resolve the 6 skips in `test_model.py`, and 6 test files for the Section 19 track were added. On 2026-08-25, 8 cases in `tests/test_io.py` were added, making it 128 → 136.

### Covered Items

| File | Verification Details |
|---|---|
| `test_io.py` | **Payload axis direction (phase is continuous axis)**, Header 227/167 equality, Reject invalid size, Return array contiguous/writable, Real data autocorrelation lag128 > lag3600 |
| `test_contract.py` | Load actual artifacts, Match group count, sample_id uniqueness, **Direct matrix match with original `.dat`** |
| `test_representation.py` | Lossless value round-trip, Crop range, Tile ↔ concatenate round-trip, Same profile definition as Analyzer |
| `test_noise_model.py` | Boundaries of 3 mixing modes (255 clip, maximum dominance), Gain scale, Same seed reproducibility, Reject invalid config |
| `test_splits.py` | No overlapping date assignments, All samples assigned, Ratio approximation, Same seed determinism, Config round-trip |
| `test_schedule.py` | beta monotonic/bounded, `alphas_cumprod` converges, posterior finite/non-negative, `t=0` restoration |
| `test_metrics.py` | Error 0 for same input, Report improvement over baseline, MMD approx 0, Standardization reference statistics |
| `test_config.py` | Config round-trip, List → tuple restore, Reject unknown keys, Load deployed config |
| `test_model.py` | U-Net output shape, skip connection, **Phase axis circular padding**, Loss backpropagation, DDIM range, full-file restoration |
| `test_profile.py` | Profile value round-trip, **`to_feature_vector` exactly matches Analyzer `make_features`**, Reject unimplemented kind |
| `test_cin_noise.py` | pink spectrum slope ≈ −1, white flat, **Phase axis harmonics exactly 1/3/5 periods**, SNR calibration accuracy, burst missing continuity, seed reproducibility |
| `test_components.py` | Entropy bounded to `[0,1]`/0 for constant input/circular equivariant, `a ∈ [α,2α]` (Lipschitz), Morphological gradient large only at pulse **edges**, Otsu threshold |
| `test_model1d.py` | 1D U-Net shape, **Phase axis circular equivariance (even with components on)**, Identical to baseline when components off, **Components do not increase parameters**, Reuse existing `ddim_sample` without modification |
| `test_ardd1d_config.py` | Load 6 deployed configs, **E1↔E5 differ only in algorithm axis**, Reject unknown keys |
| `test_ardd1d_pairs.py` | Extract profile after synthesizing raw region, seed reproducibility, Cache key independent of training hyperparameters, **Metrics separated by channel** |

### Additionally Required Tests

1. Synthetic noisy ↔ Actual Field PD distribution proximity regression test (Fixed baseline based on Phase D1 artifacts)
2. Discontinuity check of window boundaries in `denoise_full_file` (Distribution of differences between columns at the stitched points)
3. Date leakage check of noise pool — Currently `check_no_date_leakage` only looks at split assignments.
   Add assertion at the pool level (train noise dates ∩ test noise dates = ∅)
4. Empty pool defense of `sample` command (Prevent `np.stack` failure when test clean/real_noisy is 0)

---

## 14. Step-by-Step Development Order

D0 is common to both tracks and is completed. D1~D5 are **based on the 2D baseline** and remain unexecuted.
The corresponding steps of the 1D track (Section 19) were performed separately, so they are indicated in each item.

### Phase D0: Environment Preparation — **Completed (2026-08-24)**

1. `pip install -r requirements-diffusion.txt` (Install CUDA build first for GPU)
2. In `python Diffusion/tests/run_tests.py`, verify that the 6 skips in `test_model.py` turn into pass

Actual results:

```text
torch 2.11.0+cu128 / cuda_available=True / device_capability=(12, 0) / RTX 5080
128 passed, 0 failed, 0 skipped
```

- **`cu128`** must be used, not `cu124`. The RTX 5080 is Blackwell (sm_120), so there is no kernel in the cu124 build. `import torch` success alone is not enough; check up to `torch.zeros(1).cuda()`.
- An actual bug in the existing 2D `ConditionalUNet` was revealed at this stage — if `channels + skip_channels` of the decoder was not a multiple of 32 (e.g., 16+32=48), `nn.GroupNorm` failed to create. This was fixed to `group_norm()` in `models/common.py`. This was a bug unseen because `test_model.py` was skipped when torch was not installed.

### Phase D1: Pre-Synthesis Validation — **2D Incomplete / 1D Performed (19.5, failed to pass)**

Verify the premises of Section 3 numerically. Skipping this step makes the interpretation of all subsequent training results meaningless.

1. Calculate the 256 feature distribution distance between the `mix(Lab PD, Field Noise)` sample set and the actual `Field PD` set
   (`metrics.frechet_distance`, `mmd_rbf`, standardized based on train Lab PD)
2. Create a comparison table with 3 mixing modes (`maximum`, `additive`, `quadrature`) × gain range candidates
3. Reflect the closest setting to `denoise_base.json` and leave the rationale in `notes`

Completion criteria:

- A distance comparison table of the three mixing modes remains.
- In the chosen setting, the synthetic noisy distribution is closer to the actual Field PD than the "Lab PD ↔ Field PD" distance.
- If not satisfied, record that fact, and indicate subsequent results as "limited to synthetic noise."

The result of performing the same validation with CIN noise in the 1D track is in Section 19.5. **It did not pass, and the figures were invalidated by the 2026-08-25 axis fix.** The fact that emerged there, *"the Lab→Field gap in profile space is not the noise floor difference but the pulse density/size difference"*, **can no longer be used as a premise for 2D D1.** When executing 2D D1, confirm directly from the new dataset whether that directional difference actually exists first. If the conclusion diverges from the 1D side revalidation (19.5), record both results together.

### Phase D2: Dataset Finalization — **Incomplete**

1. Finalize the generated dataset to use and specify it in `ai_data_root` (Current default: `ai_data_20260825_101830`).
   It doesn't have to be the same as the anomaly detection track's selection.
2. Execute `prepare --cache --verify-manifest` → confirm split/pool/checksum
3. Check the coverage table and finalize the evaluable (group, label) list
4. If it is determined that the noise pool scope needs to be broadened, specify a broader existing execution or generate a new ai_data execution in the Analyzer, replace `ai_data_root`, and perform steps 1-3 again

Completion criteria: 0 date leaks, both train/val clean and noise pools are not empty, coverage table saved, absolute path of the selected `ai_data_root` and `raw_data_version` are recorded in `run_config.json`.
If planning to combine with anomaly detection results, check at this point whether they match the values of that run.

### Phase D3: Pipeline Connection Check — **Incomplete**

Pass the entire pipeline from train → sample → evaluate to the end using `denoise_smoke.json`.

Completion criteria: `run_config.json`, `split_config.json`, `dataset_description.txt`, `train_log.jsonl`, `checkpoints/best.pt`, `samples/*.npz`, `metrics.json`, `evaluation_report.html`, `manifest.json` are all generated in the run folder.

### Phase D4: Main Training — **Incomplete**

Train using `denoise_base.json` and check the metrics.

Completion criteria:

- `paired.mae_improvement > 0` (Improvement over baseline)
- `distribution.frechet_after < frechet_before` **AND** `mmd_after < mmd_before`
- If only one of the two conditions is met, it is not considered a failure, but specify which one improved in the report

### Phase D5: Result Interpretation and Visual Validation — **Incomplete**

1. Overlay the restoration results on the Analyzer's t-SNE to check if Field PD moves to the Lab PD cluster
2. Review whether the change in phase profile (mean 128 / max 128) before and after restoration is physically reasonable
   — Is the phase concentration structure of the PD pulse maintained? Is the signal itself not erased?
3. Check if artificial discontinuities are created at the boundaries of concatenated windows

Completion criteria: Include the figures and judgments of the above 3 items in `evaluation_report.html`.

### Phase D4b: Paper-Based Technique Comparison — **Performed in 1D track (Section 19) / 2D Unstarted**

Compare paper techniques and other series algorithms under the same conditions according to the protocol in Section 18.
Use the evaluation code (Section 8) as is, and put the paper techniques as on/off components of the baseline (Section 18.5).

| Item | Status |
|---|---|
| `ARDD-2025` components A1~A5 ablation (E1~E5, 3 seeds) | **Completed — 19.6** |
| SNR sweep & missing injection protocol | **Completed — 19.7** |
| Create `comparison_YYYYMMDD_HHMMSS/` | **Completed** — `comparison_20260824_ardd1d/` (11 runs) |
| Include `identity` baseline | **Completed** — The untreated baseline is included in all tables |
| `algorithms/` common interface, traditional techniques (`wavelet`, `vmd_emd`, etc.) | **Unstarted** |
| Connect A2/A3 to 2D backbone | **Unstarted** — Currently both components are only attached to the 1D backbone |
| Compare axis selection for A2/A3 (time axis vs phase axis 2D) | **Unstarted** — The 1D track experimented only on the phase axis |

Remaining procedure:

1. Organize the `algorithms/` interface, wrapping the current diffusion as an adapter
2. Execute starting with `identity` and traditional techniques (torch not required) — No need to wait for D4 main training
3. When a new paper is added to `ref/`, fill in the 18.2 registry and 18.7 form, and repeat the same procedure

The two methodological lessons learned from the 1D track apply equally to the 2D comparison.

- **If the schedule is different, ε prediction val losses cannot be compared with each other.** Because the loss weighting changes; in fact, linear had lower val loss but worse denoising metrics (19.6). Select candidates based solely on the metrics in Section 8.
- **3 seeds is not a formality.** A value that seemed to be the "only improving candidate" in a single run was revealed as noise when the seed was changed (19.6). A path that fixes split and synthetic noise while only changing model initialization, like `--train-seed`, is also needed for the 2D track.

Completion criteria: Paired/unpaired metrics, cost, and failure reasons per candidate are left in one table, and the rationale for adoption is recorded.
The `identity` baseline is always included in the table, and paper techniques are reported with their **component-level contributions** separated.

### Phase D6: Expansion — Unstarted

- Add a before/after restoration comparison screen to Streamlit
- Input restored data into Phase 7 classification model to evaluate if Lab → Field performance degradation is mitigated (`evaluation/downstream.py`)
- Review structure for processing the entire `128×3600` directly with latent diffusion

---

## 15. Risk Factors and Open Questions

| Item | Content | Response |
|---|---|---|
| **Synthetic ≠ Actual (Realized in 1D track)** | 1D track's D1 failed to pass (19.5). The more CIN was mixed, the further it got from actual Field PD, and the trained model worsened unpaired metrics (19.6) | Automatically mark all results "Limited to synthetic noise." **Confirm first in 2D D1 whether the same problem occurs with measured noise** |
| **Nature of Lab→Field Gap** | In profile space, Field PD has lower mean and higher max — rarer but larger pulses. This direction cannot be created by non-negative noise synthesis (19.5) | May need to re-evaluate noise synthesis itself. Re-evaluate `additive`, or consider introducing a degradation operation that lowers pulse density |
| Clean is not clean | Lab PD also includes noise floor | Define goal as "reduction to Lab level," specify in report |
| Lack of evaluation samples | `max_files=32`, feature 256 dimensions → Fréchet unstable | Use MMD in parallel, increase `max_files` if necessary |
| Field PD label bias | Corona all train, 3 Particles | Restrict evaluation to Void/Floating and specify |
| Sampling speed | 15 windows per file × 50 DDIM steps, sequential processing per file | Consider batching across files with `FullFileWindowDataset` |
| Windows memmap | Cannot delete if cache is open | Call `AiDataset.release()` before recreation |
| Meaning of metrics | Low MAE does not necessarily mean "PD is well preserved" | Judge together with profile MAE and visual validation |
| Dataset mismatch between tracks | Since both tracks choose their own dataset, combination and comparison with anomaly detection could quietly misalign | Confirm `ai_data_root` & `raw_data_version` match in combined executions. Do not combine if mismatched |
| ~~memmap cache contamination~~ | **Resolved (2026-08-24)**. `materialize_memmap` writes to `{cache}/{ai_data execution name}/{group}.npy` | Assert per-execution separation in `AiDataset.cache_folder()` and `tests/test_contract.py` |
| Document inconsistency on dataset change | Figures in 2.3 and 5 depend on a specific execution | If dataset changes, re-execute `prepare` to update the section and metric comparisons |
| **Mistake of choosing candidate by val loss** | If β schedule differs, ε prediction loss weighting changes, so val losses cannot be compared. Linear had lower val loss but worse denoising metrics | Candidate selection only by Section 8/19.4 metrics. val loss is for checking training progress within the same schedule |
| **Single seed judgment** | A value seen as the "only improving candidate" in a single run was revealed as noise upon changing seed (19.6) | 3+ seeds as per 18.1. But only change model init while **fixing split and synthetic noise** (`--train-seed`) |

---

## 16. Relationship with Analyzer / AI Roadmap

```text
Analyzer (claude.md)
  → Multiple ai_data execution folders (Verified 128×3600 raw tensor + metadata)
      ← The only input for this model. This track selects one of them
      → prpd_diffusion (Noise reduction)          ← This document
          → Phase 7 AI failure classification (Expects Lab → Field degradation mitigation)
          → Phase 8 Anomaly detection (Can use restoration residuals as additional basis)
          → Phase 10 Paper results (Before/after restoration comparison figures/tables)
```

Compliance rules:

- The 256-dimensional mean/max feature is for statistics, t-SNE, and evaluation, not deep learning input.
- By date / by type are different views of the same data, so do not sum or mix them.
- The generated dataset to use is chosen per track, but only one is used within a single run, and the choice is recorded.
  Require the same dataset only when combining and comparing with anomaly detection results.
- Do not automatically change the category. Even if the restoration result looks like another category, deliver it only as review material in the form of `suspect_score`.
- Restoration output is noise reduction results, not failure judgment.

---

## 17. Not Yet Built

- Streamlit UI (Before/after restoration comparison screen)
- Classification model integration downstream evaluation (`evaluation/downstream.py`)
- Noise pool date leakage assertion
- Latent diffusion (Direct processing of full `128×3600`)
- Training resume path — Currently trains from scratch
- **2D baseline D1~D4** — Only the 1D track in Section 19 has been run; 2D conditional U-Net has not been trained yet
- 2D track algorithm comparison — Traditional techniques in 18.3 (`wavelet`, `vmd_emd`, etc.) and `algorithms/` common interface remain unimplemented. The Section 19 track creates comparison tables with its own CLI
- Empty pool defense in `cli.py:251` (Item 4 in Section 13) — Still remains in the 2D track

Items resolved on 2026-08-24:

- Separation of memmap cache by ai_data execution → Separated into `{cache}/{execution name}/` via `AiDataset.cache_folder()`
- `ARDD-2025` components (adaptive residual & morphological attention) → Implemented in `components/` (Section 19)
- `cin_noise` synthesis → `data/cin_noise.py`
- SNR sweep and missing injection protocol → `ardd1d.cli sweep`
- `GroupNorm` channel count bug — `ConditionalUNet` creation itself failed if decoder's `channels + skip_channels` was not a multiple of 32. Fixed with `group_norm()` in `models/common.py` (Unseen before torch installation because `test_model.py` was skipped)

---

## 18. Candidate Algorithms and Comparison Experiments (Paper-Based)

It has not yet been confirmed whether the current baseline (Section 6, Conditional DDPM + DDIM) is the best for this data.
Compare various noise reduction techniques on the **same data, same split, same metrics** to leave evidence.

Papers are added by the user to `Diffusion/prpd_diffusion/ref/`, and there are currently 2 (18.2).
**Paper techniques are not ported whole, but broken down into components, toggled on/off individually for comparison** (18.4).
Only then can we know which component the improvement came from.
Do not copy figures from unread papers into this document.

### 18.1 Fair Comparison Protocol — Fixed before algorithms

Without this protocol, we measure "which experiment was advantageous" rather than "which technique is better."

- **Same generated dataset**: All candidates use the same `ai_data_root` (Section 2.2). If candidates use different datasets, comparison itself is invalid.
- **Same split**: Reuse the reference run's `split_config.json` as is. Do not recalculate splits per candidate.
- **Same synthetic noise**: Paired evaluation pairs are generated with the same seed and same `noise_model` settings.
  Candidates that changed noise synthesis are separated and indicated as a distinct experiment.
- **Same metrics**: Report both paired (`mae`, `psnr`, profile MAE) and unpaired (Fréchet, MMD) metrics (Section 8).
  It is common for only one to improve, so do not rank by a single metric.
- **Include baseline**: Always include "when input is left as is" in the same table. It is the baseline for improvement margin.
- **Repeated execution**: Execute by changing the seed at least 3 times and report the median along with min/max.
  Since sampling is probabilistic, single run differences have weak backing.
- **Record costs**: Record parameter count, training time, inference time per file, and GPU memory together.
  Do not decide adoption solely on metrics.
- **Record failures**: Leave divergent, OOM, and unexecuted candidates in the table along with reasons. Do not leave blank.
- One run folder = one algorithm. Record `algorithm.kind` and `algorithm.params` in `run_config.json`.
- Aggregated results are saved in `Results/diffusion_runs/comparison_YYYYMMDD_HHMMSS/` along with the list of referenced run_ids, `ai_data_root`, and split file paths.

```text
comparison_YYYYMMDD_HHMMSS/
├── comparison.parquet  comparison.csv   # Candidates × Metrics × Seeds
├── runs.json                            # Referenced run_ids, ai_data_root, split paths
└── comparison_report.html               # Table + Example restoration comparison figures
```

### 18.2 Paper Registry (`Diffusion/prpd_diffusion/ref/`)

Original paper PDFs and survey materials are placed in `Diffusion/prpd_diffusion/ref/`. When adding a new paper, add a row to this table, and link the component to be ported with the id in the candidate table of 18.3.

| paper_id | Literature | Location | Verified | Linked Candidate |
|---|---|---|---|---|
| `ARDD-2025` | Chen, Li, Long, Zou, Deng, *Cable partial discharge identification network based on adaptive residual diffusion denoising and morphological attention*, **Sci. Rep. 15:42848 (2025)**, doi:10.1038/s41598-025-25197-9 | `ref/1.Adaptive Residual Diffusion Denoising (ARDD).pdf` | **Original verified** | `ardd_resid`, `morph_attn`, `cin_noise` (18.3) |
| `SURVEY-5` | Internal survey material *Executive Summary — diffusion noise model* (Comparison of 5 papers) | `ref/Executive Summary_diffusion noise model.pdf` | **Original verified** (Secondary source) | 18.2b list |

Notation caution: The survey material wrote the ARDD paper as "Wu et al. (2025)", but **the author of the PDF original is Long Chen et al**. Follow the original notation when citing.

#### 18.2b 5 Candidates Selected by Survey Material (Originals unobtained)

Copying only the priorities and main points from `SURVEY-5`. **Since the originals for the 4 papers other than ARDD have not yet been obtained, do not cite figures or detailed settings.** When needed, add the PDFs to `ref/` and elevate them to the 18.2 table.

| Rank | Paper | Characteristics of Noise Model | Distance from Our Data |
|---:|---|---|---|
| 1 | Chen et al. (2025) ARDD | Gaussian + Linear β(1e-4→0.02), T=1000, DDIM 50 | **Closest.** PD domain, 1D waveform |
| 2 | Yang & Deng (2026) Time–Frequency Diffusion | Alternating time-axis Gaussian and frequency-axis blur | Transformer vibration/ultrasound. Time-axis blur is worth considering porting to PRPD |
| 3 | Yi et al. (2024) TSDM | Standard DDPM, 1D U-Net (attention) | Bearing vibration generation. For augmentation |
| 4 | Wang et al. (2024) IResUnet-DM | Standard DDPM series, Residual U-Net | Generation for data imbalance compensation |
| 5 | Crabbé et al. (2024) Frequency-domain diffusion | Frequency domain SDE (mirror Brownian) | General time series. Advantageous for spectrally sparse signals |

Since our task is **noise reduction, not generation or augmentation**, 3rd and 4th places (for generation) have low priority.
2nd and 5th places are grouped as the `freq_noise` candidate in 18.3 because "noise is defined in domains other than the time axis."

### 18.3 Candidate Algorithms

Keep the families broad, but be sure to include traditional techniques that do not require training. There are actually cases where deep learning cannot beat traditional techniques, and that fact itself is a result.

| id | Family | Training | Input | torch | Paper Basis | Status |
|---|---|---|---|---|---|---|
| `identity` | baseline (untreated) | Unneeded | raw | Unneeded | — | Must include |
| `wavelet` | Traditional · wavelet threshold | Unneeded | raw | Unneeded | TBD | Candidate |
| `vmd_emd` | Traditional · Mode decomposition (VMD/EMD) | Unneeded | raw | Unneeded | TBD | Candidate |
| `sparse_dict` | Traditional · Sparse representation / dictionary learning | Data-based | raw | Unneeded | TBD | Candidate |
| `morph_median` | Traditional · Morphology / median filter | Unneeded | raw | Unneeded | TBD | For lower bound check |
| `unet_reg` | Supervised regression (DnCNN/U-Net) | Needed | Synthetic pairs | Needed | TBD | Candidate |
| `dae` | denoising autoencoder | Needed | Synthetic pairs | Needed | TBD | Candidate |
| `self_sup` | Self-supervised (Noise2Noise/Void series) | Needed | Measured solo | Needed | TBD | Candidate — Avoids synthesis dependence |
| `ddpm_ddim` | **Current baseline** · Conditional diffusion | Needed | Synthetic pairs | Needed | Structurally same family as `ARDD-2025` | Implemented |
| `ardd_resid` | Entropy-based adaptive residual connection | Needed | Synthetic pairs | Needed | **`ARDD-2025`** Eq.(5)~(8) | Candidate · 18.4 |
| `morph_attn` | Morphological gradient attention (Parameter-free) | Unneeded | Intermediate feature | Needed | **`ARDD-2025`** Eq.(9)~(15) | Candidate · 18.4 |
| `freq_noise` | Noise defined in frequency domain (blur/mirror SDE) | Needed | Synthetic pairs | Needed | 18.2b 2nd/5th (Original unobtained) | Candidate — Low priority |
| `cold_diffusion` | Non-Gaussian degradation diffusion | Needed | Synthetic pairs | Needed | TBD | Candidate |
| `score_sde` | score-based SDE | Needed | Synthetic pairs | Needed | TBD | Candidate |
| `fast_sampler` | consistency/flow series (Step reduction) | Needed | Synthetic pairs | Needed | TBD | Candidate — Responds to sampling cost |
| `latent_diff` | latent diffusion | Needed | Compressed repr. | Needed | TBD | Unimplemented item in Section 17 |
| `unpaired_gan` | unpaired domain translation | Needed | Field/Lab solo | Needed | TBD | Candidate — Avoids synthesis assumption |

Noise model candidates (Managed separately as it's a **data-side change**, not an algorithm. See Section 3):

| id | Content | Paper Basis | Status |
|---|---|---|---|
| `measured_pool` | Mix of measured Lab/Field Noise pool (**Current default**) | — | Implemented |
| `cin_noise` | Composite Industrial Noise = White + pink (1/f) + Power narrowband (fundamental + 3rd/5th harmonic) | **`ARDD-2025`** Robustness experiment | Candidate · 18.4 |
| `burst_missing` | Missing injection turning continuous sections to 0 (5~15%) | **`ARDD-2025`** Missing experiment | Candidate — For evaluation |

Priorities: `identity` → 2 traditional techniques → `unet_reg` → current baseline → **`ARDD-2025` components (18.4)** → The rest.
Some traditional techniques and supervised regression can be run without torch, so they can be run before D4 main training.

### 18.4 `ARDD-2025` Component Decomposition and Porting Plan

The paper's proposed model (ARDDMA-Net) is a **2-stage structure**. The 1st stage is ARDD noise reduction + morphological attention, and the 2nd stage is the MSA-ResNet-1D classifier. **This track only brings the 1st stage.** The 2nd stage classifier is the domain of Phase 7 in `claude.md`, so it is not implemented here.

#### Parts Already Identical to Our Baseline (Nothing to Port)

| Item | Paper | Current Implementation | Judgment |
|---|---|---|---|
| forward diffusion | `x_t = √ᾱ_t·x0 + √(1−ᾱ_t)·ε` | Identical to `diffusion/gaussian.py` | Identical |
| Objective function | ε prediction, L2 | `loss="l2"` default | Identical |
| Step count | `T = 1000` | `T = 1000` | Identical |
| β schedule | Linear `1e-4 → 0.02` | `cosine` default, `linear` selectable | **Use as comparison axis** (A1 below) |
| Inference sampling | DDIM 50 steps | DDIM 50 steps default | Identical |
| Backbone | 1D U-Net (Conv1D+BN+ReLU, 4 stages) | 2D U-Net (Phase axis circular) | **Structure differs** (A5 below) |

In other words, the diffusion process itself is already the same as the paper. The paper's unique contributions are **adaptive residual, morphological attention, and composite noise experiments**, and these three are the actual porting targets.

#### Porting Targets and Experiment IDs

| Experiment ID | Component | Paper Basis | Implementation Location | Status (2026-08-24) |
|---|---|---|---|---|
| A1 | Replace β schedule with `linear(1e-4, 0.02)` | Eq.(1) setting | One line config | **Implemented/Executed** (Section 19 E2) |
| A2 | `ardd_resid` — Adjust residual strength `a(x,t)=α(1+H/H_max)` via local info entropy | Eq.(5)~(7) | `components/ardd_resid.py` + `entropy.py` | **Implemented/Executed** (E4) |
| A3 | `morph_attn` — Sigmoid after max fusion of multi-scale morphological gradients (dilation-erosion) | Eq.(9)~(15) | `components/morph_attn.py` | **Implemented/Executed** (E3) |
| A4 | `cin_noise` — White+pink+power narrowband parametric noise | Robustness experiment | `data/cin_noise.py` | **Implemented/Executed** (Section 19 Default Noise) |
| A5 | 1D U-Net vs current 2D U-Net | Paper backbone | `models/unet1d.py` | **Implemented/Executed** — Main body of Section 19 track |

A5 was originally "low priority," but as we decided to reproduce the paper on a 1D signal, it became **the backbone of this track itself**. However, it uses the **phase profile axis (128)**, not the paper's 1D axis (time 3,600) — see Section 19.
A2 and A3 have not yet been attached to the 2D `models/unet.py` (only connected to the 1D backbone).

**The axis selection for A2 and A3 is a key issue.** The paper's signal is a 1D time waveform, and entropy/morphology operations are defined on the time axis. Since our input is `(phase 128, time 3600)`, we must explicitly decide and experiment as follows:

- Time axis only: 1D operation on each phase row (most faithful to paper)
- 2D including phase axis: `(3×3)` 2D dilation/erosion — Since the phase axis is cyclic, circular processing is required
- Both results are recorded, and the axis selection itself is left as a comparison item

#### Do Not Transfer Paper Figures to Our Metrics

The evaluation in the paper is **classification accuracy** (98.36%, ablation table: baseline 75.81 / WT 86.52 / SVD 80.25 / PCA 82.31 / DM only 91.34 / AR only 81.44 / AR+DM 98.36), and noise reduction quality itself (MAE, distribution distance) is not measured. Our judgment criteria are the paired/unpaired metrics of Section 8.
**Do not set the accuracy figures of the paper as our improvement targets.**

However, the ablation design is borrowed as is: turning on components one by one to separate their respective contributions, which does not conflict with the comparison protocol of 18.1.

#### Evaluation Protocol to Bring from Paper (Conditions, not metrics)
| Protocol | Paper Setting | Our Application |
|---|---|---|
| SNR Sweep | Inject CIN from +3 → −12 dB, 3 dB intervals | Redefine noise gain of synthetic test pairs based on SNR and perform identical sweep. Use our metrics |
| Missing Injection | Turn 5~15% of continuous sections to 0 | Turn time-axis window to 0 and record restoration performance change |

These two protocols are currently absent axes. **Apply them to the baseline regardless of the algorithm candidate** to record "the point where it breaks down when noise becomes severe."

### 18.5 Execution Interface (Design)

Keep a common interface so as not to modify the baseline code when adding candidates.

```text
Diffusion/prpd_diffusion/algorithms/
├── base.py        # class Denoiser: fit(pairs) -> None, denoise(x) -> x_hat
├── classical.py   # wavelet, vmd, morph — torch not required
├── regression.py  # unet_reg, dae
├── diffusion.py   # Adapter wrapping current baseline in this interface
└── components/    # On/off units for paper techniques (ardd_resid, morph_attn, ...)
```

- Put `algorithm.kind` and `algorithm.params` in config, and throw exceptions on load for unknown keys.
- **Put paper techniques not as separate algorithms but as on/off options of the baseline.** E.g.:
  `{"kind": "ddpm_ddim", "components": {"ardd_resid": {"alpha": 0.5}, "morph_attn": {"scales": [3, 7, 15], "axis": "time"}}}`
  By doing this, ablation is created entirely by config combinations, and which components were turned on is left in `run_config.json`.
- All candidates output uint8/float of the same shape as input (`128 × W`). Evaluation code is common.
- Lazy import only candidates requiring torch (`classical.py` uses only numpy/scipy).
- Noise model candidates (like `cin_noise`) go into `noise_model` config, not `algorithm`.
  Do not mix algorithm comparison and noise model comparison in the same table (Section 18.1 "Same synthetic noise" rule).

### 18.6 Adoption Rules

- Do not adopt if it fails to improve **both** paired and unpaired metrics compared to baseline (`identity`).
- If the improvement margin is within the seed variation range, report as "No difference." Do not count decimal improvements as a win.
- If inference cost increases significantly, note the improvement and cost together to leave a basis for judgment.
- If the reported performance of the paper and our results differ, first check and record differences in data/preprocessing/evaluation protocols.
  Do not immediately conclude "that technique is bad" upon a reproduction failure.
- The final choice is made by reviewing D5 evaluation metrics and visual verification together, and the rationale is left in the comparison report.
- **Adopt/reject paper techniques on a component basis.** Even if the paper reported it to be good, if it has no contribution on our data, turn off only that component. Leave the rejected component and reasons in the comparison report.

### 18.7 Paper Organization Form

When adding a paper to `ref/`, fill in the following and link to the 18.2 table and 18.3 candidate id.

```text
paper_id
Title / Source / Year / DOI
Target Domain            # PD/PRPD, general image, audio, vibration, etc.
Components of Proposed Technique # Broken down list, not the whole model
Diffusion Setting        # T, β schedule, prediction target, sampling method
Noise Model              # What was defined as noise
Input Representation and Preprocessing
Data Used and Scale
Evaluation Method and Metrics # Explicitly note if same/different from our metrics (Section 8)
Applicability to Our Data # 128×3600, uint8, phase axis cyclic structure
Porting Cost             # Reimplementation difficulty, required libraries
Verification Status      # Original verified / Abstract only / Unverified
```

Do not transcribe figures from papers with "Unverified" `Verification Status` into the document body.
Figures from papers with evaluation metrics different from ours (e.g., reporting only classification accuracy) are **not set as improvement targets but used only as design references** (Section 18.4 `ARDD-2025` example).

---

## 19. ARDD 1D Track (`Diffusion/prpd_diffusion/ardd1d/`) — 2026-08-24

An independent track that reproduced the technique of the paper (`ARDD-2025`) on a **1D signal**. The 2D baseline (Section 6, `models/unet.py`, `cli.py`) was not modified, and the metric/run artifact formats were shared so they could be placed side-by-side under the comparison protocol of 18.1.

### 19.1 Premises and Limitations — Read First

| Item | Choice | Rationale |
|---|---|---|
| Input | Phase profile `(2, 128)` = mean 128 + max 128 | User specified |
| dataset | `ai_data_20260825_101830`, select only Lab PD + Field PD with `clean_groups`/`real_noisy_groups` (1,958) | Only valid execution after axis fix (Section 2.2) |
| Noise | Parametric CIN | No noise group selected + Method actually used by paper |
| Backbone | 1D U-Net, phase-axis circular padding | A5 of 18.4 |

> **2026-08-25.** All existing training/evaluation results of this track (19.5·19.6) are invalidated due to the axis inversion bug.
> The above dataset is a setting for re-execution and **has not been re-executed yet.**

**Always note three limitations along with the results.**

1. **Explicit exception to Section 11 of `claude.md`.** That document stipulates that the 256-dimensional mean/max feature is not used as deep learning input. This track intentionally deviates from that rule to verify the technique on a 1D signal like the paper. Leave `feature_input_exception: true` in `run_config.json` of each run, and **do not pass the output of this track as input to the classification/anomaly detection tracks.**
2. **It is irreversible.** Since it is a `3600 → 1` reduction, the restored profile cannot be reverted to `128×3600`.
   Therefore, the raw-space `mae`/`psnr` of Section 8 are not applied, and are replaced by profile space metrics (19.4).
3. **Do not combine the mean and max channels.** Their properties are completely opposite — see 19.4.

### 19.2 Representation (`data/profile.py`)

```text
profile_v1 : (128, 3600) uint8 → (2, 128) float32
             Channel 0 = time mean per phase, Channel 1 = time max per phase
```

> **2026-08-25 Axis Fix.** This is when the input `(128, 3600)` actually became `(phase, time)` (Section 2.1). Prior to this, the parser read the axes flipped, so the axis folded by this representation was not the time axis, and as a result, **the input to this track itself was not a phase signal.** All figures in 19.5/19.6 came from that condition and are therefore invalid.

- Concatenating the two channels results in **exactly the same vector** as the Analyzer's 256-dimensional feature (`to_feature_vector` ↔ `prpd_analyzer.features.make_features`, asserted in `tests/test_profile.py`).
  Thanks to this, the unpaired distribution metric can use the Analyzer definition exactly.
- **Do not quantize to uint8.** The mean channel is essentially a decimal value, so rounding it changes the value from `make_features` and distribution comparison becomes invalid. Storage is `[0, 255]` float32, model input/output is `[-1, 1]`.
- `phase_row_v1` (1 phase row = length 3600, matches paper length and is lossless) has only an interface and is unimplemented. If limitations 2 and 3 prove to be bottlenecks, switch here.

### 19.3 CIN Noise (`data/cin_noise.py`) — Axis Mapping is Key

This track uses `clean_groups`/`real_noisy_groups` to select only Lab PD + Field PD, so it does not use a noise group. Instead, it makes the paper's Composite Industrial Noise parametric.
**This is not a workaround, but the method actually used by the paper.**

(Pre-edit description: "Because the latest dataset has 0 `Lab Noise`/`Field Noise`" — that dataset `ai_data_20260821_160405` was invalidated by the axis bug. The new default `ai_data_20260825_101830` has 1,163 noise groups, but this track still does not select them for CIN comparison.)

The paper's signal is a single time waveform, but in our raw, the meaning of the two axes is different:
`128 phase bins = 1 AC period`, `3600 time columns = cycle index (about 1 minute)`.

| Component | Paper | Our Axis | Rationale |
|---|---|---|---|
| White | Time axis | Both axes i.i.d. | Background thermal noise |
| Pink (1/f) | Time axis | **Time axis** (drift between cycles) | Low frequency electronics noise |
| Fundamental | 50 Hz | **Phase axis 1 cycle** | Since 128 bins = 1 AC period, the power frequency is fixed to the phase |
| 3rd/5th Harmonic | 150/250 Hz | **Phase axis 3rd/5th cycle** | Identical |

Take the envelope `|·|` of the signed noise field to make it uint8, then mix it with the existing `noise_model.mix` (Since PRPS bins are peak amplitudes, there is no concept of negative numbers). Axis selection is a design judgment, so alternatives can be selected via `harmonic_axis`/`pink_axis` and are recorded in all run descriptions.

> **The table above applies only to the signed field before `|·|` (Measured 2026-08-25).**
> Full-wave rectification halves the period, so in the phase profile actually seen by the model, the 1st/3rd/5th order components are **exactly 0**, and the energy shifts to the 2nd/4th/6th orders.
>
> ```text
> Signed field     k1 0.625  k3 0.250  k5 0.125    (k2/k4/k6 = 0)
> Profile after |·| k1 0.000  k3 0.000  k5 0.000    k2 0.489  k4 0.293  k6 0.053
> ```
>
> `tests/test_cin_noise.py::test_harmonics_land_on_phase_axis_orders` only looks at the `harmonic_field` (signed component), so it fails to catch this gap. The phase profile of the noise groups measured after the axis fix has a 1st/3rd/5th order median proportion of **0.06** (`Lab Noise` 0.062, `Field Noise` 0.063), which is low in the first place, so it cannot be concluded that the rectified CIN is vastly misaligned with reality. **However, do not read the axis mapping table as a description of "the noise the model sees."** To preserve orders, an offset (`field + c`) method should be compared instead of `|·|`, and this will be decided upon re-execution in 19.8.

### 19.4 Metrics (`ardd1d/metrics.py`)

paired is calculated **per channel** in the profile space: `{mean,max}_profile_mae`, `..._psnr`, and `..._improvement` over untreated. unpaired calls the **same function** as the 2D track (`evaluation.metrics.distribution_shift`) as is.

**Why channels shouldn't be combined** — Untreated baseline (The figures below are based on the invalidated `ai_data_20260821_160405`. The principle itself that channels must be separated is valid because it comes from the nature of the representation, but the specific values must be remeasured):

```text
mean channel baseline MAE = 6.79   ← CIN raises the time-axis noise floor, causing high contamination
max  channel baseline MAE = 3.45   ← PD peak is larger than noise, so maximum synthesis barely changes it
```

In other words, under `maximum` synthesis, **contamination is virtually loaded only on the mean channel.** The max channel is almost clean from the start, so an "improvement margin of 0" is normal, and it should not be read as a failure.

### 19.5 D1 Pre-Synthesis Validation — **Failed to pass / Invalidated 2026-08-25**

> **All figures in this section are invalid.** The values below were calculated on a mixed-axis profile (Section 2.1). This is because `matrix.mean(axis=1)` was not the time mean per phase, but the mean of a row where different phases were mixed, shifted by 16 bins each. In particular, **do not use the observation that "Field PD has lower time mean and higher time max" and the resulting conclusion that "the Lab→Field gap is not a noise floor difference" as the design rationale for other tracks.**
>
> Re-validation method: Run `cli synth-check` again with the new dataset `ai_data_20260825_101830`. The conclusion may change or remain the same — **assume neither until verified.** The contents below are left only as history to compare against re-execution results.

Measured the 256 feature distance between `mix(Lab PD, CIN)` and actual `Field PD` via `cli synth-check`. The baseline is the `Lab PD ↔ Field PD` distance with nothing mixed.

```text
Baseline (Lab PD ↔ Field PD)    frechet  36.60   mmd 0.0076
maximum, CIN amplitude × 0.05   frechet  36.66   mmd 0.0078
maximum, CIN amplitude × 0.10   frechet  38.31   mmd 0.0104
maximum, CIN amplitude × 0.30   frechet  55.32   mmd 0.0414
maximum, CIN amplitude × 1.00   frechet 242.85   mmd 0.3160
```

**The more CIN is mixed, the further it gets from actual Field PD.** Diagnosis explains the reason.

```text
mean channel: Lab 3.87 → Field 2.97  (-0.90)   Field is lower
max  channel: Lab 121.42 → Field 127.08 (+5.66) Field is higher
```

Field PD produces **rarer but larger pulses in the time axis** than Lab PD. Mixing non-negative noise raises both channels together, so this directional difference cannot be reproduced.
**In other words, the Lab→Field gap in the profile space is not a "noise floor" difference.**

According to the rule in D1 of Section 14, all training results of this track are marked as **"Limited to synthetic noise."**
`ardd1d/artifacts.py` reads the `synth-check` artifacts and automatically copies this judgment into `dataset_description.txt`, `run_config.json`, and `metrics.json` of all runs.

### 19.6 ablation Results — **Invalidated 2026-08-25**

> **All figures for E0~E5 are invalid.** Trained and evaluated with a mixed-axis profile (see warning in 19.2).
> The relevant run folders are marked with `Results/diffusion_runs/STALE_AXIS_BUG.md`.
> The table below is left only as history to compare against re-execution results.
>
> Data-independent **methodological conclusions are valid**: If the β schedule differs, ε prediction val losses cannot be compared (→ select by final metrics), do not adopt candidates based on a single seed, and report channel-specific improvement margins separately.

`ai_data_20260821_160405`, 64 dates → train 46 / val 8 / test 10, pool clean 787/104/171, real_noisy 229/47/46. Trained 200 epochs, DDIM 50 steps, test 128 files.

E1, E2, and E5 run **3 seeds** (42/43/44) per the 18.1 protocol and record the median and [min, max] together. The seed only changed **model initialization and batch order** via `--train-seed` — date splits and synthetic noise realization are fixed with `config.seed=42`, so comparison conditions between candidates are identical.
E3 and E4 are single runs, so the extent of seed variation is unknown (they are treated accordingly weaker in the judgments below).

| id | schedule | Components | Seeds | mean MAE Median [min, max] | max MAE Median [min, max] | Fréchet After | MMD After |
|---|---|---|---:|---|---|---:|---:|
| **E0** | — | **identity(untreated)** | — | **6.79** | **3.45** | **107.50** | **0.0586** |
| E1 | cosine | None | 3 | **1.81** [1.72, 1.97] | 3.49 [2.96, 3.64] | 140.75 | 0.0759 |
| E2 | linear (A1) | None | 3 | 2.25 [2.19, 2.44] | 3.53 [3.51, 3.57] | 158.79 | 0.0886 |
| E3 | linear | morph_attn (A3) | 1 | 2.35 | 3.58 | 167.79 | 0.0937 |
| E4 | linear | ardd_resid (A2) | 1 | 2.33 | 3.59 | 158.19 | 0.0857 |
| E5 | linear | Both | 3 | 2.26 [2.23, 2.28] | 3.45 [3.45, 3.89] | 152.51 | 0.0865 |

How to read:

- **The mean channel improves significantly.** Untreated 6.79 → 1.72~2.44, a reduction of 64~75%. The model actually removes the noise floor raised by CIN. This is the only substantive achievement confirmed in this track.
- **The max channel is indistinguishable from untreated for any candidate.** The intervals for all candidates overlap with untreated (3.45). This is because `maximum` synthesis barely pollutes the max channel in the first place (19.4).
- **A1 (linear β) is actually worse — the only difference exceeding seed variation.**
  The E1 interval [1.72, 1.97] and the E2 interval [2.19, 2.44] do not overlap.
  ε prediction val loss is lower for linear (0.0044 vs 0.0082), but denoising metrics are better for cosine.
  **If the schedule is different, val losses cannot be compared** (loss weighting changes).
  Candidate selection must be made using the metrics of 8/19.4.
- **A2 and A3 show "No difference."** The E5 interval [2.23, 2.28] falls within the E2 interval [2.19, 2.44].
  The single-run E3 and E4 are also in the same range as E2. The two unique contributions of the paper failed to produce a measurable effect on this data and representation.
- **All unpaired metrics worsened.** Relative to actual Field PD, Fréchet went from 107.5 → 138~168, and MMD from 0.0586 → 0.076~0.094. This is consistent with the conclusion of 19.5 — a model trained to erase synthetic CIN fails to move actual Field PD toward Lab PD and instead pushes it away.

> An instance where seed repetition actually flipped a judgment: In a single run (seed 42), the max improvement margin of E5 was **+0.004, the only positive number**, which almost led to the reading that "Only E5 improved the max channel." In seeds 43/44, it yielded −0.001 and **−0.439**, revealing this to be seed noise. This is an instance that perfectly illustrates why 18.1 demands 3 or more seeds.

**Result of applying 18.6 Adoption Rules: No candidates adopted.**

| Component | Judgment | Rationale |
|---|---|---|
| A1 `linear β` | **Rejected** | mean channel deteriorated beyond seed variation. Maintain cosine |
| A2 `ardd_resid` | **No difference** | Within seed variation range compared to E2. Unable to measure contribution |
| A3 `morph_attn` | **No difference** | Same as above |
| A4 `cin_noise` | Adopt (Noise Model) | Only means of synthesis in a dataset without noise groups. However, has limitations of 19.5 |
| A5 1D Backbone | Adopt (Track Premise) | The definition of this track itself |

No candidate satisfied "improvement in **both** paired and unpaired compared to `identity`."
The parameter cost is 0 for both A2 and A3 (parameter-free, asserted in `tests/test_model1d.py`).
There are no failed or unexecuted items — all 11 runs were completed.

### 19.7 Robustness Protocol (E1, seed 42) — **Invalidated 2026-08-25**

> Since the figures came from the same run as 19.6, they are equally invalid. The protocols (SNR sweep + continuous missing injection) are reused as is, and only the values are remeasured.

Only the **protocol** was taken from the paper (metrics are ours from 19.4). Since CIN is parametric, SNR can be precisely targeted. In sweeps, random gain and noise dropout are turned off to preserve the meaning of the x-axis.

```text
SNR Sweep (+3 → −12 dB)          MAE, Model / Untreated
  SNR    mean Channel        max Channel
  +3    3.46 / 7.74      2.96 / 2.28   ← Model is worse on max
   0    5.38 / 11.24     4.74 / 4.57   ← Still worse
  −3    8.66 / 16.31    10.60 / 11.51  ← Reverses here
  −6   14.61 / 23.56    18.34 / 20.04
  −9   22.08 / 33.41    30.34 / 31.89
 −12   30.41 / 45.94    44.54 / 46.35

Continuous Missing Injection (Time Axis)
 5%    1.77 / 7.18      3.31 / 3.86
10%    1.81 / 6.97      3.69 / 4.26
15%    1.92 / 6.82      4.13 / 4.70
```

How to read:
- **The model is better on the mean channel across the entire range.** The improvement ratio decreases as noise increases, but it does not reverse.
- **The max channel has a crossover point around −3 dB.** In the range where the input is already clean (+3, 0 dB), the model actually **adds** error. In other words, the identity mapping is not well preserved in the max channel.
  This is the same phenomenon as the max improvement margin being negative in 19.6, leaving room to train "leave clean input as is" more strongly by raising `noise.dropout_probability`.
- **Both channels are robust to missing data.** However, since the missing target (clean) is the original without missing data, this is closer to "the missing data did not significantly disrupt profile statistics" rather than "it restored the missing section."

### 19.8 Remaining Work

**0. Re-execution after axis fix — Before all other items.** Since all figures in 19.5~19.7 are invalid, it is not yet known whether the rationales for judgments 1~4 below ("nothing to learn in the max channel", "component effects are not measured", "−3 dB crossover point") are still facts. Order:

```text
cli synth-check (new dataset)  → Rewrite 19.5. First confirm if CIN direction is correct
E1 baseline seed 42/43/44      → Re-establish 19.6 baseline
E2~E5                          → Re-execute ablation
E1 robustness sweep            → Rewrite 19.7
```

**0b. Decide on CIN rectification issue (Warning in 19.3).** `|·|` shifts 1st/3rd/5th order harmonics to 2nd/4th/6th orders.
Before re-executing `synth-check`, decide whether to use keep `|·|` / offset (`field + c`), and leave the chosen one in the `cin` config and run record. The two methods can also be kept as comparison items.
Either way, add a test that verifies the phase spectrum of the `sample_cin_matrix` output (current test only looks at `harmonic_field` before rectification).

Before re-execution, **the priority of the items below itself may change.** Leave them as they are and update them with the re-execution results.

Priority order (subject to review based on re-execution results):

1. **Transition to `phase_row_v1` (length 3600, lossless)** — Directly solves limitations 2/3 of 19.1.
   The current result is that "in the profile representation, there is almost nothing to learn in the max channel, and the effects of the paper's components are not measured." The paper's morphological operations look at the **width and edges of pulses**, but if the time axis is folded into mean/max, that structure itself disappears. This transition is necessary to distinguish whether A2/A3 resulting in "No difference" is an issue with the technique or the representation.
2. **Re-evaluate `mode="additive"`** — `maximum` drives pollution only into the mean channel (19.4), effectively excluding the max channel from evaluation.
3. **Increase `noise.dropout_probability`** — Response to the crossover point issue in 19.7 (adding error to clean input).
4. **Seed repetition for E3/E4** — Currently single runs, so the basis for the "No difference" judgment is weaker than E5.
5. **D1~D4 of 2D baseline (Section 6)** — Independently of this track, still unexecuted.

### 19.9 Execution

```powershell
# 0) Dependencies (RTX 5080 = sm_120, so cu128 or higher)
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.get_device_capability()); print(torch.zeros(1).cuda())"

# 1) Check data, split, and pool + cache (torch not required)
python -m Diffusion.prpd_diffusion.ardd1d.cli prepare --config Diffusion/configs/ardd1d_base.json --cache --verify-manifest

# 2) Pre-synthesis validation — Must be done before training (torch not required)
python -m Diffusion.prpd_diffusion.ardd1d.cli synth-check --config Diffusion/configs/ardd1d_base.json

# 3) Pipeline connection check
python -m Diffusion.prpd_diffusion.ardd1d.cli train --config Diffusion/configs/ardd1d_smoke.json

# 4) ablation (E1~E5). Seed repetition with --train-seed — changing config.seed changes splits and evaluation pairs, breaking comparison between candidates (18.1).
#      python -m Diffusion.prpd_diffusion.ardd1d.cli train --config Diffusion/configs/ardd1d_base.json --train-seed 43
python -m Diffusion.prpd_diffusion.ardd1d.cli train --config Diffusion/configs/ardd1d_base.json
python -m Diffusion.prpd_diffusion.ardd1d.cli train --config Diffusion/configs/ardd1d_e2_linear.json
python -m Diffusion.prpd_diffusion.ardd1d.cli train --config Diffusion/configs/ardd1d_e3_morph.json
python -m Diffusion.prpd_diffusion.ardd1d.cli train --config Diffusion/configs/ardd1d_e4_resid.json
python -m Diffusion.prpd_diffusion.ardd1d.cli train --config Diffusion/configs/ardd1d_ardd.json

# 5) Restoration / Evaluation / Robustness / Comparison
python -m Diffusion.prpd_diffusion.ardd1d.cli sample   --run Results/diffusion_runs/ardd1d_e1_...
python -m Diffusion.prpd_diffusion.ardd1d.cli evaluate --run Results/diffusion_runs/ardd1d_e1_...
python -m Diffusion.prpd_diffusion.ardd1d.cli sweep    --run Results/diffusion_runs/ardd1d_e1_... --snr 3,0,-3,-6,-9,-12 --burst-missing 0.05,0.10,0.15
python -m Diffusion.prpd_diffusion.ardd1d.cli compare  --runs Results/diffusion_runs/ardd1d_e*_* --output Results/diffusion_runs/comparison_YYYYMMDD
```

`prepare`, `synth-check`, `evaluate`, and `compare` work without torch.

### 19.10 Files

```text
Diffusion/prpd_diffusion/
├── components/          # Paper components (parameter-free)
│   ├── entropy.py       # Local information entropy H(x,t) — Eq.(7)
│   ├── ardd_resid.py    # a(x,t)=α(1+H/H_max) residual gate — Eq.(5)(6), A2
│   └── morph_attn.py    # Multi-scale morphological gradient attention — Eq.(9)~(15), A3
├── data/
│   ├── profile.py       # profile_v1 representation + Analyzer feature compatibility
│   └── cin_noise.py     # CIN generation, SNR calibration, burst missing — A4
├── models/
│   ├── common.py        # group_norm(), timestep_embedding() — 2D/1D common
│   └── unet1d.py        # Conditional 1D U-Net + component on/off — A5
└── ardd1d/
    ├── config.py  pairs.py  dataset.py  metrics.py  artifacts.py  report.py  cli.py

configs/ardd1d_{smoke,base,e2_linear,e3_morph,e4_resid,ardd}.json
tests/test_{profile,cin_noise,components,model1d,ardd1d_config,ardd1d_pairs}.py
```

`diffusion/` (schedule/gaussian/sampler), `data/splits.py`, `data/pairs.py`, `contract.py`, `training/trainer.py`, `evaluation/metrics.py`, and `runs/artifacts.py` are **reused without modification**.
Since `ddim_sample` only uses `torch.randn(condition.shape)` and channel concat, it works as is on 1D tensors.

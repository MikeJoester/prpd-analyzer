# PatchCore v2: Data Initialization Plan

Plan written 2026-09-18. This plan rebuilds the PatchCore dataset from scratch as `PatchCore/patchcore_data/prpd_v2/`.
The training side is in `PatchCore_training_plan.md`.

## 1. Goal

Build the data for a **PD vs Noise** PatchCore model that uses two memory banks: one built from PD and one from Noise (see the training plan).

| Split | Contents |
|---|---|
| **train** | 100% of Lab PD + 100% of Lab Noise + ~70% of Field PD + ~70% of Field Noise |
| **val** | ~15% of Field PD + ~15% of Field Noise |
| **test** | the remaining ~15% of Field PD + Field Noise |

All Lab data goes into training. Val and test contain Field data only, so every reported number measures **how well the model works in the field**.

## 2. Why the current `patchcore_data/prpd` is being replaced

The current export (`PatchCore/export_to_mvtec.py` → `PRPDAnomalyDataset`) has five defects. Numbers produced from it (`Results/Model_Comparison_Metrics.csv`, PatchCore AUROC 0.65) must not be compared with v2.

| # | Defect | Effect | v2 fix |
|---|---|---|---|
| 1 | Field split is a **random per-file permutation** (`np.random.permutation`) | Files from the same measurement date end up in both train and test. This leaks the date's noise environment and inflates scores. It breaks the project's date-split rule. | Split **by date** (Section 4) |
| 2 | Only `tensor[:, :256]` is exported | Uses 256 of 3,600 cycles (7%). The other 93% of each file is never seen. | Tile the **whole** file into windows (Section 5) |
| 3 | Each image is rescaled by its own max (`crop / crop.max() * 255`) | A faint noise file gets stretched to full brightness. The amplitude difference between PD and Noise, which is the main signal, is erased. | **No per-image rescaling.** Raw uint8 is already 0–255 |
| 4 | Images are named `0000.png`… | Scores can't be traced back to `sample_id`, date or label. | Name by `sample_id` + window index, plus a manifest |
| 5 | Classes come from **group folders** | The change log (260909) relabelled 23 files to `Noise`, but they still sit in the PD groups: 13 in Field PD and 10 in Lab PD. | Class comes from the **`label` column** |

The old folder is kept as `patchcore_data/prpd/` for history and is not deleted. v2 is written to a new folder.

## 3. Source dataset

- **`ai_data_root = artifacts/ai_data_20260911_011704`** is set explicitly, never "latest folder".
  - It is `raw_v2` (correct axes), has the 260909 change log applied, and holds 3,169 valid files.
- Record `ai_data_root` and `raw_data_version` in the v2 `run_config.json` and `dataset_description.txt`.
- **`Synthetic` (51 files) is excluded.** These are artificial, not measured, and one of them is all zeros. Including them would teach the Noise bank a pattern that never occurs in the field.

### 3.1 Class and domain definition

| Column | Rule |
|---|---|
| `domain` | `Lab` if group starts with `Lab`, `Field` if it starts with `Field` |
| `cls` | `Noise` if `label == "Noise"`, otherwise `PD` (Corona / Floating / Particle / Void) |

Resulting pools (measured from `metadata.parquet`, 2026-09-18):

| domain | cls | files | dates | Notes |
|---|---|---:|---:|---|
| Lab | PD | 1,435 | 9 | 288 Corona, 276 Floating, 311 Particle, 560 Void |
| Lab | Noise | 156 | 8 | 146 Lab Noise + 10 relabelled from Lab PD |
| Field | PD | 498 | 56 | 33 Corona, 125 Floating, 3 Particle, 337 Void |
| Field | Noise | 1,029 | 57 | 1,016 Field Noise + 13 relabelled from Field PD |
| **Total** | | **3,118** | | 3,169 − 51 Synthetic |

## 4. Split rules (Field only)

### 4.1 Rules

1. **Split by date, not by file.** A measurement date is assigned to **exactly one** split. This applies to PD and Noise together, since 8 Field dates contain both.
2. **Lab dates are off-limits for val/test.** One Lab date also appears in Field. Since all Lab data is in train, that date's Field files are **forced into train**.
3. **Targets:** 70/15/15 of files, computed **separately for Field PD and Field Noise**. The split is by date, so the ratios will be approximate. Report the achieved ratios.
4. **Coverage:** val and test must each contain at least one Floating date and one Void date among Field PD. Corona should appear in both if possible.
5. **Seed:** `split_seed = 42`. Changing it changes the split. Seed repeats in training (training plan, Section 6) must **not** change it.

### 4.2 Algorithm (deterministic greedy)

```text
1. Pin forced dates to train (Lab-overlap date).
2. Sort remaining Field dates by file count, descending.
   Break ties with a seeded shuffle (split_seed).
3. For each date: assign it to the split whose PD and Noise targets are
   furthest below target. Weight each deficit by how many PD and Noise
   files that date holds, so dates with PD count toward the PD target.
4. After assignment, check the coverage rule (4.1-4). If a split is
   missing a required label, swap in the smallest date that has it,
   taken from the split with the most surplus.
```

### 4.3 Known constraints (from the measured date distribution)

- **Field Noise:** one date holds **509 of 1,029 files (49%)**. It is larger than a whole 15% split, so it lands in train. The remaining 520 files over 56 dates fill the rest of train plus val and test.
- **Field PD:** the largest date holds **189 of 498 files (38%)**, so it also goes to train. The top 5 dates hold 309 files (62%).
- **Field PD / Particle:** only 3 files on 2 dates. It cannot appear in all three splits. Report it and don't interpret Particle results.
- **Field PD / Corona:** 33 files on 4 dates. Expect only 1 date per val/test split, which is too thin to support per-label claims.
- Expected val/test size is roughly **75 PD + 155 Noise files each**. Always report the actual counts.

**Actual result of the build (2026-09-18), for reference:**

| Split | Lab PD | Lab Noise | Field PD | Field Noise | Files | Dates |
|---|---:|---:|---:|---:|---:|---:|
| train | 1,435 | 156 | 348 | 720 | 2,659 | 46 |
| val | – | – | 75 | 155 | 230 | 33 |
| test | – | – | 75 | 154 | 229 | 35 |
| Total | 1,435 | 156 | 498 | 1,029 | 3,118 | 114* |

\* 114 distinct dates in total: 105 Field dates and 10 Lab dates, which share the `unknown_date` bucket. Each date belongs to exactly one split (46 + 33 + 35 = 114).

Achieved Field ratios: PD 69.9/15.1/15.1, Noise 70.0/15.1/15.0. Total images: 87,304 (28 windows per file).

### 4.4 Leakage checks (script must fail, not warn)

- `set(train.dates) ∩ set(val.dates) = ∅`, and the same for every pair of splits.
- Every `sample_id` appears in exactly one split.
- `Lab dates ∩ (val ∪ test).dates = ∅`
- Every Field file is assigned. Lab counts: 1,435 PD and 156 Noise, all in train.

## 5. Image representation

### 5.1 Windowing

Each file is `(128 phase, 3600 cycles)` uint8. The phase axis is always kept whole. Only the time axis is cut.

| Setting | Value | Why |
|---|---|---|
| Window size | **128 × 128** (phase × cycles) | Square, so the backbone can resize to 224×224 **with no crop and no distortion**. The current 128×256 plus `--resize 256 --imagesize 224` center-crop cuts phase bins off the edges. |
| Windows per file | **28** non-overlapping (`3600 = 28×128 + 16`). The last 16 cycles (0.4%) are dropped. | Covers the whole file |
| Pixel values | Raw uint8, unchanged. Grayscale is copied to 3 channels. | Keeps absolute amplitude (fixes defect #3) |
| Phase orientation | Phase on the vertical axis, time on the horizontal. Same for every image. | Consistent with Analyzer plots |

### 5.2 How many windows are exported

- **Val / test:** all 28 windows for every file. At scoring time the file-level score combines window scores (see training plan, Section 4.3).
- **Train:** export all 28 windows too, but the memory-bank builder **subsamples K windows per file** (default K = 4, seeded).
  - This is a memory limit. PatchCore holds every patch feature in RAM before the coreset step. With all windows it would need about 40 GB; with K = 4 it needs about 6 GB. See training plan, Section 3.2.

### 5.3 Folder layout (`PatchCore/patchcore_data/prpd_v2/`)

```text
prpd_v2/
├── train/{pd,noise}/<sample_id>_w00.png … _w27.png
├── val/{pd,noise}/…
├── test/{pd,noise}/…
├── manifest.csv            # one row per image (columns below)
├── files.csv               # one row per file: sample_id, file, group, label, cls, domain, date, split
├── split_dates.json        # date → split, plus forced dates and achieved ratios
├── split_report.txt        # counts per split × domain × cls × label; coverage + leakage check results
├── dataset_description.txt # human-readable summary (Analyzer convention)
├── run_config.json         # ai_data_root, raw_data_version, split_seed, window size, exclusions
└── manifest.json           # file list + sizes + checksums of the csv/json files
```

`manifest.csv` columns:

```text
image_path, sample_id, file, window_idx, t_start, t_end, group, label, cls, domain, date, split
```

## 6. Implementation

- **Script:** `PatchCore/build_prpd_v2.py`. It replaces `export_to_mvtec.py`, which stays as a record of v1.
  - It reads `metadata.parquet` and `raw_tensors_v1/*.npz` through `Diffusion.prpd_diffusion.contract.AiDataset`. It never re-reads the `.dat` files.
  - Arguments: `--ai-data-root` (required, no default), `--out`, `--split-seed 42`, `--window 128`, `--exclude-groups Synthetic`.
  - It **refuses to overwrite** an existing output folder.
- **Where to run:** the export does no training, so it can run locally. Then `rsync` `prpd_v2/` to the GPU server. Alternatively, run it on the server if the `ai_data` folder is already there.
- **Size estimate:** about 3,118 files × 28 windows = 87k PNGs of 128×128. Around 1–2 GB with PNG compression.

## 7. Checklist before training

- [ ] `split_report.txt` shows zero leakage and passes coverage.
- [ ] Achieved Field ratios are recorded (they won't be exactly 70/15/15).
- [ ] Spot-check 10 random images per split × cls against `PRPD_Analyzer` plots of the same `sample_id`.
  - Check orientation, check amplitude is not rescaled, and check the window position matches `t_start`.
- [ ] Record the v2 `run_config.json` hash. The training run will copy it.

# PRPD Anomaly Detection (`prpd_anomaly`) Development Plan

This document is the design, development, and verification plan for the `EfficientAD` package — **a module for detecting PRPD data that does not belong to existing fault types or deviates from the reference distribution**.

It inherits the data rules of the Analyzer (`claude.md`) as they are, and if the rules conflict, `claude.md` takes precedence.
It corresponds to Phase 8 of `claude.md`. It is developed in parallel with the Diffusion track (`DIFFUSION.md`), but
Tier 1 operates independently without diffusion.

Document reference date: 2026-08-24.
**Current implementation status: An empty state with only the `EfficientAD/` folder created. Code not yet started.**
There are no files including `__init__.py`, so it is not an importable package yet.
This document is a design of the contents to be filled in that folder in the future, and only the numbers marked as actual measurements are
values verified by execution, and the rest are design goals.

Things to read before starting — the facts confirmed in the diffusion track affect the premise of this track.

- **The Lab→Field gap in the profile space is not a "noise floor" difference.** Field PD has a lower
  time average and a higher time maximum (rarer but larger pulses) than Lab PD. This direction could not be
  reproduced with noise synthesis (`DIFFUSION.md` 19.5). This is directly related to the premise of the `s_noise`/`s_novel` decomposition of Phase A5.
- **There is no diffusion run available for A5 yet** — see Phase A5 in Section 13.
- The environment is ready: torch 2.11.0+cu128, RTX 5080, tests 128 passed / 0 skipped.

---

## 1. Goal

Learn the normal distribution of the reference group (usually `Lab PD`), and present data deviating from it **along with the review priority**.
It does not automatically change labels or determine faults.

Detection targets (`claude.md` Phase 8):

- Data deviating from the Lab PD distribution
- Newly appearing patterns in Field PD
- Abnormal patterns in Lab Noise and Field Noise
- Corrupted files or abnormal payloads
- Clusters occurring only on specific dates

The success criteria are not "found suspicious-looking files", but the following two.

1. **In P1 leave-one-label-out**, the left-out fault type receives a significantly high score (AUROC ≫ 0.5).
2. **In P2 PD vs Noise**, it almost perfectly separates groups with different properties from the reference group (AUROC ≈ 1).

The output of this model is **"how much does it deviate from the known normal distribution"**, and it is not a physical fault determination.
Ensure it is indicated as such in the UI/reports.

### What this module does NOT do

- Fault type classification (the role of Phase 7 classification model)
- Noise removal (the role of `prpd_diffusion`)
- Automatic category changes — only modifications approved by humans are reflected in the new metadata version
- Reinterpretation of original `.dat` — it only consumes the raw tensor verified and saved by the Analyzer
- Anomaly determination using t-SNE coordinates — t-SNE is for visualization only (`claude.md` rules)

### Distinction from Analyzer's category quality check (Important)

`prpd_analyzer.features.build_category_report` already calculates `suspect_score`, but **the purpose is different.**

| | category quality check (existing) | Anomaly detection (this document) |
|---|---|---|
| Question | "Is the label closer to an adjacent category?" | "Does it not belong to any category?" |
| Criteria | Relative distance between category centers | Normal distribution of the reference group |
| Result | Candidates for mislabeled labels | Candidates for unknown patterns, corruption, or distribution deviation |

Do not merge the two results into one. Connect them only to the extent of showing them side-by-side in the report.

---

## 2. Input Data Contract

### 2.1 Consumed Structure

It uses the **same input** as `prpd_diffusion`. Do not create a new loader.

```text
artifacts/ai_data_YYYYMMDD_HHMMSS/
├── metadata.parquet              # sample_id, group, label, date, raw_tensor_path, ...
├── raw_tensors_v1/{group}.npz    # key "raw", (number of group files, 128, 3600), uint8
├── run_config.json
└── manifest.json
```

`prpd_diffusion.contract.AiDataset` guarantees the following (already implemented/verified).

- Verification of correspondence between metadata rows ↔ npz array indices
- Verification of `sample_id` uniqueness (mixed detection by date/type)
- Verification of matching between the number of metadata rows per group and the number of raw tensors
- Re-verification of `verify_manifest()` checksum, caching of `materialize_memmap()`

### 2.2 Which dataset to use — Independent selection per track

Do not directly scan the original `Data/` folder. **Only use the ai_data execution folder generated and verified by the Analyzer.**

**The anomaly detection track selects the generation dataset to use independently of the diffusion track.**
There is no constraint that the two tracks must point to the same execution. The config of each track specifies its `ai_data_root`,
and records that value in its own run artifacts.

| Folder | root | Number of files | Group | Remarks |
|---|---|---:|---|---|
| `ai_data_20260825_101830` | By type | 3,172 | PD + Noise + Synthetic | **Only valid execution.** `raw_v2`. Based on actual measurements in this document. There is no anomaly detection config yet |
| ~~`ai_data_20260820_172229`~~ | By type | 2,021 | PD + Noise | **Invalid — phase/time axis inversion bug** |
| ~~`ai_data_20260820_170628`~~ | By type | 3,172 | PD + Noise + Synthetic | **Invalid — axis inversion bug** |
| ~~`ai_data_20260821_160405`~~ | PD only | 1,384 | PD only | **Invalid — axis inversion bug** |

**Axis inversion bug on 2026-08-25.** Analyzer's `read_dat` read the `.dat` payload as `reshape(128, 3600)`,
but the actual storage order was `3600 cycle × 128 phase bin` (phase is a continuous axis). Since `3600 % 128 = 16`,
the phase rows were mixed with each other, and the 256-dimensional features on it are all invalid. For details, see
Section 2.1 of `CLAUDE.md` and `STALE_AXIS_BUG.md` in each invalid folder.

**Since this track has no code yet, there are no invalidated artifacts.** However, the figures in Sections 2.3 and 5 below were based on the old
dataset, so they were updated based on the new execution. Since anomaly detection uses Lab/Field Noise in both the reference distribution and
the comparison target (Section 3), **an execution including the noise group** must be selected — the new execution is
the entire amount of 5 groups, so it satisfies this condition.

Selection rules:

- **Within one run, use only one ai_data execution.** Do not combine files from multiple executions
  ("Use only one root", "Prohibition of overlapping aggregation" rules in `claude.md`).
- If `ai_data_root` is left empty, `latest_run_folder()` selects the latest one by name. Since the target pointed to changes as execution folders increase,
  **always specify it in the config.**
- Record the selected value and the `raw_data_version` of that execution in `run_config.json` and `dataset_description.txt` (Section 11).
- If the dataset is changed, the split table in Section 5 and the composition figures in Section 2.3 also change. Re-perform the checks of a `prepare` nature
  in the changed execution to update the corresponding sections.

**Things that are established only when the datasets of the two tracks are the same** — A match is required only in the following tasks.

- `s_noise` / `s_novel` decomposition (Section 3.2) and A5's `decompose`: Established only in the same `sample_id` set and same date split.
  During execution, `decompose` asserts whether the `ai_data_root` and `raw_data_version` of the anomaly detection run and diffusion run are the same, and if they are different, refuses to execute.
- When placing the score/restoration results of the two tracks side-by-side in the same table or figure.

Other anomaly detection independent tasks (A1~A4's fit, score, evaluate, report) are executed regardless of what dataset the diffusion track
uses.

If a wider file range is needed and it is not in the existing execution, do not randomly mix folders, but
specify conditions in the Analyzer to **create a new ai_data execution and then designate that path in the config**.

### 2.3 Current default `ai_data_20260825_101830` (3,172) composition — 2026-08-25 actual measurement

The figures below are the values when this execution is selected. If another generated dataset is selected, re-calculate and update this table and the split table in Section 5 together.

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

The reference distribution will be `Lab PD` with 1,445 files (4 labels). It is a sufficient scale to handle 256-dimensional features,
but dividing it by label yields 286~562, so **shrinkage is applied to the Mahalanobis covariance** (Section 14 Risk Factors).
The fact that the samples per label for 256 dimensions are still at the level of 1-2 times the dimension remains unchanged.

This generated dataset includes 51 `Synthetic` files. However, 1 of them is 0 for the entire interval,
so a quality check is needed before using it as-is for the P4 known-bad protocol. The protocol uses actual files and
**artificial injection cases** (constant matrix, all 0s, 10x amplitude, phase inversion) together, but records which one it is.

---

## 3. Problem Definition: What to consider as an "Anomaly"

### 3.1 Always specify the reference group

"Anomaly" is not an absolute property, but depends on **which distribution is considered normal**.
Therefore, record the `reference_group` with all scores, and do not create scores without a reference.

| Target to Judge | Default Reference Group | Interpretation |
|---|---|---|
| Field PD | Lab PD | Field patterns deviating from the lab standard |
| Lab PD | Lab PD (Self-distribution, LOO) | Outliers/corruption within the lab |
| Field Noise | Field Noise | Abnormal noise among field noise |
| Lab Noise | Lab Noise | Abnormal noise among lab noise |

If the anomaly of `Field Noise` is measured based on `Lab PD`, all of it will come out as an anomaly. It's a meaningless result.

### 3.2 Separation of noise amount and novelty (diffusion link)

The biggest trap in the current data is that **"there is a lot of noise" and "the pattern is new" result in the same score**.
Field PD is mostly flagged as anomalous due to noise. Using it together with the diffusion reconstruction can separate them.

```text
s_noise = mean|y - x̂|              (Amount of noise removed by diffusion)
s_novel = detector(feature(x̂))     (Based on the reconstruction, novelty compared to Lab PD distribution)
```

| | s_novel low | s_novel high |
|---|---|---|
| **s_noise low** | Normal | **Truly new pattern** ← Highest priority for review |
| **s_noise high** | Normal data with heavy noise | Heavy noise + abnormal pattern (Weak basis) |

This decomposition will be added in **Phase A5**. Until then (A1~A4), scores are given using only the original before reconstruction,
and it will be specified that a high Field PD score "may be due to noise".

The diffusion run used for decomposition **must be trained and reconstructed from the same generated dataset as the anomaly detection run.**
If the two tracks have selected different datasets, A5 is not executed, and a diffusion run using the same dataset is designated, or a new diffusion run is created with that dataset before proceeding (Section 2.2).

---

## 4. Representation and detector layers

| tier | detector | Input | torch | Remarks |
|---|---|---|---|---|
| T1-A | Mahalanobis / center distance | 256 features | Not needed | Easy to interpret, contributing phase bins are provided as `reason_features` |
| T1-B | IsolationForest | 256 features | Not needed | Check installation of sklearn 1.9.0 |
| T1-C | LOF (`novelty=True`) | 256 features | Not needed | Strong for local density anomalies |
| T1-D | One-Class SVM (RBF) | 256 features | Not needed | Sensitive/slow when samples are few, low priority |
| T2-A | diffusion eps prediction error | raw tensor | Needed | **No retraining needed**, deterministic with fixed t and fixed seed |
| T2-B | Reconstruction residual statistics (`s_noise`) | raw tensor | Needed | Calculated directly from D4 output |
| T2-C | raw autoencoder | raw tensor | Needed | Separately trained, lowest priority |

This table is the **minimum set to be implemented first**. Candidate expansions based on papers and comparative experiment protocols are in Section 17.

### 4.1 T1 — 256 feature-based

The feature definition uses the Analyzer's `make_features` as-is (definitions must be the same for comparison to be valid).

```text
[mean_phase_001 ... mean_phase_128, max_phase_001 ... max_phase_128]   → (N, 256)
```

Standardization uses the mean/standard deviation calculated **only on the reference set (train dates)**
(Same convention as `prpd_diffusion.evaluation.metrics.standardize`).

Compliance with `claude.md` rules: The 256 features are used only for statistical/non-deep learning detectors,
and are not used as input for raw-data-based anomaly detection (T2).

### 4.2 T2-A — diffusion eps prediction error (Core)

The trained conditional diffusion model is used as an anomaly score without retraining.
Full DDIM sampling (15 windows × 50 steps per file) is expensive, so **only the eps error is measured without sampling.**

```text
for t in {100, 300, 500, 700}:            # Fixed grid
    x_t   = q_sample(x0, t, eps)          # eps is a fixed seed generator
    eps_hat = model(cat[x_t, y], t)
    e_t   = mean((eps_hat - eps)^2)
score = weighted_mean(e_t)                 # Calculated per window, then aggregated per file
```

- Condition `y` is the target file itself, and `x0` is also the target file (proxy for conditional reconstruction error).
- Because of the fixed seed and fixed t grid, **the same input always produces the same score**.
- When aggregating window-level scores to the file level, record both the mean and the upper percentile (e.g., 95%).
  Files that are anomalous only locally are hidden by the mean.

---

## 5. Split, fit, threshold rules

- The date-level split is inherited as-is. `prpd_diffusion.data.splits` is reused, and
  **detector fit is performed only on train dates**.
- The **threshold is set as the in-distribution score percentile of the val dates** (default 99%) and recorded in `metrics.json`.
  Arbitrary fixed values are not used (`claude.md` Phase 8).
- The test dates are used only for the final report. Using test dates for threshold adjustment overestimates the results.
- When combining multiple detectors, use the **rank (percentile) average** instead of a weighted sum of raw scores.
  Since the score scales vary by detector, arbitrary weights eliminate the basis.
- Just like the noise pool, assert that the dates of the reference set do not overlap with the dates of the target being judged.

### Split results based on `ai_data_20260825_101830` (seed=42) — 2026-08-25 actual measurement

If it is a diffusion run using the same generated dataset and the same seed, the split is identical, so the two results can be
directly matched. If the datasets differ, the date sets differ, and the correspondence does not hold.

```text
Dates: train 36 / val 45 / test 33        (Total 114 days, 0 leakages)
```

It is normal for the ratio of the number of dates to the number of samples to be disproportionate. The greedy assignment looks at the
(group, label) cell coverage and sample count, not the number of dates.

| Group | Role | fit (train) | threshold (val) | report (test) |
|---|---|---:|---:|---:|
| Lab PD | Reference distribution | **982** | **256** | 207 |
| Field PD | Target to Judge | 356 | 77 | 80 |
| Field Noise | Target to Judge | 712 | 152 | 153 |
| Lab Noise | Target to Judge | 108 | 10 | 28 |
| Synthetic | Reference | 51 | 0 | 0 |

- 982 reference distribution fit samples vs 256-dimensional features → Covariance estimation may still be unstable.
  Shrinkage is mandatory. (Increased from 787 in the old dataset, but the margin compared to the dimensions is not large.)
- `Lab Noise` has only 10 val samples, so its **self-reference threshold** is unstable. The threshold for this group
  should be determined together with other groups, or the lack of samples should be noted in the report.
- All 51 `Synthetic` files are concentrated on train dates (1 day), so they cannot be used for val/test evaluation.
  To use this group in the P4 known-bad protocol, handle it separately outside of the date split.
- This table depends on the selected dataset and seed. If the dataset or seed is changed, re-run checks of a `prepare` nature
  to update this section.

---

## 6. Verification Protocol — How to evaluate without labels
Since there are no anomaly labels, the procedure to determine "whether it works well" must be confirmed **before detector implementation**.
Without this, only scores will be produced, and reliability cannot be known.

| Protocol | Method | Judgment Criteria |
|---|---|---|
| **P1** leave-one-label-out | Exclude one of the 4 labels of Lab PD and fit → Score the excluded label as "unknown fault". Repeat 4 times | AUROC ≫ 0.5. Indicated by label |
| **P2** PD vs Noise | Fit with Lab PD → Score Lab/Field Noise | AUROC ≈ 1. If it fails, the detector itself is the problem |
| **P3** Domain shift | Fit with Lab PD → Score Field PD | A high score is normal. **Not a fault determination**. Recalculate and separate with the reconstruction in A5 |
| **P4** known-bad injection | Artificially inject constant matrix, all 0s, 10x amplitude, phase inversion | All must be caught at the top |
| **P5** Date drift | Trend of median scores by date | Check if only a specific date spikes. Candidate for measurement condition changes |

The AUROC of P1 and P2 is used as the **detector selection criteria** and recorded in the report.
P3 is not a performance metric, but an analysis target. Specify this distinction in the report.

---

## 7. Output Contract

Follow the fields of `claude.md` Phase 8, but add diffusion decomposition items.

```text
sample_id
file_path
group
label
date
detector                     # mahalanobis | isolation_forest | lof | ocsvm | diffusion_eps | ensemble
detector_version
reference_group              # Distribution considered normal
reference_dates              # List of dates used for fit
anomaly_score
score_percentile             # Percentile against the reference set
threshold
is_anomaly
s_noise                      # Filled only when linked with diffusion (null before A5)
s_novel                      # Novelty based on reconstruction (null before A5)
nearest_samples              # List of nearest sample_ids in the reference set
reason_features              # Phase bins with large contributions and directions
data_quality_status
review_status                # Unreviewed | Confirmed | Rejected | Pending
reviewer
review_note
run_id
ai_data_root                 # Generated dataset path selected by this run
raw_data_version             # Raw data version of that execution
diffusion_run                # Diffusion run referenced in A5 (null if absent)
feature_version
model_version
seed
```

Artifacts are saved with the same convention as diffusion.

```text
artifacts/anomaly_runs/anomaly_YYYYMMDD_HHMMSS/
├── run_config.json           # Full config + config_hash
├── split_config.json         # List of dates used for fit/threshold/evaluation
├── dataset_description.txt   # Human-readable description (reference group, target, threshold basis)
├── detectors/                # Trained detector objects + scaler
├── scores.parquet  scores.csv
├── protocols.json            # Results of P1~P5 (AUROC, etc.)
├── metrics.json
├── anomaly_report.html
└── manifest.json
```

---

## 8. Execution (Design)

```powershell
# 0) Dependencies — Analyzer's default stack is sufficient (T1 does not need torch)
python -m pip install -r requirements.txt

# 1) Fit reference distribution + determine threshold
#    The generated dataset to use is specified by data.ai_data_root in the config.
#    To use a different execution, overwrite with --ai-data-root (independent of diffusion track settings).
python -m src.prpd_anomaly.cli fit --config Diffusion/configs/anomaly_base.json
python -m src.prpd_anomaly.cli fit --config Diffusion/configs/anomaly_base.json `
    --ai-data-root artifacts/ai_data_20260825_101830

# 2) Score target to judge
python -m src.prpd_anomaly.cli score --run artifacts/anomaly_runs/anomaly_YYYYMMDD_HHMMSS

# 3) Execute verification protocols P1~P5
python -m src.prpd_anomaly.cli evaluate --run artifacts/anomaly_runs/anomaly_YYYYMMDD_HHMMSS

# 4) (A5) Decompose into s_noise / s_novel using diffusion reconstruction
#    If the ai_data_root / raw_data_version of the two runs differ, it fails immediately.
python -m src.prpd_anomaly.cli decompose `
    --run artifacts/anomaly_runs/anomaly_YYYYMMDD_HHMMSS `
    --diffusion-run Results/diffusion_runs/denoise_YYYYMMDD_HHMMSS

# Test
python Diffusion/tests/run_tests.py
```

`fit`, `score`, and `evaluate` operate without torch, and are executed regardless of which generated dataset the diffusion track uses.
Only `decompose` requires torch and a trained diffusion run that **used the same generated dataset**.

---

## 9. Configuration Reference (`configs/anomaly_base.json`, Design)

| section | key | Default | Description |
|---|---|---|---|
| (root) | `seed` | 42 | seed for fit, injection, aggregation |
| | `run_prefix` | `anomaly` | Result folder prefix |
| data | `ai_data_root` | `artifacts/ai_data_20260825_101830` | **Generated dataset this track will use. Designated independently of diffusion config.** Can be overwritten when running with `--ai-data-root` |
| | `memmap_cache` | `artifacts/repr_cache` | Share cache location with diffusion. `materialize_memmap` separates **per execution** as `{cache}/{ai_data run name}/{group}.npy` (Updated 2026-08-24). Even if the dataset is changed, the cache of previous executions is not reused |
| reference | `group` | `Lab PD` | Reference group considered normal |
| | `labels` | `null` | All fault types of the reference group if null |
| target | `groups` | `["Field PD", "Lab Noise", "Field Noise"]` | Target to judge |
| split | `ratios` | 0.7 / 0.15 / 0.15 | By date unit, same convention as diffusion |
| detector | `kinds` | `["mahalanobis", "isolation_forest", "lof"]` | Combine by rank average after concurrent execution |
| | `ensemble` | `rank_mean` | Weighted sum of raw scores is prohibited |
| threshold | `method` | `val_percentile` | Based on in-distribution val dates |
| | `percentile` | 99.0 | Recording required |
| protocol | `run` | `["P1", "P2", "P3", "P4", "P5"]` | Select verification protocols |
| report | `top_k` | 50 | Number of top candidates to put in the report |

`config_hash()` is created in the same way as diffusion (first 12 characters of sha256 of the entire config).
Unknown keys immediately throw an exception upon loading.

---

## 10. File Structure (Design)

```text
202608_PRPD/
├── ANOMALY.md                       # This document
├── DIFFUSION.md                     # Noise removal track
├── claude.md                        # Analyzer plan (upper rules)
├── configs/
│   └── anomaly_base.json            # Base config — not created
├── src/
│   ├── prpd_analyzer/               # Read-only reuse (feature definitions)
│   ├── prpd_diffusion/              # Data layer reuse + supplies T2 scores
│   └── prpd_anomaly/                # Folder created (empty). All files below are not created
│       ├── __init__.py              # Package declaration
│       ├── reference.py             # Defines reference group, selects fit target (date unit)
│       ├── scores.py                # detector common fit/score interface, T1-A~D
│       ├── detectors/               # Comparative candidates from Section 17 (classical.py / deep.py)
│       ├── ref/                     # Original paper PDFs (Section 17 registry) — empty
│       ├── compare.py               # Candidate comparison runner + comparison report
│       ├── diffusion_score.py       # T2-A/B — import only when torch is available
│       ├── threshold.py             # Percentile threshold, rank-based combination
│       ├── protocols.py             # P1~P5
│       ├── report.py                # HTML report + CSV/Parquet
│       ├── runs/config.py           # AnomalyConfig, config_hash
│       └── cli.py                   # fit / score / evaluate / decompose
├── tests/                           # 3 files below not created
│   ├── test_anomaly_scores.py
│   ├── test_anomaly_threshold.py
│   └── test_anomaly_protocols.py
└── artifacts/
    ├── ai_data_YYYYMMDD_HHMMSS/     # Input candidates (Do not modify). Select one with config
    ├── diffusion_runs/              # Referenced in A5
    └── anomaly_runs/
        ├── anomaly_YYYYMMDD_HHMMSS/
        └── comparison_YYYYMMDD_HHMMSS/   # Section 17 candidate comparison result
```

### Targets for Reuse (Do not write new ones)

| Reuse | Source | Status |
|---|---|---|
| `AiDataset` loader / memmap cache | `prpd_diffusion.contract` | Implementation/verification complete |
| Split by date / leakage inspection | `prpd_diffusion.data.splits` | Implementation/verification complete |
| 256 feature definition | `prpd_analyzer.features.make_features` | Implementation complete |
| Standardization, distance, profile | `prpd_diffusion.evaluation.metrics` | Implementation/verification complete |
| run folder, description, manifest | `prpd_diffusion.runs.artifacts` | Implementation complete |
| Group/label constants | `prpd_analyzer.metadata.GROUPS`, `LABELS` | Implementation complete |

When the common module consumers become 3 (when a classification model is added), promote them to `src/prpd_core/`.
Moving them now would entail touching the 46 tests of the diffusion track under development, so the cost outweighs the benefit.

---

## 11. Reproducibility Rules

- All executions are saved in the `anomaly_YYYYMMDD_HHMMSS` folder, and existing executions are not overwritten.
- Leave `run_config.json`, `split_config.json`, `dataset_description.txt`, and `manifest.json`.
- **Must record the threshold and its calculation basis** (method, percentile, reference set size, reference date list).
- Save the detector object and scaler so that the same score can be recalculated.
- Record the seed, feature version, diffusion model version (A5), and library versions in `dataset_description.txt`.
- **Record the generated dataset path (`ai_data_root`) used and the `raw_data_version` of that execution**.
  Selection is free per track, but recording is mandatory. If there is no record, it cannot be known later which data the result is from.
- When comparing side-by-side with diffusion results or performing A5 decomposition, first confirm whether the `ai_data_root` and
  `raw_data_version` of the two runs are the same. If different, do not compare, and note the fact that they are different in the report.
- Use only one root out of by-date/by-type, and do not combine the two results.
- Do not write guesses about physical causes in the report sentences. Present only scores and evidence.

---

## 12. Test Plan

| Test | Verification Content |
|---|---|
| `test_anomaly_scores.py` | Normal samples below the threshold, artificial anomalies (constant, all 0s, 10x amplitude, phase inversion) at the top |
| | Scores are exactly identical given the same seed and same data (Determinism) |
| | 3 types of detectors satisfy the same fit/score interface |
| `test_anomaly_threshold.py` | Percentile threshold is reproducible regardless of reference set size |
| | Rank average combination is independent of detector order |
| | Test dates are not mixed in threshold calculation (leakage assertion) |
| `test_anomaly_protocols.py` | Label exclusion in P1 is actually reflected in the fit set |
| | AUROC of synthetic data artificially separable in P2 is 1.0 |
| | 100% of P4 injected cases are detected |
| Common | Use only train dates for fit, no intersection between reference/target dates |
| | Selected `ai_data_root` and `raw_data_version` are recorded in `run_config.json` |
| | A single run does not refer to more than one ai_data execution |
| | `decompose` throws an exception when the diffusion run and dataset mismatch |

Use the existing runner as-is: `python Diffusion/tests/run_tests.py` (As of 2026-08-24, **128 passed / 0 failed /
0 skipped**, anomaly detection tests are still 0). torch is installed (2.11.0+cu128, RTX 5080).

---

## 13. Step-by-Step Development Sequence

Dependencies on the D track (`DIFFUSION.md`) are indicated together.

### Phase A0: Package Skeleton — In progress (only folders created) · **Can start now**

1. [Completed] Create `EfficientAD/` folder — currently empty
2. Add `__init__.py` and `runs/config.py` — `AnomalyConfig` + `config_hash`
3. Write `configs/anomaly_base.json` — Specify the generated dataset to use in `ai_data_root`.
   The value is chosen independently by this track, and does not need to be the same as the diffusion settings.
4. Connect CLI skeleton (`fit`/`score`/`evaluate`) and run folder creation, `--ai-data-root` overwrite option

Completion Criteria: `fit` loads the reference set and creates an empty run folder and `run_config.json`.
The absolute path of the selected `ai_data_root` and the `raw_data_version` of that execution are recorded in `run_config.json`.

### Phase A1: T1 detector — Not started (Precursor: A0)

1. `reference.py` — Select reference group/date, assert leakage
2. `scores.py` — T1-A(Mahalanobis), T1-B(IsolationForest), T1-C(LOF) common interface
3. `threshold.py` — val percentile threshold, rank average combination
4. Output `scores.parquet` / `scores.csv`

Completion Criteria: 3 detectors score all targets, and the threshold and basis are recorded.
`reason_features` (contributing phase bins) are filled in T1-A.

### Phase A2: Verification Protocol — Not started (Precursor: A1) · **Most important**

Execute P1~P5 and determine the detector. If this step is skipped, the reliability of all subsequent scores cannot be known.

Completion Criteria:

- The 4 AUROCs per label for P1, and the AUROC for P2 remain in `protocols.json`.
- Only detectors satisfying P2 AUROC ≥ 0.95 are included in the default configuration.
- All P4 injected cases are detected.
- Detectors that do not meet the criteria are excluded from the default configuration along with their basis.

### Phase A2b: Paper-based Candidate Comparison — Not started (Precursor: A2, Receipt of paper list)

Expand and compare candidate detector and representation combinations according to the convention in Section 17. The protocol of A2 is used as-is for the scoring criteria,
so no new evaluation code is created.

1. Clean up `detectors/` interfaces, migrate existing 3 T1 types to that interface
2. Clean up paper list (Section 17.5 format) → Determine candidates to port
3. `compare.py` — Run candidates × representations × seeds, create `comparison.parquet` and report

Completion Criteria: P1/P2 AUROC, cost, and determinism results by candidate remain in a single table, and the basis for adoption/exclusion is recorded.
Eliminated candidates and reasons are also left in the table.

### Phase A3: Report and Review Management — Not started (Precursor: A2)

1. `anomaly_report.html` — List of top candidates, group × category × date distribution, PRPD comparison figure
2. Display PRPD images and mean/max profiles per candidate side-by-side with reference normal files
3. Record/update path for `review_status` / `reviewer` / `review_note`
4. Add Streamlit tab (Optional)

Completion Criteria: Human review status can be managed without automatic category changes (`claude.md` completion criteria).

### Phase A4: Expansion by Group — Not started (Precursor: A2)

Separate Lab/Field, PD/Noise, and fit/execute reference distributions respectively (table in Section 3.1).

Completion Criteria: Execution results per reference group remain in their respective run folders, and the reference is specified in the report.

### Phase A5: Diffusion Linkage — Not started (Precursor: A2 **AND** D4 main training complete)

**Additional precondition: The diffusion run to be referenced must use the same generated dataset as this anomaly detection run.**
If the two tracks have selected different datasets, align one side before A5 (Section 2.2).

Status as of 2026-08-25 — **There is no diffusion run available for A5 yet.**

- In the **2D baseline** of Section 6 in `DIFFUSION.md`, **D4 main training is still unexecuted**.
- In the **ARDD 1D track** of Section 19 in `DIFFUSION.md`, **training was completed, but the results were all invalidated due to an axis inversion bug** (Section 2.2),
  and even if re-run, it cannot be used for A5. The output of that track is a phase profile
  `(2, 128)` and is irreversible, so it cannot be returned to `128×3600` raw. T2-A (eps error) and
  recalculation of `s_novel` based on the reconstruction predicate a raw reconstruction, so it does not hold.

One premise of the A5 design **has returned to an unconfirmed state.** D1 verification of ARDD 1D concluded that "in the profile space,
the Lab→Field gap is not a difference in noise floor, but a difference in pulse density/size"
(`DIFFUSION.md` 19.5), but those figures are invalid due to the axis inversion bug. The `s_noise`/`s_novel` decomposition
premises that "the noise contribution among the high scores of Field PD can be removed by reconstruction",
**but whether that premise holds is currently unknown.** Confirm with the D1 result of the new dataset before starting A5.
Do not assume either way.

1. `diffusion_score.py` — T2-A(eps error), T2-B(`s_noise`)
2. Recalculate `s_novel` based on the reconstruction, interpret with quadrants in Section 3.2
3. Re-run P3 — Compare Field PD score changes before/after reconstruction

Completion Criteria:

- `decompose` asserts that the `ai_data_root` and `raw_data_version` of the two runs match, and refuses execution if they do not match.
- Whether the high score of Field PD is due to noise or pattern is presented separately.
- T2-A scores are perfectly reproduced on the same input.

---

## 14. Risk Factors and Open Questions

| Item | Content | Response |
|---|---|---|
| Confusion between Noise and Anomaly | Field PD is flagged as anomaly entirely due to noise | `s_noise`/`s_novel` decomposition in A5. Specify in the report until then |
| Lack of reference group samples | 787 fit samples vs 256-dimensional features, 184~376 per label | Apply shrinkage to Mahalanobis covariance, run LOF/IF in parallel |
| Absence of evaluation labels | No "ground truth anomaly" | Judge only with P1~P4 proxy protocols. Specify limitations in the report |
| Scope of generated dataset | The default dataset is a subset created with the conditions at the time of Analyzer execution | Do not mix execution folders within a single run. To expand the scope, designate another execution as `ai_data_root` or create a new execution in the Analyzer |
| Dataset mismatch between tracks | Since the two tracks choose datasets independently, A5 decomposition and side-by-side comparison may diverge silently | Assert `ai_data_root` / `raw_data_version` match in `decompose` and comparison reports. Refuse execution upon mismatch |
| Document mismatch upon dataset change | Figures in Sections 2.3 and 5 depend on specific executions | When changing datasets, re-run composition/split checks to update the corresponding sections |
| memmap cache contamination | The shared `repr_cache` does not distinguish ai_data executions, so matrices from other datasets can be reused | Use execution-specific cache subfolders. Until then, empty and recreate the cache when changing datasets |
| Field PD Particle | Only 3 files (both in default dataset and original folder) | Note that the results for this combination are statistically meaningless |
| Lab Noise test samples | 8 test files | Specify the confidence interval of the self-reference judgment result together |
| Threshold interpretation | 99 percentile does not mean "1% are anomalies" | Specify in the UI that it is a review priority cutoff |
| Limitations of 256 features | Time axis compressed to mean/max → Temporal anomalies lost | Supplement with T2-A (raw-based). Add time window-level scores if necessary |
| Date bias | Changes in measurement conditions on a specific date appear as anomalies | Verify with P5 and display date factors separately in the report |
| Misunderstanding of automatic judgment | Top scores = read as faults | Specify with a fixed phrase in UI/reports that it means "unknown fault type" |

---

## 15. Relationship with Analyzer / Diffusion / Roadmap

```text
Analyzer (claude.md)
  → Multiple ai_data execution folders (Verified 128×3600 raw tensor + metadata)
      ← The two tracks each select one execution to use
      ├→ prpd_diffusion (Noise removal)           ← DIFFUSION.md
      │       └→ Supply reconstruction and eps error (A5)
      └→ prpd_anomaly (Anomaly detection)                ← This document
              → Phase 9 Natural Language Agent (Anomaly score inquiry tool)
              → Phase 10 Paper results (Anomaly detection case figures)
```

Compliance requirements:

- 256 features are for statistical/non-deep learning detectors, and are not input for raw-data-based anomaly detection.
- t-SNE is for visualization only and is not used as a criterion for determining anomalies.
- By-date and by-type are different views of the same data, so do not sum or mix them.
- Select the generated dataset to use per track, but use only one within a single run and record the selected value.
  Require the same dataset only when combining/comparing the results of the two tracks.
- Do not change categories automatically. Reflect only approved modifications in the new metadata version.
- The anomaly detection result means "unknown fault type", not a physical fault determination.

---

## 16. What has not been made yet

- All internal files of `EfficientAD/` — Folder created, but not even `__init__.py` exists (A0~A5 not started)
- `configs/anomaly_base.json`
- 3 `tests/test_anomaly_*.py` files
- Streamlit anomaly detection tab
- T1-D(One-Class SVM), T2-C(raw autoencoder)
- Time window-level anomaly scores (only file-level aggregation is designed)
- Review history version management — currently only `review_status` column design exists
- Full algorithm comparative experiment — Section 17 only has protocols and candidate list, comparative runner not implemented

---

## 17. Candidate Detectors and Comparative Experiments (Paper-based)

T1/T2 in Section 4 is the minimum set to build first, and it is not yet known which detector fits this data.
Compare various families with **the same dataset, same split, same protocols (P1~P5 in Section 6)** and leave evidence.

**Papers are provided by the user (As of 2026-08-21, no anomaly detection papers received).** In the same way as the diffusion track,
place the original PDFs in `EfficientAD/ref/` and create a registry row above the table in 17.2
(`DIFFUSION.md` 18.2 format). Until the list is received, the comparison protocol in 17.1 and the interface in 17.3 are confirmed.
Do not cite figures from unread papers, and figures from papers with evaluation methods different from ours (e.g., AUROC on benchmarks with labels)
are used only as design references, not improvement goals.

### 17.1 Fair Comparison Protocol — Fixed before detectors

Since there are no anomaly labels, the comparative protocol is the entirety of reliability.

- **Same generated dataset**: All candidates use the same `ai_data_root` (Section 2.2).
- **Same split and same reference group**: Fix the `reference_group` and fit dates. If the reference is different, score comparison
  does not hold (Section 3.1).
- **fit only on train dates**, threshold is val percentile, test is exclusively for final report (Section 5). No exceptions are made per candidate.
- **Scoring criteria fixed to P1~P5** (Section 6). The per-label AUROC of P1 and the AUROC of P2 are used as ranking criteria,
  and P3 is an analysis target, so it is not included in the ranking.
- **Determinism**: Scores must be exactly identical with the same input and same seed. Candidates that do not reproduce are eliminated.
- **Ignore score scale**: Comparison and combination among candidates are done only with rank (percentile), not raw scores (Section 5).
- **Separate representation axis and detector axis**. Must create a table with combinations of `(256 features | window features | raw tensor)` × `(detector)`
  to distinguish whether performance differences are due to representation or algorithm.
- **Record cost**: fit time, scoring time per file, memory, torch requirement.
- **Record failures**: Note sample shortage, convergence failure, and OOM along with the reasons.
- Save the aggregated results to `artifacts/anomaly_runs/comparison_YYYYMMDD_HHMMSS/` and leave the referenced run_id list,
  `ai_data_root`, reference group, and split file path together.

```text
comparison_YYYYMMDD_HHMMSS/
├── comparison.parquet  comparison.csv   # candidate × representation × protocol × seed
├── runs.json                            # Referenced run_ids, ai_data_root, reference_group
└── comparison_report.html
```

### 17.2 Candidate Detectors (Paper basis to be filled after receiving the list)

| id | Family | Input | torch | Paper Basis | Status |
|---|---|---|---|---|---|
| `mahalanobis` | Distance / Covariance (shrinkage) | 256 features | Not needed | TBD | T1-A, prioritize implementation |
| `knn_dist` | Distance / k-th nearest distance | 256 features | Not needed | TBD | Candidate — simple lower bound |
| `kde_gmm` | Density estimation | 256 features | Not needed | TBD | Candidate |
| `isolation_forest` | Tree splitting | 256 features | Not needed | TBD | T1-B, prioritize implementation |
| `extended_if` | Tree splitting (non-axis aligned) | 256 features | Not needed | TBD | Candidate |
| `lof` | Local density | 256 features | Not needed | TBD | T1-C, prioritize implementation |
| `ocsvm` | Boundary learning (RBF) | 256 features | Not needed | TBD | T1-D, low priority |
| `pca_recon` | Linear reconstruction error | 256 features | Not needed | TBD | Candidate — lower bound for deep AE |
| `deep_svdd` | Hypersphere compression | raw tensor | Needed | TBD | Candidate |
| `autoencoder` | Reconstruction error | raw tensor | Needed | TBD | T2-C, Candidate |
| `vae` | Probabilistic reconstruction / ELBO | raw tensor | Needed | TBD | Candidate |
| `feature_memory` | Embedding memory / Nearest neighbor (PatchCore family) | raw tensor embeddings | Needed | TBD | Candidate |
| `norm_flow` | Normalizing flow density | raw tensor embeddings | Needed | TBD | Candidate |
| `diffusion_eps` | diffusion eps prediction error | raw tensor | Needed | TBD | T2-A, **No retraining needed** |
| `diffusion_recon` | Reconstruction residual (`s_noise`) | raw tensor | Needed | TBD | T2-B, Use D4 artifacts |
| `self_sup` | Self-supervised pretext (masking / order shuffling) | raw tensor | Needed | TBD | Candidate |

Representation axis candidates:

| Representation | Content | Remarks |
|---|---|---|
| `feat256` | 256-dimensional mean/max by phase | Default. Time axis information is compressed |
| `window_feat` | Aggregate after feature per time window | Needed for local anomaly detection (Section 14) |
| `raw` | `128×3600` original | Deep detector only. Per `claude.md` rules, do not mix with 256 features |

Priority: T1 3 types (`mahalanobis`, `isolation_forest`, `lof`) → `knn_dist` / `pca_recon` (verify lower bound)
→ `diffusion_eps` (after D4 completion) → Deep family.

### 17.3 Execution Interface (Design)

```text
EfficientAD/detectors/
├── base.py        # class Detector: fit(X_ref) -> None, score(X) -> np.ndarray
├── classical.py   # mahalanobis, knn, kde/gmm, if, lof, ocsvm, pca_recon
└── deep.py        # deep_svdd, autoencoder, feature_memory, diffusion_* — delayed torch import
```

- Unify the sign of scores so that **larger is more anomalous**. Do not use the `score_samples` sign of sklearn as-is.
- Put the id list in `detector.kinds` of the config, and unknown ids throw an exception upon loading.
- All candidates fill the same `scores.parquet` schema (Section 7). The comparison runner reads only this file.

### 17.4 Adoption Rules

- **Candidates failing to satisfy P2 AUROC ≥ 0.95 are excluded from the default configuration** (Section 13 A2 completion criteria).
- Look at both the average and minimum of the P1 per-label AUROCs. A detector that works well on only one label is not adopted.
- Failing to catch all P4 injection cases results in elimination.
- Combination is done only with rank averages of top candidates, and weighted sums of raw scores are not used.
- If the results differ from the reported performance in papers, record the differences in data scale, dimensions, and reference groups. Do not immediately
  conclude that "the technique is bad" upon a failure to reproduce.
- Leave the basis for final selection (protocol figures, cost, interpretability) in the comparison report. Detectors that cannot provide
  `reason_features` are given lower priority if performances are equal.

### 17.5 Paper Summary Format

```text
paper_id
Title / Source / Year
Target domain            # PD/PRPD, general images, time series, industrial defects, etc.
Algorithm family         # Link to id in 17.2
Input representation     # Corresponds to feat256 / window_feat / raw
Normal data scale and dim # Compare with our reference set (787 files × 256 dimensions)
Evaluation method        # Presence of labels, metrics used
Applicability to our data
Porting cost
Verification status      # Verified original / Verified abstract only / Unverified
```

Figures from papers with a "Verification status" of "Unverified" are not transferred to the main text of the document.

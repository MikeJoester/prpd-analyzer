# PRPD Web Analyzer Development Plan

## 0. Model Training Environment (Required)

**Do not train new models on this local machine.** Every training run (new models, retraining and seed repetitions) happens on the remote GPU server:

1. Connect over SSH using the credentials in `.env` at the repo root (`USR` = `user@host`, `PASSWORD` = login password). Read them from `.env` when needed; **never copy the credentials into code, docs, configs or commit messages.**
2. Activate the `danenv` conda environment on the server (`conda activate danenv`) before running anything.
3. Run the training on the server. It has **2× NVIDIA RTX 4090 (24 GB each)**, so choose batch sizes and multi-GPU settings for that hardware, not for the local GPU.

Lightweight work that is not training (tests, `prepare`/`evaluate` steps without torch, analysis scripts, the Streamlit app) can still run locally.

## 1. Project Goal

Develop a data analysis Analyzer that explores Partial Discharge (PRPD) `.dat` data in a web browser and analyzes the distribution differences according to fault types and measurement dates.

The core analysis targets are the following four groups:

- `Lab PD`: Corona, Floating, Particle, Void data measured in the laboratory
- `Field PD`: Corona, Floating, Particle, Void data measured in the field
- `Lab Noise`: Noise data measured in the laboratory
- `Field Noise`: Noise data measured in the field

Noise is not combined into a single class; `Lab Noise` and `Field Noise` are analyzed separately. This allows identifying the differences in Noise characteristics according to the measurement environment and the possibility of distinguishing PD/Noise.

In the web UI, users should be able to select data groups, fault types, and dates, and check the following results:

- Data quantity and distribution by date
- Quantity and distribution by date of Lab Noise and Field Noise
- Average PRPD pattern
- Average and maximum values by phase
- 256-dimensional Feature-based t-SNE clustering
- Original files of selected data and analysis results

## 2. Confirmed Data Specifications

### 2.1 Binary Format

- File extension: `.dat`
- Header: default 227 bytes, some files 167 bytes
- payload: `uint8`
- payload size: `128 × 3600 = 460,800 bytes`
- 1 file: 1 measurement data for about 1 minute (3600 cycle = 60 Hz × 60 s)

**The payload storage order is cycle unit records. Phase is a continuous axis.**

```text
227-byte or 167-byte header
+ [cycle 0: phase 0..127][cycle 1: phase 0..127] ... [cycle 3599: phase 0..127]
```

Therefore, the parser reshapes the payload to `(3600, 128)` and transposes it to create `(phase=128, time=3600)`. Matrices going out of the module always follow the `(phase, time)` convention.

**2026-08-25 Revision History.** Prior to this, the parser immediately `reshape(128, 3600)` the payload.
Since `3600 % 128 = 16`, values of different phases were shifted by 16 bins and mixed into each "phase row", and the phase average/maximum were not values of the same phase. Evidence confirmed by actual measurement:

- payload autocorrelation **lag 128 = 0.978** vs **lag 3600 = 0.192**
  (lag 128 was dominant in 34 out of 40 samples in Data/. The remaining 6 were Noise files without phase structure)
- Phase average profile of Corona file (128 → 16 bin summary)
  - After fix: `[0, 0, 0, 0.6, 1.9, 0.4, 0, 0, 0, 0, 2.7, 21.9, 37.4, 22.0, 1.9, 0]` — localized lobe
  - Before fix: `[0.1, 0, 16.1, 25.2, 24.0, 18.2, 3.4, 0, 0, 0, 0.9, 0.6, 0, 0, 0.1, 0.2]` — blurred blob

The regression test is in `tests/test_io.py`. It fails immediately if the axis is flipped.
For artifacts created before this bug, refer to Section 10 "Artifacts invalidated by axis flip bug".

### 2.2 Folder Structure

`By Date` and `By Type` are not different datasets. The two folders are two views of the same original files classified by different criteria, and the file contents and analysis targets are identical. Therefore, do not scan the files in the two folders simultaneously or merge them for analysis.

When starting the analysis, the user selects one standard path.

- `By Date`: Used for date-centric exploration
- `By Type`: Used for exploration centered on Lab/Field and PD/Noise groups

The analyzer supports both folder structures, but uses only one structure in a single execution.

```text
Data/
├── by_date/
│   ├── Lab/Date/Fault Type/*.dat
│   ├── Field/Date/Fault Type/*.dat
│   └── Artificially_Generated/*.dat
└── by_type/
    ├── Lab_PD/Fault Type/*.dat
    ├── Field_PD/Fault Type/*.dat
    ├── Lab_Noise/*.dat
    ├── Field_Noise/*.dat
    └── Randomly_Generated/*.dat
```

Metadata mapping is as follows.

| Folder | Analysis Group |
|---|---|
| `Lab`, `Lab_PD` | `Lab PD` |
| `Field`, `Field_PD` | `Field PD` |
| `Lab_Noise` | `Lab Noise` |
| `Field_Noise` | `Field Noise` |
| `Artificially_Generated`, `Randomly_Generated` | `Synthetic` |

Since `Lab_Noise`, `Field_Noise`, and `Randomly_Generated` do not have fault type folders, set all their labels to `Noise`. Therefore, the three groups should also be displayed and filtered in the PRPD/Feature, t-SNE, and Data tabs just like the PD group.

Dates are read from the date folder in the by_date structure, and from the `YYYYMMDD` pattern in the filename in the by_type structure.

The currently confirmed number of files is the quantity based on the same dataset. Do not sum the quantities of `by_date` and `by_type`.

Currently confirmed number of files:

- `Lab PD`: 1,445
- `Field PD`: 513
- `Lab Noise`: 146
- `Field Noise`: 1,017
- `Synthetic`: 51
- Total: 3,172

File duplication prevention rules:

- If the analysis root is `Data/by_date`, read only its sub-files.
- If the analysis root is `Data/by_type`, read only its sub-files.
- Do not input both roots at the same time or merge the results.
- Even if the same file exists in multiple classification folders, count it only once based on the file identifier.

## 3. Feature extraction Design

Do not use PCA as the default Feature extraction. Assuming the original matrix of each file is $X(p,t)$, calculate the time average and maximum value per phase $p$.

$$
mean_p = \frac{1}{3600}\sum_{t=1}^{3600}X(p,t)
$$

$$
max_p = \max_{t=1,\ldots,3600}X(p,t)
$$

The final Feature is a 256-dimensional vector in the following order.

```text
[mean_phase_001 ... mean_phase_128,
 max_phase_001  ... max_phase_128]
```

The Feature matrix after processing $N$ files should be `(N, 256)`.

Processing order:

```text
Read .dat
→ Validate payload
→ reshape(3600, 128) → transpose → (phase=128, time=3600)
→ Calculate phase-axis feature
→ Combine 128 means and 128 maxs
→ Check missing values/constant Features
→ StandardScaler
→ t-SNE 2D
```

Since the maximum value can be sensitive to a single noise, compare `max` and top percentile (e.g., 99.5%) in subsequent experiments. The default product operation uses `max` according to requirements.

## 4. Web UI Configuration

The fast verification stage of the current project is implemented with Streamlit. Separate the existing `app.py` into functional functions to increase maintainability.

### 4.1 Data Settings Screen

- Select data root folder
- Select one standard root between `By Type` or `By Date`
- Display on the screen that the two structures are different classification views of the same data
- Select full data analysis mode and filter application mode
- Display total searched files, target files for analysis, actual analyzed files, and error files
- Select analysis group
  - Lab PD
  - Field PD
  - Lab Noise
  - Field Noise
  - Synthetic
- Select fault type
- Select date range or specific date
- Select `Lab Noise` and `Field Noise` independently when analyzing Noise
- Select AI dataset generation group and specify timestamp result folder

### 4.2 Summary Screen

- Total files, normally analyzed files, error files
- Compare quantity of Lab PD and Field PD
- Compare quantity of Lab Noise and Field Noise
- Quantity by fault type
- Quantity trend by date
- Group × Date heatmap
- Quantity heatmap by combination of PD/Noise and Lab/Field
- Dedicated quantity and date comparison for Lab Noise / Field Noise
- Error file list

### 4.3 PRPD Pattern Screen

- Select detailed data group: `Lab PD`, `Field PD`, `Lab Noise`, `Field Noise`
- Select fault type of the selected group
- Display only files that satisfy both group and fault type in the detailed file list
- Individual file PRPD image
- Average PRPD image of the selected group/fault type
- Average and maximum Feature of the selected group/fault type
- Average PRPD image by date
- Line chart of 128 average values
- Line chart of 128 maximum values

### 4.4 t-SNE Screen

- Generate and execute 256 Features
- Set perplexity
- Set random seed
- Limit number of analysis files
- Color: Fault type
- Marker shape: `Lab PD`, `Field PD`, `Lab Noise`, `Field Noise`
- Filter by date
- Display filename, date, group on point hover
- Display original PRPD and Feature of the selected point
- Download t-SNE coordinates CSV

### 4.5 Data Details Screen

- Filename
- Measurement date and time
- Group
- Fault type
- Original PRPD
- Average/maximum Feature by phase
- t-SNE coordinates
- Adjacent data in the same cluster

## 5. Recommended Software Structure

Start with a single Streamlit application initially, but separate analysis logic and UI.

```text
prpd-analyzer/
├── app.py
├── requirements.txt
├── claude.md
├── README.md
├── src/
│   └── prpd_analyzer/
│       ├── io.py              # DAT parser, file validation
│       ├── metadata.py        # Folder/filename metadata extraction
│       ├── features.py        # 256 Feature extraction
│       ├── quality.py         # Category quality check (Mahalanobis suspect score, feature diagnostics)
│       ├── embedding.py       # t-SNE execution
│       ├── statistics.py      # Group/date statistics
│       └── plots.py            # Plotly/PRPD visualization
└── tests/
    ├── test_io.py
    ├── test_metadata.py
    ├── test_features.py
    └── test_embedding.py
```

Subsequent tracks are added as separate packages without modifying the Analyzer. The current actual structure of `src/` is as follows.

```text
src/
├── prpd_analyzer/         # Analyzer (Implementation complete)
├── prpd_diffusion/        # Noise removal diffusion (DIFFUSION.md)
│   ├── (2D baseline)      #   Sections 1~18. Implemented, main training (D4) not executed
│   ├── components/        #   Section 19. ARDD paper components (entropy residual, morphological attention)
│   ├── models/unet1d.py   #   Section 19. Conditional 1D U-Net
│   ├── data/profile.py    #   Section 19. Phase profile representation
│   ├── data/cin_noise.py  #   Section 19. Parametric Composite Industrial Noise
│   └── ardd1d/            #   Section 19. ARDD 1D track (Training/evaluation complete)
└── prpd_anomaly/          # Anomaly detection (ANOMALY.md, folder created only, code not started)
```

If the data scale increases or multiple users need to use it simultaneously, separate the API and frontend in the next stage.

```text
React or Streamlit UI
        ↓
FastAPI Analysis API
        ↓
Python Analysis Module
        ↓
Parquet/CSV Result Cache
```

## 6. Step-by-Step Development Order

### Phase 1: Data-driven Stabilization

1. [Completed] Move DAT parser to `PRPD_Analyzer/prpd_analyzer/io.py`
2. Validate header and payload size
3. Write two folder structure metadata tests
4. Generate report for corrupted and abnormal files
5. Validate quantity and group mapping for all files
6. Validate duplication and correspondence of By Date/By Type file lists

Completion criteria:

- Clear distinction between normal and error files out of 3,172 files.
- Include 3,172 files as analysis targets in full data analysis mode.
- Both 227-byte and 167-byte header files are correctly converted to 128×3600 payload.
- Total searched, filtered, and actual analyzed counts are distinguished on the screen.
- Quantities of `Lab PD` and `Field PD` are verified as 1,445 and 513 respectively.
- Quantities of `Lab Noise` and `Field Noise` are verified as 146 and 1,017 respectively.
- Dates are extracted identically in both folder structures.
- By Date and By Type are not summed, and calculations are based on a total of 3,172 when one structure is selected.

### Phase 2: 256 Feature extraction

1. Implement `extract_mean_max_features(matrix)`
2. Validate result Shape `(N, 256)`
3. Generate Feature names
4. Save Features to CSV or Parquet
5. Check distribution of mean and maximum values

Completion criteria:

- 128 means and 128 maximums are generated for each file.
- Input data is not modified.
- Results are reproducible under identical inputs and settings.

### Phase 3: t-SNE Analysis

1. Standardize 256 Features
2. t-SNE 2D transformation
3. Display colors by fault type
4. Separate markers for Lab/Field PD and Lab/Field Noise
5. Add date filters and hover information
6. Download results CSV

Precautions:

- The t-SNE coordinate axes themselves have no physical meaning.
- Perplexity must be smaller than the number of analysis samples.
- Use date-unit separation validation.
- Save the random seed.

### Phase 4: Date and Group Analysis

1. Calculate average Features by group
2. Calculate average Features by date
3. Compare Lab/Field for the same fault type
4. Calculate cluster center changes by date
5. Compare the distribution of Lab Noise and Field Noise
6. Check distribution changes for new dates

### Phase 5: Complete Web UI

1. Organize page or tab structure
2. Add cache and progress indicators
3. Display error files and analysis logs
4. Download CSV, PNG, HTML results
5. Test across screen sizes and browsers

### Phase 6: Deployment and Operation

- Local execution: `streamlit run PRPD_Analyzer/app.py`
- Internal server deployment: Streamlit or Docker
- Specify data folder via environment variable or UI input
- Treat original data as read-only
- Save analysis results and execution settings in a separate results folder

#### Data Generation for AI Analysis

When the Analyzer execution is completed, generate an AI analysis dataset that can be jointly used by AI classification, anomaly detection, and natural language Agents later. However, deep learning model training data does not use the 256-dimensional Feature extraction results, but is generated directly from the validated original `128×3600` PRPD raw data.

The 256-dimensional mean/max Feature of the Analyzer is maintained for statistical analysis, t-SNE, and category quality checking. Deep learning inputs are managed separately as follows:

```text
Analyzer Statistics/t-SNE: 128×3600 raw data → mean/max 256 Feature
Deep Learning Training:    128×3600 raw data → raw tensor → deep learning model
```

During the AI stage, do not randomly reinterpret the original `.dat` files or double count the By Date/By Type folders. Generate raw tensor training data based on the file list and metadata validated by the Analyzer.

AI data generation order:

```text
Search all 3,172 files
→ Validate header and payload
→ Generate group/category/date metadata
→ Select analysis target group
→ Generate original 128×3600 raw matrix
→ Validate dtype/shape/missing values/constant data
→ Normalize for model input and save tensor
→ Record data quality status
→ Save raw data/metadata and execution settings
```

The common contract for deep learning training data is as follows.

```text
sample_id
file_path
source_root_type          # By Date or By Type
group                     # Lab PD, Field PD, Lab Noise, Field Noise, Synthetic
label                     # Corona, Floating, Particle, Void, Noise
date
raw_shape                 # (128, 3600)
raw_dtype                 # uint8 or float32 specified at save
raw_tensor_path           # Original raw tensor location
normalization             # Status at raw generation stage; default none
data_quality_status
quality_reasons
split                     # Unspecified; decided at AI model stage
metadata_version
raw_data_version
run_id
```

Data generation rules:

- Generate `sample_id` as a stable identifier that can identify the same file in both folder views.
- Use only one standard root between `By Date` and `By Type`, and do not merge the two results.
- Groups to include can be selected on the AI data generation screen.
- Selectable groups are `Lab PD`, `Field PD`, `Lab Noise`, `Field Noise`, `Synthetic`.
- Include only files belonging to the selected groups in the AI dataset and raw tensor generation target.
- Unselected groups retain original files but are not included in the dataset of that execution.
- Include only files with normal payload and valid `(128, 3600)` raw matrix in the AI training data.
- Exclude error files from training data but preserve them in a separate quality report and error list.
- Save `Lab Noise`, `Field Noise`, `Synthetic` with label `Noise` but maintain each group respectively.
- For unapproved suspected category data, retain the original label and save `review_status` and `suspect_score` as separate columns.
- Do not automatically change categories; reflect only human-approved modifications in the new metadata version.
- Do not decide train/validation/test split method during the raw data generation stage.
- Save split as unspecified when generating raw dataset, and decide it during AI model execution.
- When choosing date-based splitting at the AI model stage, ensure data from the same date is not mixed between train and test.
- If normalization statistics are needed, calculate them only from the train data of the AI model.
- Save random seed, raw data version, normalization status, and analysis execution ID together.
- Do not include t-SNE coordinates and 256-dimensional Features in deep learning training inputs.
- Record the selected group list and filter conditions per group in `dataset_description.txt`, `run_config.json`, and `manifest.json`.
- Multiple generation executions with different conditions can coexist. The Analyzer does not designate any execution as the sole correct answer, and subsequent tracks select which execution to use respectively.

Generated Artifacts:

Save AI analysis data in a new results folder named with the creation date and time for each execution. Do not overwrite existing execution results or mix files from different executions.

The generated execution folder is a data candidate selected and used by subsequent tracks (anomaly detection, diffusion, classification). Dataset selection rules for each track follow Section 11 "Generated Dataset Selection for Subsequent Tracks".

Recommended folder name format:

```text
ai_data_YYYYMMDD_HHMMSS/
```

Example:

```text
artifacts/ai_data_20260820_153000/
```

Internal folder artifacts:

```text
ai_data_YYYYMMDD_HHMMSS/
├── dataset_description.txt
├── metadata.parquet
├── raw_tensors_v1/
├── ai_raw_dataset_v1.parquet
├── ai_raw_dataset_v1.csv
├── run_config.json
├── data_quality_report.html
└── manifest.json
```

`dataset_description.txt` is mandatory in all execution folders and records the following.

```text
dataset_created_at
dataset_name
source_root_type
source_root_path
source_root_description
data_types_included
selected_groups
group_counts
label_counts
date_range
total_file_count
valid_file_count
excluded_file_count
raw_tensor_shape
raw_tensor_dtype
normalization_method
split_method             # unspecified; decided by AI model
split_dates              # unspecified; decided by AI model
metadata_version
raw_data_version
run_id
```

The description file must include the following content in human-readable sentences.

- Creation date and time
- Types of generated data: raw tensor, metadata, raw data list, quality report, etc.
- Original data path and standard for `By Date`/`By Type`
- Explanation that the two folders are different views of the same data and only one root was used in this execution
- Included groups: Lab PD, Field PD, Lab Noise, Field Noise, Synthetic
- List of groups selected and not selected in the actual execution
- Label mapping and file count per group
- Count of normal files and excluded files, and reasons for exclusion
- raw tensor shape, dtype, normalization method
- raw data generation settings and random seed; split method is recorded separately at the AI model stage
- raw data version, metadata version, execution ID needed for reproducibility (not Features)

In `manifest.json`, record relative paths, file sizes, creation times, and checksums if possible for each artifact. Include `dataset_description.txt` and `manifest.json` themselves in the execution artifacts list.

Deep learning input formats:

- 2D CNN: `(batch, 1, 128, 3600)`
- 1D model: `(batch, 128, 3600)` or separate axis transformation depending on the intended use of the phase/time axes
- Autoencoder: Input raw tensor and reconstruction target configured in the same `128×3600` shape
- Classification model: Connect raw tensor and label, and use group/date as evaluation and splitting criteria

raw data preprocessing principles:

- Preserve original `uint8` raw data unmodified.
- Saved versions for model inputs can be converted to `float32` according to specified normalization settings.
- By default, do not apply normalization during the raw data generation stage.
- Select `/255` normalization or standardization based on the train set during the AI model stage and record in execution settings.
- Use only train data statistics of the AI model for normalization statistics for validation/test.
- If using random crop, time window split, or augmentation, track original sample_id and transformation settings.

AI dataset generation completion criteria:

- Verify if total searched count, normal data count, and excluded data count match.
- Verify if AI raw tensor Shape is `(normal file count, 128, 3600)`.
- Compare data count by group, label, and date against original analysis report.
- Verify if split is unspecified during the raw dataset generation stage.
- Separately verify the split method and date list selected during the AI model execution stage.
- Verify that identical files from By Date and By Type are not redundantly saved.
- Ensure the same `run_id` rules and raw tensor results can be reproduced when generating with the same inputs and settings.
- Be able to distinguish execution results purely by creation date/time folder.
- `dataset_description.txt` must exist in each execution folder explaining the generated data types and original paths.
- File count, group count, and excluded count in the description file must match actual artifacts.
- Files other than the selected groups must not be included in the dataset.

#### Analysis Results Report and Category Quality Check

In operation, do not just save simple t-SNE figures; generate a separate report for data where the current category or label is highly likely to mismatch the actual PRPD pattern. This report is not for automatically changing categories, but is used as review material to prioritize targets for researchers or operators to re-examine the original files.

Suspected data candidates are calculated by combining the following grounds:

- **(Implementation complete)** High **pooled shrinkage Mahalanobis distance** based on `(group, label)` cell
- **(Implementation complete)** When another label center within the same group is closer than its own center (`margin > 0`)
- When specific files consistently deviate from clusters by date
- When the predicted category of the classification model repeatedly mismatches the existing category
- When the categories of t-SNE neighbor data mostly differ from the existing category
- When there are data quality issues such as payload errors, constant Features, or abnormal average/maximum values

##### Distance Definition (`PRPD_Analyzer/prpd_analyzer/quality.py`)

The 256-dimensional phase feature has strongly correlated adjacent bins, so **Euclidean distance is not used.**
Lab PD label 4-class classification macro F1 measured by date-based `GroupKFold(5)` on `ai_data_20260825_101830` (3,172 files):

| feature set | euclid-centroid | maha-pooled | maha-perclass |
|---|---:|---:|---:|
| mean(128) | 0.476 | 0.552 | 0.379 |
| max(128) | 0.522 | 0.540 | 0.280 |
| **mean+max(256)** | 0.545 | **0.644** | 0.349 |

Three things are determined here:

1. **Covariance is estimated as pooled.** Cell-by-cell estimation is worse than Euclidean because the sample size (286~562 per cell) cannot handle 256 dimensions.
2. **mean+max > max > mean.** 256-dimensional combination actually contributes.
3. **Use a fixed value α=0.5 for shrinkage instead of LedoitWolf.** LW under-shrinks on this data and chose 0.013 on pooled. After log1p transformation, the α sweep has a plateau of 0.4~0.6 and peaks at 0.5 with a macro F1 of 0.728.

Apply `log1p` exclusively for score calculation. **Do not change the saved 256-dimensional feature definition (Section 3) and t-SNE inputs.** The self-cell average is calculated as a **leave-one-out average** excluding that sample. Without correction, a paradox occurs where cells with fewer samples appear to be the safest.

Competitive candidates are limited to **other labels within the same group.** Mixing groups disguises Lab/Field domain gaps as mislabeled signals (same-label classification performance differs: Lab PD 0.728 vs Field PD 0.28).

##### Threshold — Empirical percentile

Theoretical chi-square thresholds are not used. This is because strong shrinkage compresses the distances, making the actual distribution much denser than chi2(256).

```text
Measured d²      median 70.9   p95 232.1   p99 314.2
chi2(256)                  p95 294.3   p99 311.6   (median ≈ 255)
```

Mark the top 1% / 5% based on all valid files as `suspect_flag_p99` / `suspect_flag_p95`, and record the applied absolute d² cut values together for reproducibility. Apply flags to `margin` in the same way. Relative position within its own cell is provided as `cell_percentile`.

Cells with less than `MIN_CELL_COUNT` (default 10) are excluded from the criteria. This is because the average is virtually itself, bringing the distance close to 0 and **falsely reassuring problematic files as "normal".** Leave these files in the report as `reference_status='insufficient'`, `suspect_score=NaN` but display them separately on the screen. In the current data, `Field PD / Particle` (3 files) is the only excluded cell.

Record the following information for each candidate, not a single judgment value.

```text
sample_id
file_path
current_group
current_category
date
suspect_score                          # Mahalanobis d² to its own cell (LOO corrected)
suspect_reason
nearest_category                       # Nearest other label within the same group
d2_nearest
margin                                 # suspect_score - d2_nearest. If positive, mislabeling suspected
cell_percentile
reference_status                       # ok | insufficient
suspect_flag_p99 / suspect_flag_p95
margin_flag_p99 / margin_flag_p95
model_predicted_category               # After Phase 7
model_prediction_probability           # After Phase 7
tsne_neighbor_categories               # Not implemented
data_quality_status
review_status
reviewer
review_note
```

`suspect_score` does not mean a category change. It indicates that the higher the score, the higher the priority for review, and the final category is modified through a separate approval process after checking the original PRPD and measurement context. Do not finalize categories based solely on model predictions or t-SNE positions.

##### Feature Quality Diagnostics — Not a classifier

Create a separate table (`feature_set_diagnostics`) to verify if `Mean` / `Max` / `Mean+Max` actually separate PD classes. **Do not use this for calculating suspect scores.**

Silhouette alone cannot make this judgment. No preprocessing exceeds 0.06, and in particular, it cannot distinguish between `max` (0.008) and `mean+max` (0.012).

```text
Lab PD, label-based silhouette
  raw 0.034 | StandardScaler 0.012 | log1p 0.058 | L2 normalization 0.053 | cosine 0.062
By scope (mean+max, StandardScaler)
  ALL -0.098 | Lab PD +0.012 | Field PD -0.272 | PD only -0.036
```

This means the classes are not clustered in spherical shapes, which is exactly why Mahalanobis is needed. Therefore, **include the date-based CV macro F1 in the same table and use this to determine the superiority of the feature sets.**
In actual measurements, `mean+max` ranked 1st in **all** combinations of 12 scopes × transformations.

Report generation items:

- Total files, normally analyzed files, error files
- Number of data by Lab PD, Field PD, Lab Noise, Field Noise, Synthetic
- Number of suspected data by category
- Suspected data distribution by group × category × date
- List of top suspect score files
- Confusion matrix of existing category and predicted category
- Comparison of average PRPD by category and suspected file PRPD
- Comparison of suspected file mean/max Features and representative normal files
- Original file path, analysis version, Feature version, model version, random seed

Output format:

- HTML/PDF summary report for human review
- CSV or Parquet detailed list for subsequent review
- PRPD images and Feature graphs for each suspected file
- `run_config.json` for execution reproducibility

Operating procedure:

1. Analyze all 3,172 files from a single standard root.
2. Separate error files and data quality anomalies first.
3. Calculate category suspect scores with validated 256-dimensional Features.
4. Record the top suspected data list and grounds in the report.
5. Researchers verify the original PRPD, date, Lab/Field, and PD/Noise context.
6. Record the review results as `review_status`, `reviewer`, `review_note`.
7. Save only approved modified categories as a new metadata version.
8. Preserve original categories before modification and modification history.

Completion criteria:

- The category quality check report is generated in every analysis execution.
- Suspected data is displayed with scores and judgment grounds.
- Human review status can be managed without automatic category changes.
- Metadata versions before and after incorrect category modifications and change history are tracked.
- Identical data by date/by type is not double-counted in the report.

## 7. Performance Plan

Apply the following to avoid reading all 3,172 files every time.

- Cache file lists and metadata
- Cache Features per file
- Cache t-SNE results by analysis condition
- Include the selected root structure in the cache key, and do not mix by date/by type results.
- Save Feature results as Parquet
- Limit the maximum number of samples on the screen
- Design full analysis to be separable as a background task

For large-scale t-SNE execution, use the following sequence.

```text
All files
→ Apply filters
→ Uniform or seed-based sampling
→ Generate 256 Features
→ StandardScaler
→ t-SNE
```

## 8. Test Plan

### Unit Tests

- Shape and dtype of normal DAT files
- Invalid header length
- Invalid payload length
- Mapping `랩_PD` → `Lab PD`
- Mapping `현장_PD` → `Field PD`
- Mapping `랩_Noise` → `Lab Noise`
- Mapping `현장_Noise` → `Field Noise`
- Mapping `랩_Noise`, `현장_Noise`, `임의생성` → `Noise` label
- Date extraction from date folder and filename
- Feature Shape `(N, 256)`
- Calculation results of average and maximum values
- Perplexity correction when sample size is small

### Integration Tests

- Execution from data folder selection to t-SNE CSV download
- Result when only Lab PD is selected
- Result when only Field PD is selected
- Comparison result between Lab PD and Field PD
- Result when only Lab Noise is selected
- Result when only Field Noise is selected
- Comparison result between Lab Noise and Field Noise
- Result when only a specific date is selected
- Verify if normal file analysis continues even if there are error files

### Analysis Validation

- Verify if results are identical when repeatedly analyzing the same files with the same seed
- Date-based train/test splitting
- Reporting data imbalance by fault type
- Analysis separating Lab Noise and Field Noise
- Comparison analysis between PD and Noise by Lab/Field
- Comparison of Feature results based on `max` and percentile

## 9. Completion Criteria

The initial release must satisfy the following.

- A data folder can be specified in the web browser.
- Lab PD, Field PD, Lab Noise, and Field Noise can be independently selected.
- Fault types and dates can be filtered.
- 256-dimensional Features are generated from each file.
- t-SNE results can be viewed by fault type and group.
- Average PRPD, average Feature, and maximum Feature can be viewed.
- Analysis results can be downloaded as CSV.
- Error files and analysis settings are displayed to the user.
- The possibility of date bias can be checked through date-based validation.

## 10. Current Implementation and Next Steps

The current `app.py` has the following features.

- Streamlit UI using the `src/prpd_analyzer` module
- Lab/Field PD and Noise group filters
- Date and fault type filters
- 256-dimensional Feature combining 128 phase averages and 128 maximums
- Summary statistics and quantity dashboard by date
- Average/maximum phase Feature and original PRPD image
- 256-dimensional Feature-based t-SNE visualization
- Checking error files and downloading result CSV
- Category quality review report (pooled shrinkage Mahalanobis, Top 1%/5% percentile) and CSV download
- Feature quality diagnostics table (Mean / Max / Mean+Max × silhouette + date-based CV macro F1)
- Generating raw `128×3600` AI dataset of selected groups
- Generating `ai_data_YYYYMMDD_HHMMSS` result folder and `dataset_description.txt`
- Generated execution result in `artifacts/ai_data_20260825_101830/` and saved raw tensor by group (3,172 files, `raw_v2`)
- `PRPD_Analyzer/scripts/build_ai_dataset.py` — generates the same dataset via CLI instead of UI (using the same functions)

### Execution Status Verification

- Local server starts normally with `python -m streamlit run PRPD_Analyzer/app.py --server.headless true --server.port 8501`.
- Verified `HTTP 200` response from `http://localhost:8501` request.
- A process directly run in the VS Code integrated terminal may terminate when VS Code is closed.
- To keep it running even after closing VS Code, run it as a detached background process in PowerShell.

```powershell
Start-Process python `
        -ArgumentList "-m streamlit run PRPD_Analyzer/app.py --server.headless true --server.port 8501" `
        -WorkingDirectory "C:\Users\USER\Documents\Yong\202608_PRPD" `
        -WindowStyle Hidden
```

- Detached execution processes are accessed at `http://localhost:8501`.
- Verified `HTTP 200` response with detached execution method even after closing VS Code integrated terminal.
- Stop the running Streamlit process by terminating `python.exe` in Task Manager or terminating the process on that port.

### Artifacts Invalidated by Axis Flip Bug (2026-08-25)

The axis flip bug from Section 2.1 propagated to **all** artifacts below the parser. The following are invalid and must not be used as inputs for training, evaluation, statistics, or paper figures. The files are kept for history preservation.

| Invalid Artifact | Indicator File |
|---|---|
| `artifacts/ai_data_20260820_170628` (3,172) | `STALE_AXIS_BUG.md` |
| `artifacts/ai_data_20260820_172229` (2,021) | `STALE_AXIS_BUG.md` |
| `artifacts/ai_data_20260821_160405` (1,384) | `STALE_AXIS_BUG.md` |
| `Results/diffusion_runs/` 12 ARDD 1D runs + comparison table + `ardd1d_synth_check*` | `STALE_AXIS_BUG.md` |
| Observation of Lab→Field profile in `DIFFUSION.md` 19.5 | Warning in that section |

What is **not** invalid: impossibility of comparing β schedules, prohibiting single seed adoption, including `ai_data_root` in cache key — these are methodological conclusions independent of the data, so they remain valid.

The replacement dataset is `artifacts/ai_data_20260825_101830` (3,172 files, all 5 groups, `raw_v2`) and the `ai_data_root` of all 8 `configs/*.json` points here.
`artifacts/repr_cache/` and `profile_cache/` are separated by execution name, so a new run creates a new cache.

### Subsequent Tracks Status (2026-08-25)

| Track | Document | Location | Status |
|---|---|---|---|
| Noise removal diffusion (2D baseline) | `DIFFUSION.md` Sections 1~18 | `Diffusion/prpd_diffusion/` | Data contract, splitting, model, evaluation implemented, **main training (D4) still not executed** |
| ARDD 1D Track | `DIFFUSION.md` Section 19 | `Diffusion/prpd_diffusion/ardd1d/` | Implementation complete. **All training and evaluation results invalid due to axis bug — needs re-execution** |
| Anomaly Detection | `ANOMALY.md` | `EfficientAD/` | **Only empty folder created. Code not started (Starts from Phase A0)** |
| AI Fault Classification | `claude.md` Phase 7 | Not created | Not started |

Tests: 154 passed / 0 failed / 0 skipped (torch 2.11.0+cu128 installation complete, RTX 5080 sm_120).
8 tests in `tests/test_io.py` defend against axis regression, and 18 tests in `tests/test_quality.py` defend the suspect score definition.

The ARDD 1D track is an **explicit exception** to the rule "do not use 256-dimensional features as deep learning input" in Section 11 of `claude.md`. The purpose is to verify the technique on 1D signals similar to the paper (ARDD-2025), and it is recorded as `feature_input_exception: true` in each run. The output of this track is a non-reversible representation, so it is not passed as input to the classification/anomaly detection tracks. For detailed premises and limitations, refer to Section 19.1 of `DIFFUSION.md`.

The ARDD 1D track selects only Lab PD + Field PD with `clean_groups`/`real_noisy_groups`, so it **does not use noise groups** and uses parametric CIN instead of synthesizing actual noise (19.3).

Each track independently selects the ai_data execution to use (Section 11 "Generated Dataset Selection for Subsequent Tracks").
After the axis bug fix, both tracks point to the same execution, but the pool diverges based on the group selection in the config.

| Track | `ai_data_root` | pool (after group selection) |
|---|---|---|
| diffusion 2D baseline | `artifacts/ai_data_20260825_101830` | 3,121 (PD + 4 Noise groups) |
| ARDD 1D | `artifacts/ai_data_20260825_101830` | 1,958 (Lab PD 1,445 + Field PD 513) |

The pool has grown compared to the old execution (2D baseline 2,021 → 3,121, ARDD 1D 1,384 → 1,958).
This is because the two old executions were subsets filtered and sampled in the UI, whereas the new execution includes all 5 groups.
Since date splitting and sample sizes will change, relevant figures must be recalculated upon re-execution.

If `ai_data_root` is left empty, the latest folder is automatically selected, changing the target whenever execution folders increase.
**Always explicitly specify it in the config.**

All tracks compare several algorithms referring to papers. The comparison conventions and candidate lists are in Section 18 of `DIFFUSION.md` and Section 17 of `ANOMALY.md`. Place the original papers in `ref/` of each track folder and register them in the document's registry table.

- diffusion: Received ARDD paper (Sci. Rep. 2025) and 2 survey materials in `Diffusion/prpd_diffusion/ref/` (2026-08-21).
  Porting and comparing ARDD components is **completed in the ARDD 1D track** (`DIFFUSION.md` 19.6).
- Anomaly Detection: Not received.

The proposed techniques from the papers are not ported entirely but **broken down into components to be turned on and off for comparison**, and the figures from the papers, which use different evaluation metrics from ours, are used only as design references, not improvement targets.

Facts verified in the ARDD 1D track that also affect other tracks:

- ~~**The Lab→Field gap in the profile space is not a difference in "noise floor".** Field PD has a lower time average and higher time maximum than Lab PD~~ — **Invalidated on 2026-08-25.** This observation was calculated from profiles with mixed axes (Section 2.1). Do not use this as a basis for noise synthesis design until recalculated with the new dataset. See `DIFFUSION.md` Section 19.5.
- **ε prediction val losses cannot be compared between different β schedules.** Since the loss weighting varies, candidate selection must be strictly based on final evaluation metrics.
- **Do not adopt candidates based on single seed results.** Actually, the "only improvement" in a single execution turned out to be noise when the seed was changed. Repeat seeds but **fix the date splits and synthetic noise.**

The next implementation priorities are as follows:

1. Add unit tests and date-based validation
2. Save Feature and t-SNE result caches (include `ai_data_root` in cache key)
3. Manage analysis execution settings and result versions
4. Expand to AI classification/anomaly detection stages

## 11. Expansion Roadmap after Analyzer

After the data analysis Analyzer is completed and the raw matrix and metadata are stabilized, develop the following features sequentially utilizing the same data assets.

```text
Analysis Analyzer
→ Validated 128×3600 raw tensor/metadata repository
→ AI Fault Classification
→ Anomaly Detection
→ Natural Language Analysis Agent
→ Generation of reproducible results for papers
```

Subsequent features do not reinterpret the original `.dat` files individually. Use the common data contract.

```text
sample_id
file_path
group              # Lab PD, Field PD, Lab Noise, Field Noise, Synthetic
label              # Corona, Floating, Particle, Void, Noise
date
raw_tensor_path
raw_shape          # (128, 3600)
ai_data_root       # Execution folder providing this data
raw_data_version
normalization
split
```

The 256-dimensional mean/max Feature is used for the Analyzer's statistics, t-SNE, and category quality checking, and is not used as input for deep learning classification and raw-data based anomaly detection.

Since By Date and By Type are different views of the same data, files are not used redundantly in AI training and validation.

### Generated Dataset Selection for Subsequent Tracks

There may be multiple `ai_data_YYYYMMDD_HHMMSS` execution folders with different conditions in `artifacts/`. Anomaly detection (`prpd_anomaly`), diffusion (`prpd_diffusion`), and classification models **independently select the execution they will use among these.** The selection of one track does not force the selection of another track.

Selection rules:

- Specify the execution path to use (`ai_data_root`) in the configuration file of each track. Do not rely on automatic selection of the latest folder. This is because the target pointed by automatic selection changes as execution folders increase.
- **Within a single run, use only one ai_data execution folder.** Do not merge or double count files from multiple execution folders. This is the same reason for the rule of not mixing `By Date`/`By Type` roots.
- Record the absolute path of the selected `ai_data_root` and the `raw_data_version` of that execution in the track's `run_config.json`, `dataset_description.txt`, and result tables. Without this record, it's impossible to verify which data the results belong to after the fact.
- Different datasets can be selected per track, but **when combining or comparing results from two tracks side-by-side, they must be the same dataset.** Example: Decomposing the anomaly score into `s_noise`/`s_novel` using a diffusion reconstruction is only valid within the same `sample_id` set and same date split. Joint runs verify the match of `ai_data_root` and `raw_data_version` of both runs, and do not execute if they mismatch.
- Include the selected `ai_data_root` in the keys of intermediate caches (memmap, feature, t-SNE results). If the cache does not distinguish executions, changing the dataset will result in reusing the previous execution's data as is.
- Do not modify files and metadata in the execution folder being used. To change the scope or conditions, create a new execution in the Analyzer and specify that path in each track's settings.
- Changing a dataset changes the date split, sample size per group, and reference distribution for that track. Recalculate relevant figures and update documents/reports upon replacement.

Each track document (`ANOMALY.md`, `DIFFUSION.md`) specifies its own defaults and rationale for selection under this rule.

### Phase 7: AI Fault Classification

The goal is to automatically classify `Corona`, `Floating`, `Particle`, `Void`, and `Noise`, and evaluate whether a model trained on `Lab PD` operates on `Field PD`.

Development sequence:

1. Validate 256 Feature and label quality
2. Train/validation/test split by date
3. Implement baseline models
         - Logistic Regression
         - Random Forest
         - XGBoost or LightGBM
4. Compare CNN, Vision Transformer, or time-series models based on `128×3600` raw tensor if necessary
5. Correct class imbalance
6. Cross-domain evaluation of Lab/Field
7. Display probabilities and prediction grounds in UI

Evaluation metrics:

- macro F1
- balanced accuracy
- precision, recall, F1 by class
- confusion matrix
- calibration and prediction probabilities
- performance by date
- performance degradation from Lab → Field

The classification model does not use random splitting by file as the default validation. If data from the same date or same measurement condition is mixed in train and test, performance can be overestimated, so date-based splitting is prioritized.

### Phase 8: Anomaly Detection

The goal is to detect PRPD data that is not included in the existing fault types or deviates from the normal group distribution.

Analysis targets:

- Data deviating from the Lab PD distribution
- Newly appearing patterns in Field PD
- Abnormal patterns in Lab Noise and Field Noise
- Corrupted files or abnormal payloads
- Clusters occurring only on specific dates

Candidate methods:

- Isolation Forest
- Local Outlier Factor
- One-Class SVM
- `128×3600` raw tensor-based Autoencoder
- Distance-based detection by fault type
- t-SNE is used only for visualization and not directly as a criterion for determining anomalies.

The above list is a starting candidate, and expanded candidates referencing papers and fair comparison conventions are documented in Section 17 of `ANOMALY.md`. Whichever technique is adopted, compare with the same dataset, same date split, and same validation protocol, and record the rationale for adoption/exclusion in the results.

Save the following information in the anomaly detection results.

```text
sample_id
anomaly_score
threshold
detector
reference_group
nearest_samples
reason_features
ai_data_root          # Generated dataset selected by this execution
raw_data_version
```

Anomaly detection runs independently select the generated dataset to use separately from the diffusion track, and record the selected values in the fields above and `run_config.json`. Dataset matching is only required in runs that combine with diffusion reconstructions (Section 11).

Do not arbitrarily fix the threshold, but set it based on the percentile of validation data or cases confirmed by operators. Represent anomaly detection in the UI so that it means "unknown fault type" rather than immediately concluding it is a physical fault.

### Phase 9: Natural Language Analysis Agent

The goal is to allow users to input questions about dates, groups, and fault types in natural language, and have the Agent answer based on saved analysis results and visualizations.

Example questions:

- "What is the difference in Corona Features between Lab PD and Field PD in September 2024?"
- "Which phase interval has the highest average value in Field Noise?"
- "Find files in recent dates that show a different cluster from previous dates."
- "Explain the classification result and anomaly detection score of this file."

The Agent's tools are limited to the following scope.

- Query metadata filters
- Query statistics by group/date
- Query average/maximum values of 256 Features
- Query classification model predictions
- Query anomaly detection scores
- Query saved t-SNE coordinates and adjacent samples
- Generate graphs for specified results

The Agent does not arbitrarily modify original files, and displays the used filters, number of files, analysis version, and model version together with each answer. If there is no supporting data, it does not guess, but asks for additional conditions or answers "no analysis results found."

Recommended structure:

```text
Web UI
        ↓
Natural-language Agent
        ↓ tool calls
Analysis result store / model registry / chart service
```

### Phase 10: Paper Writing Support

The goal is to convert the results generated from the Analyzer and AI models into reproducible tables, figures, and methodology documents for papers.

Targets for automatic generation:

- Dataset composition and quantity table by group
- Comparison table between Lab/Field PD and Noise
- Data distribution table by date
- 256 Feature definition and formulas
- Average PRPD and maximum Feature figures
- t-SNE figures
- Classification performance table and confusion matrix
- Anomaly detection case figures
- Experimental setup and data splitting information
- File identifiers for result CSVs and figures

Must record the following information in paper results.

- Data standard root: `By Date` or `By Type`
- Deduplication rule
- Number of files and number of excluded files
- Feature version
- Model version
- random seed
- List of train/validation/test dates
- Python and main library versions
- Execution time and configuration file

When the Agent generates paper sentences, it writes them solely based on the saved figures and results, and does not arbitrarily add unverified physical causes or clinical/industrial conclusions. The final paper sentences and interpretations are reviewed by researchers.

### Phase 11: Model and Result Management

Version control the following artifacts for subsequent AI features.

```text
artifacts/
├── metadata.parquet
├── features_mean_max_v1.parquet
├── raw_tensors_v1/
├── ai_raw_dataset_v1.parquet
├── ai_data_YYYYMMDD_HHMMSS/        # Multiple executions can coexist. Subsequent tracks select one each
│   ├── dataset_description.txt
│   ├── manifest.json
│   ├── metadata.parquet
│   ├── raw_tensors_v1/
│   ├── ai_raw_dataset_v1.parquet
│   ├── run_config.json
│   └── data_quality_report.html
├── diffusion_runs/                 # Record selected ai_data_root in run_config.json
├── anomaly_runs/                   # Record selected ai_data_root in run_config.json
├── embeddings/
├── models/
│   ├── classifier_v1
│   └── anomaly_detector_v1
├── model_runs/
│   └── run_YYYYMMDD_HHMMSS/
│       ├── split_config.json
│       ├── model_config.json
│       └── evaluation_report.json
├── reports/
└── runs/
                └── run_config.json
```

Each model is saved together with the following.

- Used generated dataset path (`ai_data_root`)
- Training raw data version
- Used groups and labels
- Date splits
- Preprocessing scaler
- Normalization settings
- Model parameters
- random seed
- Evaluation metrics
- Training data identifiers

## 12. Subsequent Stage Completion Criteria

### AI Classification

- Independent testing by date unit is possible.
- Performance by fault type and Lab → Field performance can be verified.
- Prediction probabilities and model versions can be saved.

### Anomaly Detection

- Anomaly detection can be executed separating Lab/Field and PD/Noise.
- Anomaly scores, thresholds, reference groups, and original files can be tracked.
- The generated dataset to use can be selected independently of the diffusion track, and the selected value is recorded in the execution results.
- Dataset mismatches are automatically detected in executions combining diffusion results.
- Displayed so that anomaly detection is not confused with physical fault determination.

### Natural Language Agent

- User's questions can be converted into date, group, and fault type filters.
- Answers include the number of files and analysis result grounds.
- Does not guess beyond the analysis results.

### Paper Support

- Tables and figures can be regenerated with identical execution settings.
- Data splits, Features, models, and seeds can be tracked.
- Paper drafts are provided for researcher review only.

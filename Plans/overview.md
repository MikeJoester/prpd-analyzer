# Dataset Overview and t-SNE Visualization Plan

This document provides an overview of the recently reorganized dataset (based on `Data/README_DATA.md`) and outlines the plan for applying a 256-dimensional t-SNE visualization as requested by the professor.

## 1. Dataset Reorganization Overview

The raw dataset, originally a mix of various collection dates and classes, has been successfully reorganized and categorized into two main folder structures: **by_date** and **by_type**. This enables easier access for both chronological analysis and model training based on labels.

### `Data/by_date/` (Grouped by Collection Context)
This folder separates the data based on its collection origin (Lab vs. Field) by analyzing collection patterns and timestamps.
*   **Lab/**: Experimental data collected in controlled environments. Characterized by dense, clustered multi-class collection on specific dates (12 dates, 1,591 files).
*   **Field/**: In-situ data collected from field operations. Characterized by sparse, specific-class occurrences (208 dates, 1,530 files).
*   **Artificially_Generated/**: 51 files containing pure noise, artificially inserted for model robustness (indicated by a 0 timestamp).

### `Data/by_type/` (Grouped for Machine Learning)
This folder contains copies of all 3,172 files directly structured by their Partial Discharge (PD) label and collection context, which is ideal for direct ingestion into classification models.
*   **Lab_PD/**: 1,445 files divided into `Corona`, `Floating`, `Particle`, and `Void`.
*   **Field_PD/**: 513 files divided into `Corona`, `Floating`, `Particle`, and `Void`.
*   **Lab_Noise/**: 146 experimental noise files.
*   **Field_Noise/**: 1,017 field noise files.
*   **Randomly_Generated/**: 51 artificial noise files.

---

## 2. t-SNE Visualization Plan (256x1 Feature Extraction)

To evaluate the separability of the new dataset classes, we will generate a 2D scatter plot using t-SNE. Based on the reference code in `TSNE_Visualization.py`, the following plan maps out the execution steps to adapt the extraction and plotting pipeline for the new `Data/by_type/` structure.

### Step A: Data Preprocessing and Format Validation
*   **Current State:** `README_DATA.md` notes that the raw data files are `.dat` binaries (with a Unix timestamp header). However, `TSNE_Visualization.py` expects `.csv` files containing a numeric matrix.
*   **Action:** If the files in `by_type` are currently `.dat` binaries, they must first be converted into `.csv` (or parsed dynamically via a Python script) to form the required 2D Phase-Resolved Partial Discharge (PRPD) matrices before extracting features.

### Step B: 256x1 Dimension Feature Extraction
*   **Logic:** Re-use the `extract_features` function from `TSNE_Visualization.py`.
*   **Process:** For each sample's matrix data, two 128-dimensional vectors will be computed across the phase axis (axis 0):
    1.  `max_vals`: The maximum amplitude/value at each phase angle (128x1).
    2.  `density`: The ratio of the max value to the number of non-zero occurrences (`max_vals / non_zeros`) (128x1).
*   **Output:** The two vectors are concatenated to yield the final `256x1` feature vector for the sample.

### Step C: Code Adaptation for New Folders
*   Update the `get_standard_label()` mapping in the script to correctly interpret the subfolder names inside `Data/by_type/` (e.g., merging `Lab_Noise`, `Field_Noise`, and `Randomly_Generated` into a single `Noise` label, and keeping `Corona`, `Floating`, `Particle`, and `Void`).
*   Update the `folders` list in the `__main__` block to point to `["Data/by_type/Lab_PD", "Data/by_type/Lab_Noise", "Data/by_type/Field_PD", "Data/by_type/Field_Noise", "Data/by_type/Randomly_Generated"]`.

### Step D: t-SNE Dimensionality Reduction and Plotting
*   Compile the matrix `X` (N x 256) and label vector `y` (N x 1) for all samples.
*   Run `sklearn.manifold.TSNE` (`n_components=2`, `perplexity=30`) to project the 256-dimensional vectors down to 2 dimensions.
*   Use `seaborn.scatterplot` to visualize the projection, uniquely color-coding the five main classes (`Noise`, `Corona`, `Floating`, `Particle`, `Void`) to assess how well the distinct discharge types form identifiable clusters. We can also optionally use different marker shapes to distinguish `Lab` vs `Field` data.

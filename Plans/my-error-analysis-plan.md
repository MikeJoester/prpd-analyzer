# Subsection X.X: Error Analysis

*Author Note: This section should immediately follow your main Results/Performance Metrics section. The goal is to contextualize the AUROC/PRO scores by examining the failure cases.*

## 1. Quantitative Breakdown of Errors
*   **Objective:** Provide a high-level summary of *how* the model is failing before showing *what* it is failing on.
*   **Key Elements to Include:**
    *   A breakdown of False Positives (FPs - normal samples flagged as anomalous) vs. False Negatives (FNs - anomalies missed).
    *   Mention of the threshold used to determine these binary classifications (e.g., F1-max threshold).
    *   *Drafting Prompt:* "While PatchCore achieved an image-level AUROC of [X], an analysis of the confusion matrix at the optimal F1 threshold reveals a tendency toward [False Positives / False Negatives]..."

## 2. Categorization of Failure Modes
*   **Objective:** Group the errors into logical, dataset-specific categories.
*   **Category 1: Over-sensitivity (False Positives)**
    *   *Description:* Discuss normal images that yielded high anomaly scores.
    *   *Common Causes:* Variations in lighting, harmless surface marks, shadows, or background noise not adequately represented in the training set.
*   **Category 2: Under-sensitivity (False Negatives)**
    *   *Description:* Discuss anomalous images that yielded low anomaly scores.
    *   *Common Causes:* Subtle defects, low-contrast anomalies, or defects that closely mimic normal structural features.
*   **Category 3: Localization Mismatches (if evaluating pixel-level AUROC/PRO)**
    *   *Description:* Cases where the model flagged the image correctly, but the anomaly heatmap highlighted the wrong area.

## 3. Visual Analysis (Anomaly Maps)
*   **Objective:** Show the reader exactly what the model is "seeing" when it makes a mistake.
*   **Figure Recommendation:** Include a figure (e.g., a 3x3 grid) showing:
    *   Column 1: Original Image
    *   Column 2: Ground Truth Mask
    *   Column 3: PatchCore Predicted Heatmap
*   **Discussion Points:** 
    *   Point out specific examples in the figure (e.g., "As seen in Figure X, row 2, the anomaly heatmap strongly activates on the edge of the object rather than the central defect...").

## 4. Root Cause Hypothesis (PatchCore Mechanics)
*   **Objective:** Connect the observed errors back to how PatchCore works under the hood. 
*   **Potential Causes to Discuss:**
    *   **The Backbone:** Does the ImageNet pre-trained WideResNet struggle to extract meaningful features from your highly specialized dataset? 
    *   **The Memory Bank (Coreset):** Did the subsampling process (e.g., retaining only 1% or 10% of normal features) discard rare but normal variations, leading to FPs?
    *   **Neighborhood Pooling:** Did the local patch aggregation blur out extremely small, pixel-level defects?

## 5. Implications and Potential Solutions
*   **Objective:** End on a constructive note by outlining how future research or deployment strategies can mitigate these errors.
*   **Ideas to Include:**
    *   Fine-tuning the feature extractor on domain-specific data.
    *   Adjusting the coreset sampling ratio.
    *   Applying targeted data augmentation to the normal training set to cover benign variations.
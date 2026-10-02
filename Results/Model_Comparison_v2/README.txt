Final PD-vs-Noise comparison table on the v2 data (PatchCore vs EfficientAD vs One-Class SVM).

    python Comparison/compare_v2.py --out Results/Model_Comparison_v2

Every method and baseline is scored by the same function from its saved eval/file_scores.csv, using the
window aggregation chosen on val (val_selection.json). Cells are mean +/- sd over the training seeds, with the median and range in brackets; the 256-D
baselines are deterministic (one value).

Optimal F1 / Precision / Recall: MACRO averages over the two classes (PD, Noise) at the threshold that
maximizes macro F1 on the TEST set itself (same definition style as the old v1 "Optimal F1"). Because the
threshold is tuned on test, these are optimistic upper bounds. AUROC is threshold-free and unaffected.
The val-threshold F1/P/R are still computed and saved in per_seed.csv (columns f1, precision, recall).
Pixel-wise AUROC is N/A (no pixel ground truth); window-level AUROC replaces it.

Test set: 75 PD + 154 Noise Field files from dates never seen in training.

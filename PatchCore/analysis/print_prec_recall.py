import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve

# PatchCore
data = np.load("Results/patchcore/patchcore_raw_scores.npz")
scores = data['scores']
y_true = data['labels']

precision, recall, thresholds = precision_recall_curve(y_true, scores)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
best_idx = np.argmax(f1_scores)

print(f"--- PatchCore ---")
print(f"Optimal Threshold: {thresholds[best_idx] if best_idx < len(thresholds) else 'N/A'}")
print(f"F1-Score: {f1_scores[best_idx]:.4f}")
print(f"Precision: {precision[best_idx]:.4f}")
print(f"Recall:    {recall[best_idx]:.4f}")


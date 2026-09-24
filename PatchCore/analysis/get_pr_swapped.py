import numpy as np
from sklearn.metrics import precision_recall_curve

data = np.load("Results/patchcore/patchcore_raw_scores.npz")
scores = data['scores']
y_true = data['labels']
y_true_swapped = 1 - y_true

precision, recall, thresholds = precision_recall_curve(y_true_swapped, scores)
f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
best_idx = np.argmax(f1_scores)
best_f1 = f1_scores[best_idx]
best_p = precision[best_idx]
best_r = recall[best_idx]

print(f"Optimal F1 Score: {best_f1:.4f}")
print(f"Precision: {best_p:.4f}")
print(f"Recall: {best_r:.4f}")

import numpy as np
from sklearn.metrics import precision_recall_curve

data = np.load("Results/patchcore/patchcore_raw_scores.npz")
scores = data['scores']
y_true = data['labels']

precision, recall, thresholds = precision_recall_curve(y_true, scores)
f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
best_idx = np.argmax(f1_scores)
best_f1 = f1_scores[best_idx]
print(f"Optimal F1 Score: {best_f1}")

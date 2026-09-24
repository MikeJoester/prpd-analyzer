import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

data = np.load("Results/patchcore/patchcore_raw_scores.npz")
scores = data['scores']
y_true = data['labels']

fpr, tpr, thresholds = roc_curve(y_true, scores)
# Youden's J statistic
J = tpr - fpr
best_idx = np.argmax(J)
best_thresh = thresholds[best_idx]

y_pred = (scores >= best_thresh).astype(int)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"Optimal Threshold (Youden's J): {best_thresh:.4f}")
print("--- CONFUSION MATRIX ---")
print(f"                 Predicted Normal (0)  |  Predicted Anomaly (1)")
print(f"Actual Normal (0) : {tn:4d} (True Neg)      | {fp:4d} (False Pos)")
print(f"Actual Anomaly(1) : {fn:4d} (False Neg)     | {tp:4d} (True Pos)")


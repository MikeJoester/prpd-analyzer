import numpy as np
import pandas as pd

# Load CSV to get labels
df = pd.read_csv('artifacts/ai_data_20260911_011704/ai_raw_dataset_v1.csv')
lab_labels = df[df['group'] == 'Lab PD']['label'].values
field_labels = df[df['group'] == 'Field PD']['label'].values

# Reconstruct the random permutations used in train_efficient_ad.py
np.random.seed(42)

lab_indices = np.random.permutation(len(lab_labels))
lab_val_end = int(0.85 * len(lab_labels))
lab_test_labels = lab_labels[lab_indices[lab_val_end:]]

field_indices = np.random.permutation(len(field_labels))
field_val_end = int(0.85 * len(field_labels))
field_test_labels = field_labels[field_indices[field_val_end:]]

# Combine to match test_dataset
normal_test_labels = np.concatenate([lab_test_labels, field_test_labels])

# Load scores
data = np.load("Results/patchcore/patchcore_raw_scores.npz")
scores = data['scores']
y_true = data['labels']

# We only care about y_true == 0 (Normal) for these specific types
normal_scores = scores[y_true == 0]

if len(normal_scores) != len(normal_test_labels):
    print("Mismatch!", len(normal_scores), len(normal_test_labels))

results = []
for label, score in zip(normal_test_labels, normal_scores):
    results.append({'Label': label, 'Score': score})
    
res_df = pd.DataFrame(results)

# Calculate threshold to show False Positives
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_true, scores)
best_thresh = thresholds[np.argmax(tpr - fpr)]

print(f"Optimal Threshold: {best_thresh:.4f}")
print("-" * 50)
print(f"{'PD Defect Type':<15} | {'Count':<6} | {'Avg Anomaly Score':<20} | {'False Positives':<15}")
print("-" * 50)

for label in res_df['Label'].unique():
    subset = res_df[res_df['Label'] == label]
    avg_score = subset['Score'].mean()
    fps = (subset['Score'] >= best_thresh).sum()
    print(f"{label:<15} | {len(subset):<6} | {avg_score:<20.4f} | {fps}/{len(subset)} ({(fps/len(subset))*100:.1f}%)")


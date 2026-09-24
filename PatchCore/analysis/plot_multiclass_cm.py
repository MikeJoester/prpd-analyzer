import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

# Load CSV to get labels
df = pd.read_csv('artifacts/ai_data_20260911_011704/ai_raw_dataset_v1.csv')
lab_labels = df[df['group'] == 'Lab PD']['label'].values
field_labels = df[df['group'] == 'Field PD']['label'].values

# Reconstruct the random permutations
np.random.seed(42)

lab_indices = np.random.permutation(len(lab_labels))
lab_val_end = int(0.85 * len(lab_labels))
lab_test_labels = lab_labels[lab_indices[lab_val_end:]]

field_indices = np.random.permutation(len(field_labels))
field_val_end = int(0.85 * len(field_labels))
field_test_labels = field_labels[field_indices[field_val_end:]]

normal_test_labels = np.concatenate([lab_test_labels, field_test_labels])

# Load scores
data = np.load("Results/patchcore/patchcore_raw_scores.npz")
scores = data['scores']
y_true = data['labels']

fpr, tpr, thresholds = roc_curve(y_true, scores)
best_thresh = thresholds[np.argmax(tpr - fpr)]
y_pred = (scores >= best_thresh).astype(int)

# Create an array for the actual labels of ALL test samples
# y_true == 1 are 'Field Noise'
all_actual_labels = []
normal_idx = 0
for i in range(len(y_true)):
    if y_true[i] == 1:
        all_actual_labels.append('Field Noise')
    else:
        all_actual_labels.append(normal_test_labels[normal_idx])
        normal_idx += 1

types = ['Particle', 'Void', 'Floating', 'Corona', 'Noise', 'Field Noise']
# We'll map 'Noise' to 'PD Noise' to differentiate from Field Noise
types_display = ['Particle (PD)', 'Void (PD)', 'Floating (PD)', 'Corona (PD)', 'Noise (PD)', 'Field Noise']

cm = np.zeros((len(types), 2), dtype=int)

for actual, pred in zip(all_actual_labels, y_pred):
    row_idx = types.index(actual)
    cm[row_idx, pred] += 1

fig, ax = plt.subplots(figsize=(8, 8))
cax = ax.matshow(cm, cmap='Blues', aspect='auto')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=int(cm[i, j]), va='center', ha='center', size='xx-large',
                color='white' if cm[i, j] > (cm.max()/2) else 'black')

ax.set_xticks([0, 1])
ax.set_xticklabels(['Predicted\nNormal', 'Predicted\nAnomaly'])
ax.xaxis.set_ticks_position('bottom')

ax.set_yticks(range(len(types_display)))
ax.set_yticklabels(types_display)

plt.title(f'Detailed Confusion Matrix\n(Threshold = {best_thresh:.4f})', pad=20, size='x-large')
plt.ylabel('Actual Ground Truth', size='large')
plt.xlabel('PatchCore Prediction', size='large')
plt.tight_layout()
plt.savefig('Results/patchcore/Detailed_Confusion_Matrix.png', dpi=300)
print("Saved to Detailed_Confusion_Matrix.png")

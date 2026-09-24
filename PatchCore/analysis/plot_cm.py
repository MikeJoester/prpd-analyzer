import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve

data = np.load("Results/patchcore/patchcore_raw_scores.npz")
scores = data['scores']
y_true = data['labels']

fpr, tpr, thresholds = roc_curve(y_true, scores)
best_thresh = thresholds[np.argmax(tpr - fpr)]
y_pred = (scores >= best_thresh).astype(int)

cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(cm, cmap='Blues')

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large',
                color='white' if cm[i, j] > (cm.max()/2) else 'black')

ax.set_xticklabels([''] + ['Predicted Normal\n(PD)', 'Predicted Anomaly\n(Noise)'])
ax.set_yticklabels([''] + ['Actual Normal\n(PD)', 'Actual Anomaly\n(Noise)'])
ax.xaxis.set_ticks_position('bottom')

plt.title('PatchCore Confusion Matrix (Threshold = {:.4f})'.format(best_thresh), pad=20, size='x-large')
plt.ylabel('True Label', size='large')
plt.xlabel('Predicted Label', size='large')
plt.tight_layout()
plt.savefig('Results/patchcore/PatchCore_Confusion_Matrix.png', dpi=300)
print("Saved to PatchCore_Confusion_Matrix.png")

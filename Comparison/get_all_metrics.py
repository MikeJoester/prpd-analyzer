import os
import torch
import numpy as np
import glob
from sklearn.metrics import precision_recall_curve
from sklearn.svm import OneClassSVM
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.abspath('.'))
from EfficientAD.train_efficient_ad import PRPDAnomalyDataset, PDN, Autoencoder

def extract_svm_features(dataset_tensors):
    features = []
    for tensor in dataset_tensors:
        max_vals = np.max(tensor, axis=1)
        non_zero_count = np.count_nonzero(tensor, axis=1)
        mean_non_zero = max_vals / (non_zero_count + 1e-15)
        feat = np.concatenate([max_vals, mean_non_zero])
        features.append(feat)
    return np.array(features)

def optimal_pr(y_true, y_scores, name):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    best_idx = np.argmax(f1_scores)
    
    print(f"--- {name} ---")
    print(f"F1-Score:  {f1_scores[best_idx]:.4f}")
    print(f"Precision: {precision[best_idx]:.4f}")
    print(f"Recall:    {recall[best_idx]:.4f}")
    print(f"Threshold: {thresholds[best_idx] if best_idx < len(thresholds) else 'N/A'}\n")

ai_data_root = sorted(glob.glob("artifacts/ai_data_*"))[-1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = PRPDAnomalyDataset(ai_data_root, mode='train')
test_dataset = PRPDAnomalyDataset(ai_data_root, mode='test')

noise_path = os.path.join(ai_data_root, "raw_tensors_v1", "Field Noise.npz")
test_defect = list(np.load(noise_path)['raw'])
test_good = list(test_dataset.data)
all_test_data = test_good + test_defect
y_true = [0] * len(test_good) + [1] * len(test_defect)

# SVM
print("Evaluating SVM...")
train_features = extract_svm_features(train_dataset.data)
test_features = extract_svm_features(all_test_data)
svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
svm.fit(train_features)
svm_scores = -svm.decision_function(test_features)
optimal_pr(y_true, svm_scores, "SVM Baseline")

# EfficientAD
print("Evaluating EfficientAD...")
teacher = PDN(out_channels=384).to(device)
student = PDN(out_channels=768).to(device)
autoencoder = Autoencoder(out_channels=384).to(device)

teacher.load_state_dict(torch.load("models/efficientad_teacher.pth"))
student.load_state_dict(torch.load("models/efficientad_student.pth"))
autoencoder.load_state_dict(torch.load("models/efficientad_autoencoder.pth"))

teacher.eval()
student.eval()
autoencoder.eval()

ead_scores = []
with torch.no_grad():
    for img in all_test_data:
        img_t = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) / 255.0
        teacher_output = teacher(img_t)
        student_output = student(img_t)
        ae_output = autoencoder(img_t)
        
        st_dist = torch.mean((teacher_output - student_output[:, :384]) ** 2, dim=1)
        ae_dist = torch.mean((ae_output - student_output[:, 384:]) ** 2, dim=1)
        anomaly_map = st_dist + ae_dist
        ead_scores.append(anomaly_map.max().item())

optimal_pr(y_true, ead_scores, "EfficientAD")

# PatchCore
data = np.load("Results/patchcore/patchcore_raw_scores.npz")
optimal_pr(data['labels'], data['scores'], "PatchCore (ResNet-50)")


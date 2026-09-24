import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import glob
from pathlib import Path
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from sklearn.svm import OneClassSVM
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from EfficientAD.train_efficient_ad import PRPDAnomalyDataset, PDN, Autoencoder

def extract_svm_features(dataset_tensors):
    features = []
    for tensor in dataset_tensors:
        # Expected shape: (128, 3600), SVM uses max and max/(count+1e-15)
        # Using the whole width or sliced? SVM used the whole width.
        # Actually in evaluate_svm.py:
        max_vals = np.max(tensor, axis=1)
        non_zero_count = np.count_nonzero(tensor, axis=1)
        mean_non_zero = max_vals / (non_zero_count + 1e-15)
        feat = np.concatenate([max_vals, mean_non_zero])
        features.append(feat)
    return np.array(features)

def optimal_f1(y_true, y_scores):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    f1_scores = 2 * recall * precision / (recall + precision + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    return best_f1, best_thresh

def calculate_all_metrics():
    ai_data_root = sorted(glob.glob("artifacts/ai_data_*"))[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    print("Loading datasets...")
    # Train normal (for SVM)
    train_dataset = PRPDAnomalyDataset(ai_data_root, mode='train')
    train_tensors = [train_dataset.data[i] for i in range(len(train_dataset))]
    
    # Test normal
    test_dataset = PRPDAnomalyDataset(ai_data_root, mode='test')
    test_normal = [test_dataset.data[i] for i in range(len(test_dataset))]
    y_true_normal = [0] * len(test_normal)
    
    # Test defective (noise)
    noise_path = os.path.join(ai_data_root, "raw_tensors_v1", "Field Noise.npz")
    test_defect = list(np.load(noise_path)['raw'])
    y_true_defect = [1] * len(test_defect)
    
    all_test_tensors = test_normal + test_defect
    y_true = np.array(y_true_normal + y_true_defect)
    
    results = []

    # ==========================================
    # Model 1: SVM Baseline
    # ==========================================
    print("Evaluating SVM...")
    train_feats = extract_svm_features(train_tensors)
    test_feats = extract_svm_features(all_test_tensors)
    svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)
    svm.fit(train_feats)
    # SVM outputs +1 for inlier, -1 for outlier. Score samples gives distance to separating hyperplane.
    # Higher score = more normal. We invert it so higher = anomaly.
    svm_scores = -svm.score_samples(test_feats) 
    
    svm_auroc = roc_auc_score(y_true, svm_scores)
    svm_f1, svm_thresh = optimal_f1(y_true, svm_scores)
    results.append({
        'Model': 'SVM Baseline',
        'Image AUROC': svm_auroc,
        'Pixel AUROC': 'N/A',
        'Optimal F1': svm_f1
    })

    # ==========================================
    # Model 2: EfficientAD
    # ==========================================
    print("Evaluating EfficientAD...")
    teacher = PDN(out_channels=384).to(device)
    student = PDN(out_channels=768).to(device)
    autoencoder = Autoencoder(out_channels=384).to(device)
    
    teacher.eval()
    student.load_state_dict(torch.load("artifacts/anomaly_runs/efficient_ad/student.pt", map_location=device))
    autoencoder.load_state_dict(torch.load("artifacts/anomaly_runs/efficient_ad/autoencoder.pt", map_location=device))
    student.eval()
    autoencoder.eval()
    
    efficientad_scores = []
    with torch.no_grad():
        # Evaluate 1 by 1
        for tensor in all_test_tensors:
            # Prepare tensor [1, 1, 128, 3600]
            batch = torch.tensor(tensor, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            # EfficientAD was trained on sliding windows or the whole thing? 
            # The training used random crops of 128x256. Inference uses the whole width.
            teacher_out = teacher(batch)
            student_out = student(batch)
            student_T = student_out[:, :384, :, :]
            student_AE = student_out[:, 384:, :, :]
            ae_out = autoencoder(batch)
            
            ae_out = nn.functional.interpolate(ae_out, size=teacher_out.shape[2:], mode='bilinear')
            student_AE = nn.functional.interpolate(student_AE, size=teacher_out.shape[2:], mode='bilinear')
            
            local_map = torch.mean((teacher_out - student_T) ** 2, dim=1, keepdim=True)
            global_map = torch.mean((ae_out - student_AE) ** 2, dim=1, keepdim=True)
            combined_map = 0.5 * local_map + 0.5 * global_map
            image_score = torch.max(combined_map).item()
            efficientad_scores.append(image_score)
            
    efficientad_scores = np.array(efficientad_scores)
    ead_auroc = roc_auc_score(y_true, efficientad_scores)
    ead_f1, ead_thresh = optimal_f1(y_true, efficientad_scores)
    # For pixel AUROC, we don't have exact pixel masks for noise (we'd need an all-1 mask vs all-0 mask)
    # So we can just put N/A or compute it. Let's put N/A.
    results.append({
        'Model': 'EfficientAD',
        'Image AUROC': ead_auroc,
        'Pixel AUROC': 'N/A', # Skipping as dummy masks were used
        'Optimal F1': ead_f1
    })
    print(f"EfficientAD Optimal Threshold: {ead_thresh:.6f}")

    # ==========================================
    # Model 3: PatchCore
    # ==========================================
    print("Loading PatchCore results...")
    if os.path.exists("Results/patchcore/patchcore_raw_scores.npz"):
        pc_data = np.load("Results/patchcore/patchcore_raw_scores.npz")
        pc_scores = pc_data['scores']
        pc_labels = pc_data['labels']
        pc_auroc = roc_auc_score(pc_labels, pc_scores)
        pc_f1, pc_thresh = optimal_f1(pc_labels, pc_scores)
        
        # We know PatchCore pixel AUROC from logs is 0.640
        results.append({
            'Model': 'PatchCore (ResNet-50)',
            'Image AUROC': pc_auroc,
            'Pixel AUROC': 0.640,
            'Optimal F1': pc_f1
        })
    else:
        print("PatchCore scores not found. Waiting for PatchCore to finish...")

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("Results/Model_Comparison_Metrics.csv", index=False)
    print("\nResults saved to Model_Comparison_Metrics.csv:")
    print(df.to_string())

if __name__ == "__main__":
    calculate_all_metrics()

import numpy as np
import glob
from pathlib import Path
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from EfficientAD.train_efficient_ad import PRPDAnomalyDataset

def extract_features(data):
    # data shape: (N, 128, 3600)
    # We use the exact formula from the original Training_SVM.py:
    # Feature 1: max_each_columns
    # Feature 2: max_devide_nb_zeros
    max_each_columns = np.max(data, axis=2)
    nonzero_counts = np.count_nonzero(data, axis=2)
    max_devide_nb_zeros = max_each_columns / (nonzero_counts + 1e-15)
    
    features = np.concatenate([max_each_columns, max_devide_nb_zeros], axis=1) # (N, 256)
    return features

def evaluate_svm_baseline(ai_data_root):
    print("Loading Datasets for Baseline One-Class SVM...")
    # Load raw data using the same deterministic split as EfficientAD
    train_dataset = PRPDAnomalyDataset(ai_data_root, mode='train')
    test_dataset = PRPDAnomalyDataset(ai_data_root, mode='test')
    
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("Data missing.")
        return
        
    print("Extracting 128x2 (256 flattened) features...")
    # The dataset returns (1, 128, 256) crops, but for the statistical baseline we want the full 3600 cycles if possible.
    # To keep it completely fair, we'll extract features exactly from the dataset class arrays.
    # train_dataset.data is (1804, 128, 3600)
    X_train = extract_features(train_dataset.data)
    X_test = extract_features(test_dataset.data)
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training One-Class SVM (RBF Kernel)...")
    # nu represents the proportion of outliers we expect in the training data (we assume 1% anomalies)
    svm = OneClassSVM(kernel='rbf', nu=0.01)
    svm.fit(X_train_scaled)
    
    print("Scoring Test Set...")
    # score_samples returns the raw scoring function. We invert it so higher = more anomalous
    train_scores = -svm.score_samples(X_train_scaled)
    test_scores = -svm.score_samples(X_test_scaled)
    
    print("-" * 50)
    print("BASELINE MODEL: ONE-CLASS SVM RESULTS")
    print("-" * 50)
    print(f"Train Average Anomaly Score: {np.mean(train_scores):.4f}")
    print(f"Test Average Anomaly Score : {np.mean(test_scores):.4f}")
    print(f"Test Score Variance        : {np.var(test_scores):.4f}")
    print("-" * 50)

if __name__ == "__main__":
    candidates = sorted(glob.glob("artifacts/ai_data_*"))
    if candidates:
        evaluate_svm_baseline(candidates[-1])
    else:
        print("No dataset found.")

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import glob
from pathlib import Path

# Import architectures and dataset from the training script
from EfficientAD.train_efficient_ad import PRPDAnomalyDataset, PDN, Autoencoder

def evaluate_efficient_ad(ai_data_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading Test Dataset (30% Unseen Field PD Dates)...")
    test_dataset = PRPDAnomalyDataset(ai_data_root, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    if len(test_dataset) == 0:
        print("No test data found.")
        return
        
    # 1. Initialize Networks
    teacher = PDN(out_channels=384).to(device)
    teacher.eval() # Randomly initialized frozen teacher
    
    student = PDN(out_channels=768).to(device)
    autoencoder = Autoencoder(out_channels=384).to(device)
    
    # 2. Load Trained Weights
    student_path = "artifacts/anomaly_runs/efficient_ad/student.pt"
    ae_path = "artifacts/anomaly_runs/efficient_ad/autoencoder.pt"
    
    if os.path.exists(student_path) and os.path.exists(ae_path):
        student.load_state_dict(torch.load(student_path, map_location=device))
        autoencoder.load_state_dict(torch.load(ae_path, map_location=device))
        print("Successfully loaded trained Student and Autoencoder weights.")
    else:
        print("Error: Trained weights not found. Please train first.")
        return
        
    student.eval()
    autoencoder.eval()
    
    scores = []
    
    print("Starting Inference on Test Set...")
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Forward passes
            teacher_out = teacher(batch)
            student_out = student(batch)
            student_T = student_out[:, :384, :, :]
            student_AE = student_out[:, 384:, :, :]
            ae_out = autoencoder(batch)
            
            # Match spatial dimensions
            ae_out = nn.functional.interpolate(ae_out, size=teacher_out.shape[2:], mode='bilinear')
            student_AE = nn.functional.interpolate(student_AE, size=teacher_out.shape[2:], mode='bilinear')
            
            # 1. Local Anomaly Map (Structural Anomalies)
            local_map = torch.mean((teacher_out - student_T) ** 2, dim=1, keepdim=True)
            
            # 2. Global Anomaly Map (Logical Anomalies)
            global_map = torch.mean((ae_out - student_AE) ** 2, dim=1, keepdim=True)
            
            # 3. Final Combined Anomaly Map
            combined_map = 0.5 * local_map + 0.5 * global_map
            
            # 4. Image-level score (Maximum pixel value)
            image_score = torch.max(combined_map).item()
            scores.append(image_score)
            
    avg_score = np.mean(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    
    print("-" * 50)
    print("TEST SET EVALUATION RESULTS (Unseen Field PD)")
    print("-" * 50)
    print(f"Total Files Evaluated : {len(scores)}")
    print(f"Average Anomaly Score : {avg_score:.6f}")
    print(f"Max Anomaly Score     : {max_score:.6f} (Most anomalous file)")
    print(f"Min Anomaly Score     : {min_score:.6f} (Most normal file)")
    print("-" * 50)
    print("Note: EfficientAD successfully scanned the test set. Scores represent the deviation from the normal Field+Lab distribution.")

if __name__ == "__main__":
    candidates = sorted(glob.glob("artifacts/ai_data_*"))
    if candidates:
        evaluate_efficient_ad(candidates[-1])
    else:
        print("No dataset found.")

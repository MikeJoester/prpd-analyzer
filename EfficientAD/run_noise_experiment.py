import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import glob
from pathlib import Path

from EfficientAD.train_efficient_ad import PDN, Autoencoder

class NoiseAnomalyDataset(Dataset):
    def __init__(self, ai_data_root, mode='train'):
        self.ai_data_root = Path(ai_data_root)
        self.mode = mode
        
        # Load Lab Noise
        lab_noise_path = self.ai_data_root / "raw_tensors_v1" / "Lab Noise.npz"
        if lab_noise_path.exists():
            lab_noise = np.load(lab_noise_path)['raw']
        else:
            lab_noise = np.zeros((0, 128, 3600), dtype=np.uint8)
            
        # Load Field Noise
        field_noise_path = self.ai_data_root / "raw_tensors_v1" / "Field Noise.npz"
        if field_noise_path.exists():
            field_noise = np.load(field_noise_path)['raw']
        else:
            field_noise = np.zeros((0, 128, 3600), dtype=np.uint8)
            
        # Deterministic simple split (70/30)
        np.random.seed(42)
        field_indices = np.random.permutation(len(field_noise))
        split_idx = int(0.7 * len(field_noise))
        
        if mode == 'train':
            # Training: Lab Noise + 70% Field Noise
            train_field_noise = field_noise[field_indices[:split_idx]]
            self.data = np.concatenate([lab_noise, train_field_noise], axis=0)
        else:
            # Test: 30% Field Noise
            self.data = field_noise[field_indices[split_idx:]]
            
        print(f"Loaded {len(self.data)} samples for {mode} mode.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if sample.shape[1] > 256:
            start = torch.randint(0, sample.shape[1] - 256 + 1, (1,)).item()
            sample = sample[:, start:start+256]
            
        tensor = torch.from_numpy(sample).float() / 255.0
        return tensor.unsqueeze(0)

def run_noise_experiment(ai_data_root, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = NoiseAnomalyDataset(ai_data_root, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    teacher = PDN(out_channels=384).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
        
    student = PDN(out_channels=768).to(device)
    autoencoder = Autoencoder(out_channels=384).to(device)
    
    optimizer = optim.Adam(list(student.parameters()) + list(autoencoder.parameters()), lr=1e-4)
    
    print("Starting EfficientAD Training on NOISE...")
    
    for epoch in range(epochs):
        student.train()
        autoencoder.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            with torch.no_grad():
                teacher_out = teacher(batch)
                
            student_out = student(batch)
            student_T = student_out[:, :384, :, :]
            student_AE = student_out[:, 384:, :, :]
            
            ae_out = autoencoder(batch)
            ae_out = nn.functional.interpolate(ae_out, size=teacher_out.shape[2:], mode='bilinear')
            student_AE = nn.functional.interpolate(student_AE, size=teacher_out.shape[2:], mode='bilinear')
            
            diff_st = (teacher_out - student_T) ** 2
            b, c, h, w = diff_st.shape
            d_st = diff_st.view(b, c, -1)
            q = torch.quantile(d_st, 0.999, dim=2, keepdim=True)
            loss_st = diff_st[diff_st >= q.unsqueeze(-1)].mean()
            
            loss_ae = torch.nn.functional.mse_loss(ae_out, teacher_out)
            loss_stae = torch.nn.functional.mse_loss(student_AE, ae_out.detach())
            
            loss = loss_st + loss_ae + loss_stae
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
        
    print("Training Complete! Starting Evaluation on 30% Unseen Field Noise...")
    
    test_dataset = NoiseAnomalyDataset(ai_data_root, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    student.eval()
    autoencoder.eval()
    
    scores = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
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
            
            scores.append(torch.max(combined_map).item())
            
    print("-" * 50)
    print("NOISE MODEL: TEST SET EVALUATION RESULTS (Unseen Field Noise)")
    print("-" * 50)
    print(f"Total Files Evaluated : {len(scores)}")
    print(f"Average Anomaly Score : {np.mean(scores):.6f}")
    print(f"Max Anomaly Score     : {np.max(scores):.6f} (Most anomalous noise file)")
    print(f"Min Anomaly Score     : {np.min(scores):.6f} (Most normal noise file)")
    print(f"Test Score Variance   : {np.var(scores):.6e}")
    print("-" * 50)

if __name__ == "__main__":
    candidates = sorted(glob.glob("artifacts/ai_data_*"))
    if candidates:
        run_noise_experiment(candidates[-1])
    else:
        print("No dataset found.")

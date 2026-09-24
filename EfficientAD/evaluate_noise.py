import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import glob
from pathlib import Path

# Import architectures
from EfficientAD.train_efficient_ad import PDN, Autoencoder

class NoiseDataset(Dataset):
    def __init__(self, ai_data_root, noise_type='Field Noise'):
        self.ai_data_root = Path(ai_data_root)
        noise_path = self.ai_data_root / "raw_tensors_v1" / f"{noise_type}.npz"
        if noise_path.exists():
            self.data = np.load(noise_path)['raw']
            print(f"Loaded {len(self.data)} samples of {noise_type}.")
        else:
            self.data = np.zeros((0, 128, 3600), dtype=np.uint8)
            print(f"No {noise_type} data found.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if sample.shape[1] > 256:
            start = torch.randint(0, sample.shape[1] - 256 + 1, (1,)).item()
            sample = sample[:, start:start+256]
        tensor = torch.from_numpy(sample).float() / 255.0
        return tensor.unsqueeze(0)

def evaluate_noise(ai_data_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Networks & Load Weights
    teacher = PDN(out_channels=384).to(device)
    student = PDN(out_channels=768).to(device)
    autoencoder = Autoencoder(out_channels=384).to(device)
    
    student_path = "artifacts/anomaly_runs/efficient_ad/student.pt"
    ae_path = "artifacts/anomaly_runs/efficient_ad/autoencoder.pt"
    
    if os.path.exists(student_path) and os.path.exists(ae_path):
        student.load_state_dict(torch.load(student_path, map_location=device))
        autoencoder.load_state_dict(torch.load(ae_path, map_location=device))
    else:
        print("Trained weights not found.")
        return
        
    teacher.eval()
    student.eval()
    autoencoder.eval()
    
    # Evaluate Lab Noise and Field Noise
    for noise_type in ['Lab Noise', 'Field Noise']:
        dataset = NoiseDataset(ai_data_root, noise_type)
        if len(dataset) == 0:
            continue
            
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        scores = []
        
        with torch.no_grad():
            for batch in loader:
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
        print(f"RESULTS FOR: {noise_type}")
        print("-" * 50)
        print(f"Total Evaluated       : {len(scores)}")
        print(f"Average Anomaly Score : {np.mean(scores):.6f}")
        print(f"Max Score             : {np.max(scores):.6f}")
        print(f"Min Score             : {np.min(scores):.6f}")

if __name__ == "__main__":
    candidates = sorted(glob.glob("artifacts/ai_data_*"))
    if candidates:
        evaluate_noise(candidates[-1])

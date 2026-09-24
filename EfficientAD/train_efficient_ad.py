import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import glob
from pathlib import Path

# --- 1. Dataset ---
class PRPDAnomalyDataset(Dataset):
    def __init__(self, ai_data_root, mode='train'):
        self.ai_data_root = Path(ai_data_root)
        self.mode = mode
        
        # Load Lab PD
        lab_pd_path = self.ai_data_root / "raw_tensors_v1" / "Lab PD.npz"
        if lab_pd_path.exists():
            lab_pd = np.load(lab_pd_path)['raw']
        else:
            lab_pd = np.zeros((0, 128, 3600), dtype=np.uint8)
            
        # Load Field PD
        field_pd_path = self.ai_data_root / "raw_tensors_v1" / "Field PD.npz"
        if field_pd_path.exists():
            field_pd = np.load(field_pd_path)['raw']
        else:
            field_pd = np.zeros((0, 128, 3600), dtype=np.uint8)
            
        # Deterministic split
        np.random.seed(42)
        
        # Split Field PD (70/15/15)
        field_indices = np.random.permutation(len(field_pd))
        field_train_end = int(0.7 * len(field_pd))
        field_val_end = int(0.85 * len(field_pd))
        
        field_train = field_pd[field_indices[:field_train_end]]
        field_val = field_pd[field_indices[field_train_end:field_val_end]]
        field_test = field_pd[field_indices[field_val_end:]]
        
        if mode == 'train':
            # 100% Lab PD + 70% Field PD
            self.data = np.concatenate([lab_pd, field_train], axis=0)
        elif mode == 'val':
            # 15% Field PD
            self.data = field_val
        else: # test
            # 15% Field PD
            self.data = field_test
            
        print(f"Loaded {len(self.data)} samples for {mode} mode.")

    def __len__(self):


        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # Crop to (128, 256) for speed
        if sample.shape[1] > 256:
            start = torch.randint(0, sample.shape[1] - 256 + 1, (1,)).item()
            sample = sample[:, start:start+256]
            
        # Normalize to float32
        tensor = torch.from_numpy(sample).float() / 255.0
        return tensor.unsqueeze(0) # (1, 128, 256)

# --- 2. Architecture ---
class PDN(nn.Module):
    def __init__(self, out_channels=384):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=4, stride=1, padding=3)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x

class Autoencoder(nn.Module):
    def __init__(self, out_channels=384):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        z = self.encoder(x)
        # resize for decoder using upsample
        z = nn.functional.interpolate(z, size=(4, 8), mode='bilinear')
        return self.decoder(z)

# --- 3. Training Loop ---
def train_efficient_ad(ai_data_root, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dataset = PRPDAnomalyDataset(ai_data_root, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    
    # Init networks
    teacher = PDN(out_channels=384).to(device)
    teacher.eval() # Teacher is frozen
    for param in teacher.parameters():
        param.requires_grad = False
        
    student = PDN(out_channels=768).to(device) # Student outputs 768 (384 for Teacher, 384 for AE)
    autoencoder = Autoencoder(out_channels=384).to(device)
    
    optimizer = optim.Adam(list(student.parameters()) + list(autoencoder.parameters()), lr=1e-4)
    
    print("Starting EfficientAD Training...")
    
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
            # Resize AE output to match teacher output shape
            ae_out = nn.functional.interpolate(ae_out, size=teacher_out.shape[2:], mode='bilinear')
            student_AE = nn.functional.interpolate(student_AE, size=teacher_out.shape[2:], mode='bilinear')
            
            # Loss ST (Hard Feature Loss)
            diff_st = (teacher_out - student_T) ** 2
            # Mine 99.9th percentile
            b, c, h, w = diff_st.shape
            d_st = diff_st.view(b, c, -1)
            q = torch.quantile(d_st, 0.999, dim=2, keepdim=True)
            loss_st = diff_st[diff_st >= q.unsqueeze(-1)].mean()
            
            # Loss AE
            loss_ae = torch.nn.functional.mse_loss(ae_out, teacher_out)
            
            # Loss STAE
            loss_stae = torch.nn.functional.mse_loss(student_AE, ae_out.detach())
            
            loss = loss_st + loss_ae + loss_stae
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")
        
    # Create artifact folder if it doesn't exist
    os.makedirs("artifacts/anomaly_runs/efficient_ad", exist_ok=True)
    torch.save(student.state_dict(), "artifacts/anomaly_runs/efficient_ad/student.pt")
    torch.save(autoencoder.state_dict(), "artifacts/anomaly_runs/efficient_ad/autoencoder.pt")
    print("Training Complete! Models saved.")

if __name__ == "__main__":
    candidates = sorted(glob.glob("artifacts/ai_data_*"))
    if candidates:
        train_efficient_ad(candidates[-1])
    else:
        print("No dataset found.")

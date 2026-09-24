import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from Diffusion.prpd_diffusion.models.ldm_wrapper import LDMAutoencoderWrapper

class PRPDTensorDataset(Dataset):
    def __init__(self, ai_data_root, groups):
        self.ai_data_root = Path(ai_data_root)
        self.samples = []
        
        for group in groups:
            npz_path = self.ai_data_root / "raw_tensors_v1" / f"{group}.npz"
            if npz_path.exists():
                data = np.load(npz_path)['raw'] # shape: (N, 128, 3600) uint8
                self.samples.append(data)
                print(f"Loaded {len(data)} samples from {group}")
            else:
                print(f"Warning: {npz_path} not found.")
                
        if len(self.samples) > 0:
            self.all_data = np.concatenate(self.samples, axis=0)
        else:
            self.all_data = np.zeros((0, 128, 3600), dtype=np.uint8)
            
    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        # The data is uint8 [0, 255]. Convert to float32 [-1, 1]
        tensor = torch.from_numpy(self.all_data[idx]).float()
        tensor = (tensor / 127.5) - 1.0
        
        # Add channel dimension: (1, 128, 3600)
        tensor = tensor.unsqueeze(0)
        
        # For training, crop to (128, 256) to fit in memory
        if tensor.shape[2] > 256:
            start = torch.randint(0, tensor.shape[2] - 256 + 1, (1,)).item()
            tensor = tensor[:, :, start:start+256]
            
        return tensor

def train_autoencoder(ai_data_root, epochs=10, batch_size=4, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Datasets as per ldm_plan.md
    groups = ["Lab PD", "Field PD", "Lab Noise", "Field Noise"]
    dataset = PRPDTensorDataset(ai_data_root, groups)
    
    if len(dataset) == 0:
        print("No data found! Check ai_data_root.")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 2. Asymmetrical Autoencoder Model
    model = LDMAutoencoderWrapper().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting Autoencoder Stage 1 Training on {len(dataset)} files...")
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            recon, posterior = model(batch)
            
            # Simple L2 reconstruction loss
            recon_loss = torch.nn.functional.mse_loss(recon, batch)
            
            # KL divergence loss against standard normal
            prior = torch.distributions.Normal(torch.zeros_like(posterior.mean), torch.ones_like(posterior.stddev))
            kl_loss = torch.distributions.kl.kl_divergence(posterior, prior).mean()
            
            loss = recon_loss + 0.00001 * kl_loss
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
        
    os.makedirs("artifacts/ldm", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/ldm/autoencoder_best.pt")
    print("Stage 1: Autoencoder training complete and saved to artifacts/ldm/autoencoder_best.pt")

if __name__ == "__main__":
    import glob
    candidates = sorted(glob.glob("artifacts/ai_data_*"))
    if not candidates:
        print("No ai_data folders found. Run scripts/build_ai_dataset.py first.")
    else:
        latest = candidates[-1]
        print(f"Using latest dataset: {latest}")
        train_autoencoder(latest)

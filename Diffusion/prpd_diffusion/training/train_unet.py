import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from Diffusion.prpd_diffusion.models.ldm_wrapper import LDMAutoencoderWrapper, LDMUNetWrapper

class PRPDDiffusionDataset(Dataset):
    def __init__(self, ai_data_root):
        self.ai_data_root = Path(ai_data_root)
        
        # Load Clean Pool
        clean_path = self.ai_data_root / "raw_tensors_v1" / "Lab PD.npz"
        if clean_path.exists():
            self.clean_data = np.load(clean_path)['raw']
        else:
            self.clean_data = np.zeros((0, 128, 3600), dtype=np.uint8)
            
        # Load Noise Pool
        noise_samples = []
        for group in ["Lab Noise", "Field Noise"]:
            n_path = self.ai_data_root / "raw_tensors_v1" / f"{group}.npz"
            if n_path.exists():
                noise_samples.append(np.load(n_path)['raw'])
                
        if noise_samples:
            self.noise_data = np.concatenate(noise_samples, axis=0)
        else:
            self.noise_data = np.zeros((0, 128, 3600), dtype=np.uint8)

        print(f"Diffusion Dataset: {len(self.clean_data)} Clean samples, {len(self.noise_data)} Noise samples.")
            
    def __len__(self):
        return len(self.clean_data)

    def __getitem__(self, idx):
        clean = self.clean_data[idx]
        
        # Randomly pick a noise sample to mix
        if len(self.noise_data) > 0:
            noise_idx = torch.randint(0, len(self.noise_data), (1,)).item()
            noise = self.noise_data[noise_idx]
            
            # Additive mix and clamp to uint8 max
            noisy = np.clip(clean.astype(np.uint16) + noise.astype(np.uint16), 0, 255).astype(np.uint8)
        else:
            noisy = clean
            
        # Convert to float32 [-1, 1]
        clean_t = (torch.from_numpy(clean).float() / 127.5) - 1.0
        noisy_t = (torch.from_numpy(noisy).float() / 127.5) - 1.0
        
        clean_t = clean_t.unsqueeze(0)
        noisy_t = noisy_t.unsqueeze(0)
        
        # Crop to (128, 256) for training
        if clean_t.shape[2] > 256:
            start = torch.randint(0, clean_t.shape[2] - 256 + 1, (1,)).item()
            clean_t = clean_t[:, :, start:start+256]
            noisy_t = noisy_t[:, :, start:start+256]
            
        return clean_t, noisy_t


def get_alphas_cumprod(timesteps=1000):
    # Linear beta schedule
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas_cumprod


def train_unet(ai_data_root, epochs=50, batch_size=4, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = PRPDDiffusionDataset(ai_data_root)
    if len(dataset) == 0:
        print("No clean data found!")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 1. Load Frozen Autoencoder
    autoencoder = LDMAutoencoderWrapper().to(device)
    ae_path = "artifacts/ldm/autoencoder_best.pt"
    if os.path.exists(ae_path):
        autoencoder.load_state_dict(torch.load(ae_path))
        print("Loaded Autoencoder weights.")
    autoencoder.eval()
    for param in autoencoder.parameters():
        param.requires_grad = False
        
    # 2. Init UNet (in_channels is automatically out_channels * 2 in 1D wrapper)
    unet = LDMUNetWrapper(out_channels=4).to(device)
    optimizer = optim.Adam(unet.parameters(), lr=lr)
    
    alphas_cumprod = get_alphas_cumprod(1000).to(device)
    
    print("Starting UNet Stage 2 Training...")
    
    unet.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for clean_t, noisy_t in dataloader:
            clean_t = clean_t.to(device)
            noisy_t = noisy_t.to(device)
            
            optimizer.zero_grad()
            
            # Encode both to latent space
            with torch.no_grad():
                Z_0 = autoencoder.encode(clean_t)
                Z_cond = autoencoder.encode(noisy_t)
            
            # Sample random timesteps
            t = torch.randint(0, 1000, (Z_0.shape[0],), device=device).long()
            
            # Add noise to Z_0
            noise = torch.randn_like(Z_0)
            a_t = alphas_cumprod[t].view(-1, 1, 1, 1)
            Z_t = torch.sqrt(a_t) * Z_0 + torch.sqrt(1 - a_t) * noise
            
            # Concatenate Z_t and conditioning
            unet_input = torch.cat([Z_t, Z_cond], dim=1)
            
            # Predict noise
            predicted_noise = unet(unet_input, t.float())
            
            # MSE Loss
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
        
    torch.save(unet.state_dict(), "artifacts/ldm/ldm_unet_best.pt")
    print("Stage 2: UNet training complete and saved to artifacts/ldm/ldm_unet_best.pt")

if __name__ == "__main__":
    import glob
    candidates = sorted(glob.glob("artifacts/ai_data_*"))
    if not candidates:
        print("No ai_data folders found.")
    else:
        latest = candidates[-1]
        print(f"Using latest dataset: {latest}")
        train_unet(latest)

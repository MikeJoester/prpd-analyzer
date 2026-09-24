import os
import torch
import numpy as np
import glob
from pathlib import Path
from Diffusion.prpd_diffusion.models.ldm_wrapper import LDMAutoencoderWrapper, LDMUNetWrapper

def ddim_sample(unet, autoencoder, noisy_img, device, steps=50):
    """Faster sampling using DDIM logic."""
    with torch.no_grad():
        Z_cond = autoencoder.encode(noisy_img)
        Z_t = torch.randn_like(Z_cond)
        
        # Schedules
        timesteps = 1000
        betas = torch.linspace(0.0001, 0.02, timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # DDIM indices
        c = timesteps // steps
        time_steps = list(range(0, timesteps, c))
        time_steps.reverse()
        
        for i, step in enumerate(time_steps):
            t = torch.full((Z_t.shape[0],), step, device=device, dtype=torch.long)
            unet_input = torch.cat([Z_t, Z_cond], dim=1)
            predicted_noise = unet(unet_input, t.float())
            
            alpha_cumprod_t = alphas_cumprod[step]
            alpha_cumprod_prev = alphas_cumprod[time_steps[i+1]] if i < len(time_steps)-1 else torch.tensor(1.0).to(device)
            
            # Predict x0
            x0_pred = (Z_t - torch.sqrt(1 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_cumprod_prev) * predicted_noise
            
            Z_t = torch.sqrt(alpha_cumprod_prev) * x0_pred + dir_xt
            
        denoised_img = autoencoder.decode(Z_t)
        return denoised_img

def run_evaluation(ai_data_root):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Models
    autoencoder = LDMAutoencoderWrapper().to(device)
    autoencoder.load_state_dict(torch.load("artifacts/ldm/autoencoder_best.pt", map_location=device))
    autoencoder.eval()
    
    unet = LDMUNetWrapper(out_channels=4).to(device)
    unet.load_state_dict(torch.load("artifacts/ldm/ldm_unet_best.pt", map_location=device))
    unet.eval()
    
    print("Models loaded successfully.")
    
    # 2. Load Data
    ai_data_root = Path(ai_data_root)
    clean_data = np.load(ai_data_root / "raw_tensors_v1" / "Lab PD.npz")['raw']
    noise_data = np.load(ai_data_root / "raw_tensors_v1" / "Field Noise.npz")['raw']
    
    np.random.seed(42)
    num_samples = 50
    clean_indices = np.random.choice(len(clean_data), num_samples, replace=False)
    noise_indices = np.random.choice(len(noise_data), num_samples, replace=False)
    
    baseline_maes = []
    model_maes = []
    
    print(f"Running Paired Synthetic Evaluation on {num_samples} samples...")
    for i in range(num_samples):
        c_idx = clean_indices[i]
        n_idx = noise_indices[i]
        
        clean = clean_data[c_idx]
        noise = noise_data[n_idx]
        
        # Mix
        noisy = np.clip(clean.astype(np.uint16) + noise.astype(np.uint16), 0, 255).astype(np.uint8)
        
        # Normalize
        clean_t = (torch.from_numpy(clean).float() / 127.5) - 1.0
        noisy_t = (torch.from_numpy(noisy).float() / 127.5) - 1.0
        
        # Crop to (128, 256)
        clean_t = clean_t[:, :256].unsqueeze(0).unsqueeze(0).to(device)
        noisy_t = noisy_t[:, :256].unsqueeze(0).unsqueeze(0).to(device)
        
        # Sample
        denoised_t = ddim_sample(unet, autoencoder, noisy_t, device, steps=20)
        
        # Calculate MAE
        baseline_mae = torch.nn.functional.l1_loss(noisy_t, clean_t).item()
        model_mae = torch.nn.functional.l1_loss(denoised_t, clean_t).item()
        
        baseline_maes.append(baseline_mae)
        model_maes.append(model_mae)
        
    avg_base = np.mean(baseline_maes)
    avg_model = np.mean(model_maes)
    
    print("-" * 50)
    print("PERFORMANCE REPORT: Asymmetrical 2D LDM")
    print("-" * 50)
    print(f"Baseline MAE (Noisy vs Clean): {avg_base:.4f}")
    print(f"LDM Denoised MAE (Output vs Clean): {avg_model:.4f}")
    
    if avg_model < avg_base:
        print("✅ SUCCESS: The LDM successfully reduced noise! (Model MAE < Baseline MAE)")
    else:
        print("❌ FAILURE: The LDM made it worse. (Model MAE >= Baseline MAE)")

if __name__ == "__main__":
    candidates = sorted(glob.glob("artifacts/ai_data_*"))
    latest = candidates[-1]
    run_evaluation(latest)

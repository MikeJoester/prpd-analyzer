# Plan: Applying Latent Diffusion Models (LDM) to Phase-Resolved Partial Discharge (PRPD) Data

## 1. Scope of the Latent Diffusion Paper
The paper *"High-Resolution Image Synthesis with Latent Diffusion Models"* (Rombach et al.) introduces a two-stage approach to generative diffusion:
1. **Perceptual Compression (Autoencoder):** An autoencoder (KL-VAE or VQ-VAE) is trained to compress high-dimensional pixel space into a compact, lower-dimensional latent space. This step removes imperceptible high-frequency details (which is excellent for sensor noise).
2. **Latent Diffusion:** A UNet-based diffusion model is trained strictly within this latent space. Because the space is much smaller, training and sampling are computationally efficient. The model uses Cross-Attention mechanisms to condition the generation process on classes, text, or other images.

## 2. Applicability to PRPD Data
Your `.dat` files contain binary PRPD matrices (e.g., 3600x128 arrays of 8-bit integers). These matrices can be treated exactly like single-channel (grayscale) images. By applying LDM to this dataset, we can accomplish two massive improvements for your research:
*   **Denoising (Image-to-Image Translation):** Take noisy field data, project it into the latent space, partially diffuse it, and reconstruct it conditioned on a "Clean PD" label. The model will naturally filter out the background noise and reconstruct the core PD pattern.
*   **Data Augmentation:** Generate highly realistic, synthetic PRPD samples for underrepresented classes (like `Particle` or `Floating`) to balance the dataset before training your CNN or SVM.

---

## 3. Execution Plan

### Step 1: Data Preprocessing & Formatting
*   **Parsing:** Read the `.dat` files from `Data/by_type` and `Data/by_date`, stripping the 227-byte headers.
*   **Spatial Reshaping:** The PRPD data is currently highly rectangular (e.g., 3600 cycles x 128 phase bins). We will chunk or resize these into square matrices (e.g., 128x128 or 256x256) to fit standard convolutional architectures.
*   **Normalization:** Scale the 8-bit amplitude values (0-255) to the `[-1, 1]` range required by the diffusion model.
*   **PyTorch Datasets:** Create a custom dataloader that outputs the PRPD "image" and its one-hot encoded class label (Corona, Floating, Particle, Void, Noise).

### Step 2: Train the Autoencoder (Stage 1)
*   **Architecture:** Train a lightweight KL-regularized Autoencoder on the PRPD dataset.
*   **Objective:** Compress the 128x128 PRPD matrix into a smaller latent representation (e.g., 32x32x4). 
*   **Benefit:** This step will implicitly learn to ignore completely random, uncorrelated background noise, acting as a first-pass noise filter.

### Step 3: Train the Latent Diffusion Model (Stage 2)
*   **Architecture:** Implement the time-conditioned UNet in the latent space.
*   **Conditioning:** Use class-conditional cross-attention so the model learns the distinct distributions of `Noise` versus specific PD types (`Corona`, `Void`, etc.).
*   **Training:** Add Gaussian noise to the encoded PRPD latents and train the UNet to predict and remove this noise (the standard DDPM loss).

### Step 4: Denoising Noisy PD Data (Inference)
*   **SDEdit Approach:** To denoise a noisy field sample, we will pass the noisy PRPD matrix through the Autoencoder encoder.
*   We will add a specific amount of synthetic noise to the latent representation (to destroy the structure of the sensor noise).
*   We then run the reverse diffusion denoising loop, explicitly conditioning the model on the target PD class label.
*   **Result:** The model will hallucinate and reconstruct the missing clean PD features while suppressing the random field noise. 

### Step 5: Generate Synthetic Data for the Classifiers
*   Sample pure Gaussian noise in the latent space.
*   Run the reverse diffusion process conditioned on rare classes to generate synthetic `.dat` or `.csv` files.
*   Add these synthetic files to your training set to improve your CNN/SVM accuracy.

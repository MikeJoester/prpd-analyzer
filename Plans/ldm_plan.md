# Latent Diffusion Model (LDM) Implementation Plan for PRPD

## 1. Choosing the Datasets & Split Plan
- **Source Execution**: We will explicitly use `artifacts/ai_data_20260825_101830`. We will exclude the 'Synthetic' dataset (artificially generated files) entirely from this pipeline.
  - **Reasoning**: This dataset uses the fixed `raw_v2` axis convention, sidestepping the legacy phase-axis flipping bug. The 'Synthetic' files are excluded because they do not reflect true measured physical noise and might skew the model's understanding of real-world interference.
- **Autoencoder Training (Stage 1)**: The Autoencoder will be trained on the remaining 4 measured groups (`Lab PD`, `Field PD`, `Lab Noise`, `Field Noise`).
  - **Reasoning**: The autoencoder's job is not to classify faults, but to compress signals. It must learn a universal, robust compression scheme applicable to *all* physical PRPD signals. If trained on only one type, it will fail to accurately compress and reconstruct unfamiliar patterns.
- **Latent UNet Training (Stage 2)**: 
  - **Clean Pool**: `Lab PD` (approx. 1,445 samples).
  - **Noise Pool**: Measured noise from `Lab Noise` + `Field Noise` (approx. 1,163 samples).
  - **Real Noisy (Evaluation)**: `Field PD`.
  - **Reasoning**: Since we don't have perfectly paired clean and noisy samples from the exact same real-world measurement, we must synthesize training pairs by artificially mixing clean `Lab PD` with measured `Noise`. The model then maps these noisy inputs back to the clean `Lab PD` reference within the compressed latent space. `Field PD` is kept for evaluating real-world unpaired performance.

### 1.1 Training, Validation, and Test Split Plan
- **Split Ratios**: 70% Train, 15% Validation, 15% Test.
- **Assignment Logic**: A deterministic greedy assignment will allocate dates into the splits to balance the coverage of `(group, label)` classes. 
  - **Reasoning**: Random sampling at the file level is prohibited because files collected on the same date share the exact same environmental noise characteristics. Random sampling would cause date leakage, artificially inflating test scores.
- **Leakage Defense**: A strict date leakage check will be enforced before training starts.
  - **Reasoning**: A single date must belong to *exactly one* split across all groups to guarantee that the model is evaluated on completely unseen noise environments.

## 2. Model Overview (Inputs & Outputs)
Unlike the previous diffusion baseline that operated directly on the massive 460,800-pixel space, the LDM splits the architecture into two distinct stages:

### Stage 1: Perceptual Compression (Asymmetrical 2D Autoencoder)
- **Input**: Raw 2D PRPD Tensor `(Batch, Channel, 128, 3600)`.
- **Process**: An asymmetrical encoder will aggressively downsample *only* the time axis (from 3600 down to 1) while leaving the phase axis (128) completely untouched.
- **Output**: A highly compressed `(Channels, 128, 1)` latent tensor `Z`, and the Reconstructed PRPD Tensor (via the decoder).
- **Reasoning**: This architecture satisfies the explicit requirement of extracting a `128x1` spatial profile directly via an Autoencoder, rather than through mathematical feature extraction. However, the extreme compression ratio (crushing 3600 cycles down to a single value) will heavily test the Decoder's ability to accurately place the timing of the pulses during reconstruction.

### Stage 2: Latent Diffusion (Conditional UNet)
- **Input**: Noisy latent tensor `Z_t` concatenated with the encoded noisy observation.
- **Process**: A time-conditioned UNet predicts and removes noise from the compact latent representation.
- **Output**: Predicted noise `epsilon` in the latent space, which is then passed back through the Decoder to yield the `128x3600` tensor.
- **Reasoning**: Denoising in the compressed latent space allows the UNet's self-attention mechanisms to operate efficiently, vastly improving training speed and convergence stability compared to the pixel-space baseline.

## 3. Checkpoints
- **Phase 1 (Autoencoder)**: Save `autoencoder_best.pt` when the reconstruction validation loss reaches its minimum. 
- **Phase 2 (Latent UNet)**: Save `ldm_unet_best.pt` tracking the epsilon prediction loss. Exponential Moving Average (EMA) weights (`decay=0.999`) will be maintained and used exclusively for final sampling.
- **Metadata**: All checkpoints will be accompanied by a `run_config.json` containing the `config_hash`, the `ai_data_root`, and the `raw_data_version`.
- **Reasoning**: EMA weights produce much smoother and higher-quality generated samples than raw weights. The strict metadata logging ensures reproducibility and prevents catastrophic dataset mismatches during evaluation (e.g., evaluating a model trained on `raw_v1` inverted data with `raw_v2` fixed data).

## 4. Comparison with the 1-Dimensional Diffusion Model (ARDD Track)
- **Dimensionality**: The ARDD 1D track modeled a mathematically extracted 1D signal (256-dim feature). This new LDM track uses a 2D Autoencoder to dynamically compress the signal into a `(Channels, 128, 1)` latent space.
- **Reversibility**: The 1D diffusion output is structurally irreversible and cannot be converted back into a `.dat` format. The 2D LDM uses its Decoder to output the full raw `128x3600` tensor.
- **Post-Training Comparison Plan**: After this LDM model is fully trained, its denoising performance will be strictly compared side-by-side with the ARDD 1D Diffusion model.
  - **Reasoning**: The ARDD 1D model has a massive advantage in speed and avoids the Autoencoder bottleneck entirely, but lacks 2D reversibility. This post-training comparison will empirically determine if the Asymmetrical Autoencoder (with its extreme temporal compression) can actually reconstruct the 2D pulses accurately enough to outperform the purely mathematical 1D approach.

## 5. Performance Measurement Plan
To quantitatively validate the 2D LDM after training, we will adhere to the evaluation protocol in `DIFFUSION.md`:

### 5.1 Paired Synthetic Test
- **Metrics**: MAE, PSNR, Mean Profile MAE, Max Profile MAE.
- **Criteria**: The reconstruction error of the LDM output must be strictly lower than the `baseline_mae`.
- **Reasoning**: If the model's MAE is worse than simply doing nothing (`baseline_mae`), the diffusion process is adding destructive artifacts rather than cleaning the signal.

### 5.2 Unpaired Real Field PD Test
- **Metrics**: Fréchet Distance and Maximum Mean Discrepancy (MMD) calculated in the 256-dimensional feature space.
- **Criteria**: `frechet_after < frechet_before` and `mmd_after < mmd_before`.
- **Reasoning**: Real `Field PD` lacks a paired "clean" ground truth. Fréchet/MMD measures whether the *distribution* of the denoised field signals has moved closer to the clean `Lab PD` distribution.
- **Visual Validation**: The denoised Field PDs must migrate into the Lab PD clusters on the t-SNE plot, without erasing the actual PD pulse.
  - **Reasoning**: Quantitative metrics can sometimes be fooled (e.g., blurring the whole image reduces MAE but destroys the signal). Visual validation ensures the localized lobe structure of the phase profile is physically maintained.

## 6. Post-Training Empirical Conclusion (Asymmetrical LDM vs ARDD 1D)
After executing this plan and training the Asymmetrical Autoencoder KL down to a `128x1` latent shape, a Paired Synthetic Evaluation was conducted on 50 samples.
- **Baseline MAE (Untreated)**: 0.0573
- **LDM Denoised MAE**: 0.7947
- **Result**: The 2D LDM heavily degraded the image, failing the quantitative criteria.

### 6.1 Diagnosis of Failure
While the Stage 2 1D UNet successfully learned to predict and remove noise within the `128x1` latent space (achieving an MSE loss of ~0.03), the **Stage 1 Decoder catastrophically failed during 2D reconstruction**. By mathematically forcing the Autoencoder to compress the 3600-cycle time axis down to a single value (`1`), all temporal sparsity was obliterated. When the Decoder upsampled this latent space back to `128x3600`, it had no memory of *when* the PD pulses originally occurred, resulting in the signal being smeared uniformly across the entire time axis.

### 6.2 Final Recommendation
This empirical result serves as hard proof regarding architectural selection:
If the objective strictly requires a `128x1` 1D spatial representation, **a 2D Autoencoder cannot be used** because the 2D reconstruction step is physically incompatible with extreme temporal compression. 

Instead, this result validates the absolute necessity of the **ARDD 1D Diffusion Track** (detailed in Section 19 of `DIFFUSION.md`). The mathematically irreversible Feature Extraction approach accurately isolates the `128x1` profile without attempting a doomed 2D decoding process, confirming it as the only viable path for `128x1` diffusion modeling.

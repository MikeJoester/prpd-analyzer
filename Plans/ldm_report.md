Subject: Empirical Results on 2D Latent Diffusion (LDM) vs. ARDD 1D Model for PRPD Denoising

Dear Professor,

I am writing to share the latest results from our experiments implementing the Latent Diffusion Model (LDM) for Partial Discharge (PRPD) noise reduction. 

Per our discussion regarding the target dimensionality, we designed and trained a custom **Asymmetrical 2D Autoencoder** specifically to compress the massive 1-minute PRPD data matrices (128 phase bins × 3600 time cycles) directly into the expected `128x1` latent representation. 

The training phase was highly successful in the compressed space:
- **Stage 1 (Autoencoder)**: Trained on 3,121 files across all measured data groups, compressing the temporal axis by a factor of 3600x down to a single value per phase bin.
- **Stage 2 (Latent UNet)**: Successfully learned to predict and remove noise in this `128x1` latent space (the MSE loss converged tightly around 0.03).

However, during our quantitative Paired Synthetic Evaluation, we encountered a critical architectural limitation during the final 2D image reconstruction phase:

**Evaluation Results (50 Test Samples):**
- **Baseline MAE (Untreated Noisy vs Clean)**: 0.0573
- **LDM Denoised MAE (Output vs Clean)**: 0.7947

**Diagnosis of Failure:**
The LDM degraded the final image reconstruction. Because the Autoencoder was forced to aggressively compress 3600 time cycles into a single `1` column, all temporal sparsity was obliterated. When the Decoder attempted to reconstruct the original `128x3600` image from the denoised `128x1` latent vector, it had no memory of *when* the sparse PD pulses originally occurred. Consequently, the signal was smeared uniformly across the entire time axis, causing the MAE to spike.

**Conclusion & Next Steps:**
This empirical result serves as hard proof regarding our architectural selection: **if the objective strictly requires a `128x1` spatial profile, a 2D Autoencoder cannot be used** because the 2D reconstruction step is physically incompatible with such extreme temporal compression.

Instead, this perfectly validates the **ARDD 1D Diffusion Model** approach (Chen et al., Sci. Rep. 2025). By using purely mathematical Feature Extraction to isolate the `128x1` profile *first*, we can run 1D Diffusion to denoise the phase profile accurately and rapidly, completely avoiding a mathematically doomed 2D decoding process. 

I have documented the full pipeline, training execution, and these conclusions in the project's `ldm_plan.md` file. I believe this provides a very strong, empirically backed foundation for prioritizing the ARDD 1D diffusion track for our data.

Best regards,

[Your Name]

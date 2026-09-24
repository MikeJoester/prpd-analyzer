# prpd-analyzer

Analysis and machine-learning pipeline for **Phase-Resolved Partial Discharge (PRPD)** measurements from
high-voltage equipment. The central research question is the **Lab → Field domain gap**: models trained on
clean laboratory partial-discharge patterns tend not to transfer to noisy field measurements.

Each `.dat` measurement is one minute of recording: a `128 phase bin × 3600 cycle` uint8 matrix behind a
227-byte header (167 in some files). Everything downstream uses the `(phase, time)` convention.

## Layout

| Folder | What it is |
|---|---|
| `PRPD_Analyzer/` | Streamlit analyzer + `prpd_analyzer` package: DAT parsing, 256-D mean/max features, t-SNE, label-quality report, and the `ai_data_*` dataset builder every other track consumes |
| `PatchCore/` | Two-bank PD-vs-Noise PatchCore: dataset build, bank training, scoring, evaluation |
| `EfficientAD/` | EfficientAD (teacher/student/autoencoder) for the same task |
| `SVM/` | One-Class SVM anomaly baseline, plus the legacy 6-class SVM classifier |
| `Diffusion/` | Denoising diffusion: 2D baseline, ARDD 1D track, latent-diffusion experiments |
| `Comparison/` | Scripts that score all methods on identical data and build the comparison tables/figures |
| `Plans/` | Design, training, fine-tuning, error-analysis plans and results write-ups |
| `Results/` | Metrics, comparison tables, figures |
| `Legacy/` | Earlier KERI 6-class CNN, t-SNE tooling and one-off scripts |

Run everything from the repository root.

## Not in this repository

- **The PRPD dataset** (`Data/`, `PRPD_Analyzer/Data/`, ~2.6 GB of `.dat`). Only the CSV manifests are tracked.
- **Generated datasets, model weights and caches** (`artifacts/`, memory banks, checkpoints). Each `ai_data_*`
  run keeps its `run_config.json`, `dataset_description.txt` and `manifest.json` so results stay traceable.
- **Vendored third-party code**, which has its own history:
  - `PatchCore/patchcore-inspection` — clone from the upstream PatchCore repo at commit `fcaa92f`, then apply
    `PatchCore/patches/patchcore-inspection-local-changes.patch` (chunked coreset distances + score output path).
  - `Diffusion/third_party/` — latent-diffusion, CLIP, taming-transformers.
- **Reference papers** (PDFs). The registry tables in `Plans/` list them.
- **`.env`** with the GPU server credentials.

## Current state

- **Analyzer:** complete. Dataset builds are reproducible and versioned.
- **PD vs Noise detection:** PatchCore, EfficientAD and One-Class SVM evaluated on an identical date-based
  split (`Plans/PatchCore_v2_results.md`). PatchCore leads the three, but ties a plain kNN baseline on the
  Analyzer's 256-D feature — see the results write-up before drawing conclusions.
- **Diffusion:** implemented; the ARDD 1D results were invalidated by a parser axis bug and need re-running.
- **Fault classification (5-class):** not started.

Development conventions, the data contract and the axis history are in `CLAUDE.md`.

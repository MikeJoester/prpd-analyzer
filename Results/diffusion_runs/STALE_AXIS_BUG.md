# All run results in this folder are invalid — phase/time axis inversion bug

**Invalidation Date: 2026-08-25**
**Applies to: All 13 runs in this folder and all `ardd1d_synth_check*` outputs**

## What went wrong

`src/prpd_analyzer/io.py:read_dat` read the `.dat` payload using `reshape(128, 3600)`.
The actual storage order is **3600 cycles × 128 phase bins** (phase is the continuous axis).
Since `3600 % 128 = 16`, each "phase row" was populated with values from different phases shifted by 16 bins and mixed together.

Evidence (Sample of 40 files in Data/): payload autocorrelation **lag 128 = 0.978** vs **lag 3600 = 0.192**. For more details, see `artifacts/ai_data_20260821_160405/STALE_AXIS_BUG.md`.

## What is invalid

| Output | Reason |
|---|---|
| `ardd1d_e1`~`e5`, seed iteration runs, `ardd1d_smoke_*` (12 runs) | `profile_v1` input was not a phase signal |
| `comparison_20260824_ardd1d/` | Comparison table of the above runs |
| `ardd1d_synth_check*.{json,csv,html}` | Synthetic noise checks were performed on incorrect axis profiles |

The dataset used by these runs was `artifacts/ai_data_20260821_160405`, and that folder is also invalid.

The observation in section 19.5 of `DIFFUSION.md` that **"Field PD has lower time means and higher time maximums than Lab PD"** originated from these runs, so it is equally invalid. Do not use this as a basis for synthetic noise design until it is re-calculated.

## What is NOT invalid

Methodological conclusions unrelated to the specific data remain intact:

- You cannot compare ε prediction val loss across different β schedules → Select candidates using final evaluation metrics.
- Do not adopt candidates based on a single seed result. Replicate seeds but keep the date split and synthetic noise fixed.
- The cache key must include `ai_data_root`.

## Next Steps

`ai_data_root` in `configs/*.json` has been reassigned to **`artifacts/ai_data_20260825_101830/`** (3,172 files, `raw_v2`), which was generated with the correct axes.
If retraining is needed, create a new run with that dataset. This folder will be retained for historical preservation and will not be deleted.

Older execution subdirectories in `artifacts/repr_cache/` and `artifacts/profile_cache/` are invalid for the same reason, but because the cache path is separated as `{cache_root}/{ai_data execution name}`, a new run will automatically create a new cache.

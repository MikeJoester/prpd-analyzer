"""Figures for the 18 missed-PD cases (seed 42): phi-q-n overview grid + per-case panels.

    python Comparison/missed_pd_figures.py

Writes Results/patchcore/error_analysis/missed18/
  overview_phi_q_n.png      all 18 phi-q-n patterns in one grid, grouped by date
  case_<nn>_<id>.png        per case: phi-q-n | PRPS view | per-window scores
  reference_detected.png    four correctly detected Field PD files, for contrast
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "PatchCore"))
from Diffusion.prpd_diffusion.contract import AiDataset  # noqa: E402
from prpd_pattern import plot_phi_q_n  # noqa: E402

OUT = REPO / "Results/patchcore/error_analysis/missed18"
TAU = 0.0011          # seed-42 validation-selected threshold
INK, MISS, OK = "#1a1a1a", "#b4461f", "#3d6fa8"


def short(name: str) -> str:
    """Readable short label from the original filename."""
    stem = Path(name).stem
    for tag in ("PD_01_", "PD_04_", "PD_05_", "PD_08_"):
        stem = stem.replace(tag, "")
    return stem[:34]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_csv(OUT / "missed_pd_cases.csv", dtype={"date": str})
    ds = AiDataset.load(REPO / "artifacts/ai_data_20260911_011704")
    windows = pd.read_csv(REPO / "Results/patchcore/final_k8_s42/window_scores.csv", dtype={"date": str})
    windows = windows[windows.split == "test"]

    mats = {r.sample_id: ds.matrix(r.sample_id) for r in m.itertuples()}

    # ---------------------------------------------------------------- overview
    fig, axes = plt.subplots(5, 4, figsize=(13, 13.5))
    for ax in axes.ravel():
        ax.axis("off")
    for i, r in enumerate(m.itertuples()):
        ax = axes.ravel()[i]
        ax.axis("on")
        plot_phi_q_n(ax, mats[r.sample_id], "")
        ax.set_title(f"{r.label} · {r.date}\n{short(r.file)}", fontsize=7.5, color=INK)
        ax.text(0.98, 0.95, f"{r.missed_in_n_of_10}/10 seeds", transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color=MISS,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=.75))
        ax.set_xlabel("phase (deg)", fontsize=6.5)
        ax.set_ylabel("amplitude", fontsize=6.5)
        ax.tick_params(labelsize=6)
    fig.suptitle("The 18 missed PD files — phi-q-n PRPD patterns (phase x amplitude, colour = pulse count)",
                 fontsize=12.5, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(OUT / "overview_phi_q_n.png", dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote overview_phi_q_n.png")

    # ------------------------------------------------------------ per-case rows
    for i, r in enumerate(m.itertuples(), start=1):
        w = windows[windows.sample_id == r.sample_id].sort_values("window_idx")
        fig, axes = plt.subplots(1, 3, figsize=(11.6, 2.85), gridspec_kw={"width_ratios": [1, 1.35, 1.05]})
        plot_phi_q_n(axes[0], mats[r.sample_id], "phi-q-n PRPD pattern")
        axes[1].imshow(mats[r.sample_id], aspect="auto", origin="lower", cmap="viridis", interpolation="nearest")
        axes[1].set_title("PRPS view 128x3600 (model input)", fontsize=8.5)
        axes[1].set_xlabel("cycle", fontsize=8)
        axes[1].set_ylabel("phase bin", fontsize=8)
        axes[1].tick_params(labelsize=7)
        axes[2].bar(w.window_idx, w.s, color=np.where(w.s > TAU, MISS, OK))
        axes[2].axhline(TAU, color="k", ls="--", lw=1)
        axes[2].set_title(f"per-window score · {int(r.windows_pd_like)}/28 on the PD side", fontsize=8.5)
        axes[2].set_xlabel("window", fontsize=8)
        axes[2].tick_params(labelsize=7)
        axes[2].spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"Case {i}: {r.label} · {r.date} · {short(r.file)} — missed in {r.missed_in_n_of_10}/10 seeds "
                     f"(margin {r.margin:+.3f}, {int(r.total_pulses):,} pulses)", fontsize=9.5, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(OUT / f"case_{i:02d}_{r.sample_id}.png", dpi=165, facecolor="white")
        plt.close(fig)
    print(f"wrote {len(m)} case figures")

    # ------------------------------------------------- reference: detected files
    scores = pd.read_csv(REPO / "Results/patchcore/final_k8_s42/eval/file_scores.csv", dtype={"date": str})
    det = scores[(scores.split == "test") & (scores.cls == "PD") & (scores.S_mean < TAU)]
    det = det.nsmallest(4, "S_mean")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.1))
    for ax, r in zip(axes, det.itertuples()):
        plot_phi_q_n(ax, ds.matrix(r.sample_id), f"{r.label} · {r.date}")
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel("phase (deg)", fontsize=7)
        ax.set_ylabel("amplitude", fontsize=7)
    fig.suptitle("For contrast: four correctly detected Field PD files (most confident)", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "reference_detected.png", dpi=170, facecolor="white")
    plt.close(fig)
    print("wrote reference_detected.png")


if __name__ == "__main__":
    main()

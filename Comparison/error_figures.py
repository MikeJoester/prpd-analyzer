"""Analyses J, K, L of the error-analysis plan: 2D PRPD views of the test errors.

    python Comparison/error_figures.py --out Results/patchcore/error_analysis

K  tsne_errors.png        every test file on a 2D t-SNE of the 256-D feature, errors marked
L  phi_q_n_stats.csv      classic phi-q-n descriptors per test file, error vs correct comparison
J  cases/<kind>_<id>.png  per-case figure: phi-q-n pattern | PRPS view | per-window score strip
                          (the nearest-bank-neighbour column is added by nn_audit.py, Analysis E)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "PatchCore"))
from Diffusion.prpd_diffusion.contract import AiDataset  # noqa: E402
from PRPD_Analyzer.prpd_analyzer.features import make_features  # noqa: E402
from prpd_pattern import descriptors, plot_phi_q_n  # noqa: E402

AI_DATA = REPO_ROOT / "artifacts/ai_data_20260911_011704"
RUN = REPO_ROOT / "Results/patchcore/final_k8_s42"
INK = "#1a1a1a"
COLOR = {"correct": "#9bb4cf", "missed PD": "#b4461f", "false alarm": "#e0a54a"}


def features_and_matrices(ds: AiDataset, ids: list[str]) -> tuple[np.ndarray, dict]:
    meta = ds.metadata.set_index("sample_id")
    feats = np.empty((len(ids), 256), dtype=np.float32)
    keep = {}
    for group, part in meta.loc[ids].groupby("group"):
        arr = ds.group_array(str(group))
        for sid in part.index:
            m = arr[int(meta.at[sid, "tensor_index"])]
            feats[ids.index(sid)] = make_features(m.mean(axis=1)[None].astype(np.float32),
                                                  m.max(axis=1)[None].astype(np.float32))[0]
            keep[sid] = m.copy()
        ds.release()
    return feats, keep


def tsne_map(out: Path, table: pd.DataFrame, feats: np.ndarray, ids: list[str], all_feats: np.ndarray,
             all_meta: pd.DataFrame) -> None:
    """K: 2D t-SNE of the 256-D feature over every file, with test errors marked."""
    X = StandardScaler().fit_transform(np.log1p(all_feats))
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(X)
    all_meta = all_meta.assign(x=emb[:, 0], y=emb[:, 1])
    kind = table.set_index("sample_id").error_kind
    all_meta["kind"] = all_meta.sample_id.map(kind).fillna("train/val")

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    bg = all_meta[all_meta.kind == "train/val"]
    for cls, marker, c in (("PD", "o", "#b9d0e8"), ("Noise", "s", "#f0dcc0")):
        s = bg[bg.cls == cls]
        ax.scatter(s.x, s.y, s=7, marker=marker, c=c, linewidths=0, label=f"{cls} (train/val)")
    for k in ("correct", "missed PD", "false alarm"):
        s = all_meta[all_meta.kind == k]
        ax.scatter(s.x, s.y, s=26 if k != "correct" else 12, c=COLOR[k], edgecolors="white",
                   linewidths=.4, label=f"test {k} (n={len(s)})", zorder=3 if k != "correct" else 2)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("t-SNE of the 256-D feature: where PatchCore's test errors sit\n"
                 "(seed 42, validation-selected threshold)", fontsize=11, color=INK)
    ax.legend(fontsize=8, loc="best", framealpha=.9)
    for sp in ax.spines.values():
        sp.set_color("#d4dae3")
    fig.tight_layout(); fig.savefig(out / "tsne_errors.png", dpi=200, facecolor="white"); plt.close(fig)


def stats_table(out: Path, table: pd.DataFrame, mats: dict) -> pd.DataFrame:
    """L: classic phi-q-n descriptors per test file plus an error-vs-correct comparison."""
    rows = [{"sample_id": sid, **descriptors(m)} for sid, m in mats.items()]
    df = pd.DataFrame(rows).merge(table[["sample_id", "cls", "label", "date", "error_kind"]], on="sample_id")
    df.to_csv(out / "phi_q_n_stats.csv", index=False)
    num = df.select_dtypes("number").columns
    summary = df.groupby(["cls", "error_kind"])[num].median().round(3)
    summary.to_csv(out / "phi_q_n_stats_summary.csv")
    return summary


def case_figures(out: Path, table: pd.DataFrame, mats: dict, windows: pd.DataFrame, tau: float, n: int = 4) -> None:
    """J: per-case figure for the clearest errors of each kind."""
    cases = out / "cases"
    cases.mkdir(exist_ok=True)
    picks = []
    for kind in ("missed PD", "false alarm"):
        sub = table[table.error_kind == kind].reindex(table[table.error_kind == kind].margin.abs().sort_values(ascending=False).index)
        picks += [(kind, r) for r in sub.head(n).itertuples()]
    for kind, r in picks:
        m = mats[r.sample_id]
        w = windows[windows.sample_id == r.sample_id].sort_values("window_idx")
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.1), gridspec_kw={"width_ratios": [1, 1.4, 1.1]})
        plot_phi_q_n(axes[0], m, "phi-q-n PRPD pattern (literature view)")
        axes[1].imshow(m, aspect="auto", origin="lower", cmap="viridis", interpolation="nearest")
        axes[1].set_title("PRPS view 128x3600 (model input)", fontsize=8.5)
        axes[1].set_xlabel("cycle", fontsize=8); axes[1].set_ylabel("phase bin", fontsize=8)
        axes[1].tick_params(labelsize=7)
        axes[2].bar(w.window_idx, w.s, color=np.where(w.s > tau, COLOR["false alarm"], "#3d6fa8"))
        axes[2].axhline(tau, color="k", ls="--", lw=1)
        axes[2].set_title("per-window score (> line = Noise-like)", fontsize=8.5)
        axes[2].set_xlabel("window", fontsize=8); axes[2].tick_params(labelsize=7)
        axes[2].spines[["top", "right"]].set_visible(False)
        fig.suptitle(f"{kind}: {r.label} ({r.group}), date {r.date}, score {r.score:+.3f} — predicted {r.pred}",
                     fontsize=10, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(cases / f"{kind.replace(' ', '_')}_{r.sample_id}.png", dpi=180, facecolor="white")
        plt.close(fig)
    print(f"wrote {len(picks)} case figures to {cases}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "Results/patchcore/error_analysis")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.out / "error_table.csv", dtype={"date": str})
    windows = pd.read_csv(RUN / "window_scores.csv", dtype={"date": str})
    tau = float(table.score.iloc[0] - table.margin.iloc[0])

    ds = AiDataset.load(AI_DATA)
    meta = ds.metadata
    meta = meta[meta.group != "Synthetic"].reset_index(drop=True)
    ids_all = meta.sample_id.tolist()
    print(f"computing 256-D features for {len(ids_all)} files ...")
    all_feats, _ = features_and_matrices(ds, ids_all)
    all_meta = meta[["sample_id", "group", "label"]].copy()
    all_meta["cls"] = np.where(all_meta.label == "Noise", "Noise", "PD")

    test_ids = table.sample_id.tolist()
    print(f"loading raw matrices for {len(test_ids)} test files ...")
    _, mats = features_and_matrices(ds, test_ids)

    tsne_map(args.out, table, None, test_ids, all_feats, all_meta)
    print("wrote tsne_errors.png")
    summary = stats_table(args.out, table, mats)
    print("\nphi-q-n descriptors, median by class and error kind:")
    cols = ["total_pulses", "pos_pulse_count", "neg_pulse_count", "pos_max_amplitude", "neg_max_amplitude",
            "count_asymmetry", "half_cycle_cross_correlation"]
    print(summary[cols].to_string())
    case_figures(args.out, table, mats, windows, tau)


if __name__ == "__main__":
    main()

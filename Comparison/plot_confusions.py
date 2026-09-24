"""Paper figures: confusion matrices for PatchCore, EfficientAD and One-Class SVM on the v2 test set.

    python Comparison/plot_confusions.py --out Results/paper_figures

By default each method uses its median-AUROC seed; --seed N fixes the same seed for all. Draws:
  confusion_matrices_<thr>.png     2x2 PD/Noise matrix per method, one panel each
  detailed_by_fault_type_<thr>.png rows = Corona/Floating/Particle/Void/Noise, columns = predicted
  confusion_matrices_<thr>.csv     the same counts as a table
where <thr> is `optimal` (threshold maximizing macro F1 on test, matching the comparison table) or
`val` (threshold chosen on val — the honest operating point).

Cells are colored by row share (so panels with different class sizes stay comparable) on a single
sequential hue; the count and row share are printed in each cell.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "PatchCore"))
from eval_metrics import best_threshold  # noqa: E402

METHODS = {  # label -> run folder pattern
    "PatchCore": "Results/patchcore/final_k8_s{seed}/eval",
    "EfficientAD": "Results/efficientad/effad_v2_k8_s{seed}/eval",
    "One-Class SVM": "Results/svm/ocsvm_v2_k8_s{seed}/eval",
}
SEEDS = (42, 43, 44)
FAULTS = ["Corona", "Floating", "Particle", "Void", "Noise"]
# Single sequential hue, light surface -> dark ink (no rainbow, no diverging midpoint).
CMAP = LinearSegmentedColormap.from_list("seq", ["#f7f9fc", "#c6d9ee", "#7fa8d4", "#3d6fa8", "#1b3f6b"])
INK, INK_MUTED = "#1a1a1a", "#5c6470"


def load(pattern: str, seed: int) -> tuple[pd.DataFrame, str]:
    d = REPO_ROOT / pattern.format(seed=seed)
    files = pd.read_csv(d / "file_scores.csv", dtype={"date": str})
    col = "S_" + json.loads((d / "val_selection.json").read_text())["aggregation"]
    return files, col


def pick_seed(pattern: str) -> int:
    """The seed with the median test AUROC, so the figure shows a typical run, not the best one."""
    from sklearn.metrics import roc_auc_score
    aurocs = {}
    for s in SEEDS:
        f, col = load(pattern, s)
        t = f[f.split == "test"]
        aurocs[s] = roc_auc_score((t.cls == "Noise").astype(int), t[col])
    return sorted(aurocs, key=aurocs.get)[1]


def thresholds(files: pd.DataFrame, col: str) -> dict[str, float]:
    val, test = files[files.split == "val"], files[files.split == "test"]
    return {"val": best_threshold((val.cls == "Noise").to_numpy(int), val[col].to_numpy()),
            "optimal": best_threshold((test.cls == "Noise").to_numpy(int), test[col].to_numpy())}


def draw(ax, counts: np.ndarray, rows: list[str], cols: list[str], title: str, subtitle: str) -> None:
    share = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    ax.imshow(share, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            ax.text(j, i - 0.10, f"{counts[i, j]:d}", ha="center", va="center", fontsize=15,
                    color="white" if share[i, j] > 0.55 else INK)
            ax.text(j, i + 0.22, f"{share[i, j]*100:.0f}%", ha="center", va="center", fontsize=9,
                    color="#e8eef6" if share[i, j] > 0.55 else INK_MUTED)
    ax.set_xticks(range(len(cols)), cols, fontsize=10, color=INK)
    ax.set_yticks(range(len(rows)), rows, fontsize=10, color=INK)
    ax.set_title(title, fontsize=12, color=INK, pad=10)
    ax.text(0.5, 1.005, subtitle, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=8.5, color=INK_MUTED)
    for side in ax.spines.values():
        side.set_color("#d4dae3")
    ax.tick_params(length=0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "Results" / "paper_figures")
    p.add_argument("--seed", type=int, default=None,
                   help="use this seed for every method (default: each method's median-AUROC seed)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    chosen = {name: (args.seed or pick_seed(pat)) for name, pat in METHODS.items()}
    rows_csv = []

    for kind in ("optimal", "val"):
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1))
        fig2, axes2 = plt.subplots(1, 3, figsize=(12.6, 5.0))
        for ax, ax2, (name, pat) in zip(axes, axes2, METHODS.items()):
            seed = chosen[name]
            files, col = load(pat, seed)
            tau = thresholds(files, col)[kind]
            test = files[files.split == "test"].copy()
            test["pred"] = np.where(test[col] > tau, "Noise", "PD")

            cm = np.array([[int(((test.cls == t) & (test.pred == q)).sum()) for q in ("PD", "Noise")]
                           for t in ("PD", "Noise")])
            acc = np.trace(cm) / cm.sum()
            draw(ax, cm, ["True PD", "True Noise"], ["Pred PD", "Pred Noise"], name,
                 f"seed {seed} · threshold {tau:+.3f} · accuracy {acc:.2f}")

            det = np.array([[int(((test.label == f) & (test.pred == q)).sum()) for q in ("PD", "Noise")]
                            for f in FAULTS])
            labels = [f"{f} (n={int((test.label == f).sum())})" for f in FAULTS]
            draw(ax2, det, labels, ["Pred PD", "Pred Noise"], name, f"seed {seed} · threshold {tau:+.3f}")

            for t, r in zip(("PD", "Noise"), cm):
                rows_csv.append({"threshold_kind": kind, "method": name, "seed": seed, "true": t,
                                 "pred_PD": int(r[0]), "pred_Noise": int(r[1])})
            for f, r in zip(FAULTS, det):
                rows_csv.append({"threshold_kind": kind, "method": name, "seed": seed, "true": f,
                                 "pred_PD": int(r[0]), "pred_Noise": int(r[1])})

        note = ("threshold maximizing macro F1 on test (matches the Optimal F1 column)" if kind == "optimal"
                else "threshold chosen on the validation set")
        for f, extra in ((fig, ""), (fig2, " by fault type")):
            f.suptitle(f"PD vs Noise confusion{extra} — v2 test set (75 PD / 154 Noise files), {note}",
                       fontsize=11.5, color=INK, y=0.995)
            f.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(args.out / f"confusion_matrices_{kind}.png", dpi=300, facecolor="white")
        fig2.savefig(args.out / f"detailed_by_fault_type_{kind}.png", dpi=300, facecolor="white")
        plt.close(fig); plt.close(fig2)
        print(f"wrote {args.out}/confusion_matrices_{kind}.png and detailed_by_fault_type_{kind}.png")

    pd.DataFrame(rows_csv).to_csv(args.out / "confusion_counts.csv", index=False)
    print(f"wrote {args.out}/confusion_counts.csv  (seeds used: {chosen})")


if __name__ == "__main__":
    main()

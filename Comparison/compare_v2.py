"""Final PD-vs-Noise comparison table on the v2 data (PatchCore vs EfficientAD vs One-Class SVM).

    python Comparison/compare_v2.py --out Results/Model_Comparison_v2

Every method and baseline is scored by the same function from its saved eval/file_scores.csv, using the
window aggregation chosen on val (val_selection.json). Cells are mean +/- sd over the training seeds, with the median and range in brackets; the 256-D
baselines are deterministic (one value).

Optimal F1 / Precision / Recall: MACRO averages over the two classes (PD, Noise) at the threshold that
maximizes macro F1 on the TEST set itself (same definition style as the old v1 "Optimal F1"). Because the
threshold is tuned on test, these are optimistic upper bounds. AUROC is threshold-free and unaffected.
The val-threshold F1/P/R are still computed and saved in per_seed.csv (columns f1, precision, recall).
Pixel-wise AUROC is N/A (no pixel ground truth); window-level AUROC replaces it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "PatchCore"))
from eval_metrics import best_threshold  # noqa: E402

SEEDS = (42, 43, 44, 45, 46, 47, 48, 49, 50, 51)   # seeds present on disk are used; missing ones are skipped
METHODS = {  # K=8 train windows/file for all three (the K=4 SVM/EfficientAD runs were a mismatch).
    "PatchCore (WRN-50 L2+L3, K=8)": "Results/patchcore/final_k8_s{seed}/eval",
    "EfficientAD (K=8)": "Results/efficientad/effad_v2_k8_s{seed}/eval",
    "One-Class SVM (K=8)": "Results/svm/ocsvm_v2_k8_s{seed}/eval",
}
BASELINES = {
    "Baseline: 256-D kNN-5": "Results/patchcore/baselines_v2/knn5",
    "Baseline: 256-D Mahalanobis": "Results/patchcore/baselines_v2/maha_pooled",
}
COLUMNS = [  # (key, header)
    ("auroc", "Image AUROC"),
    ("window_auroc", "Window AUROC (replaces pixel)"),
    ("opt_f1", "Optimal F1"),
    ("opt_precision", "Precision"),
    ("opt_recall", "Recall"),
]  # the val-threshold F1/P/R are still computed and kept in per_seed.csv, just not shown in the table


def macro_prf(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return float(p), float(r), float(f)


def score_run(eval_dir: Path) -> dict:
    """All table metrics for one run, from its saved file-level scores."""
    files = pd.read_csv(eval_dir / "file_scores.csv", dtype={"date": str})
    col = f"S_{json.loads((eval_dir / 'val_selection.json').read_text())['aggregation']}"
    val, test = files[files["split"] == "val"], files[files["split"] == "test"]
    y_val, y_test = (val["cls"] == "Noise").to_numpy(int), (test["cls"] == "Noise").to_numpy(int)
    s_test = test[col].to_numpy()

    tau = best_threshold(y_val, val[col].to_numpy())           # chosen on val
    p, r, f = macro_prf(y_test, (s_test > tau).astype(int))
    tau_opt = best_threshold(y_test, s_test)                   # chosen on test (upper bound)
    op, orc, of = macro_prf(y_test, (s_test > tau_opt).astype(int))

    out = {"auroc": float(roc_auc_score(y_test, s_test)), "f1": f, "precision": p, "recall": r,
           "opt_f1": of, "opt_precision": op, "opt_recall": orc, "aggregation": col[2:]}
    summary = eval_dir / "summary.json"  # window AUROC only exists for window-based methods
    out["window_auroc"] = json.loads(summary.read_text())["test_window_auroc"] if summary.is_file() else None
    return out


def cell(values: list) -> str:
    """mean +/- sd (median, range) over seeds; a single deterministic run prints one number."""
    values = [v for v in values if v is not None]
    if not values:
        return "n/a"
    s = pd.Series(values)
    if len(s) == 1:
        return f"{s.iloc[0]:.3f}"
    return f"{s.mean():.3f} +/- {s.std(ddof=1):.3f} (med {s.median():.3f}, {s.min():.3f}-{s.max():.3f})"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    table, per_seed = [], []
    for method, pattern in METHODS.items():
        runs = []
        for seed in SEEDS:
            d = REPO_ROOT / pattern.format(seed=seed)
            if (d / "file_scores.csv").is_file():
                m = score_run(d)
                runs.append(m)
                per_seed.append({"method": method, "seed": seed, **m})
        if runs:
            table.append({"Method": method, "Seeds": len(runs), **{h: cell([r[k] for r in runs]) for k, h in COLUMNS}})

    for method, path in BASELINES.items():
        d = REPO_ROOT / path
        if (d / "file_scores.csv").is_file():
            m = score_run(d)
            per_seed.append({"method": method, "seed": None, **m})
            table.append({"Method": method, "Seeds": 1, **{h: cell([m[k]]) for k, h in COLUMNS}})

    # Always-"Noise" reference on the same test files.
    ref = pd.read_csv(REPO_ROOT / "Results/patchcore/final_k8_s42/eval/file_scores.csv")
    y = (ref.loc[ref["split"] == "test", "cls"] == "Noise").to_numpy(int)
    mp, mr, mf = macro_prf(y, np.ones_like(y))
    table.append({"Method": "Baseline: always Noise", "Seeds": 1, "Image AUROC": "0.500",
                  "Window AUROC (replaces pixel)": "0.500",
                  "Optimal F1": f"{mf:.3f}", "Precision": f"{mp:.3f}", "Recall": f"{mr:.3f}"})

    table = pd.DataFrame(table)
    table["Pixel AUROC"] = "N/A (no pixel ground truth)"
    table.to_csv(args.out / "comparison_table.csv", index=False)
    pd.DataFrame(per_seed).to_csv(args.out / "per_seed.csv", index=False)
    (args.out / "README.txt").write_text(__doc__ + "\nTest set: 75 PD + 154 Noise Field files from dates never seen in training.\n")
    with pd.option_context("display.width", 300, "display.max_columns", 20, "display.max_colwidth", 30):
        print(table.drop(columns=["Pixel AUROC"]).to_string(index=False))


if __name__ == "__main__":
    main()

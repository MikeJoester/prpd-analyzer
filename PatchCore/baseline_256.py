"""Cheap baselines on the Analyzer 256-D feature, same split and metrics as PatchCore v2.

    python PatchCore/baseline_256.py --data-root PatchCore/patchcore_data/prpd_v2 --out Results/patchcore/baselines_v2

Plans/PatchCore_training_plan.md, Section 6.1. Runs locally (no GPU):
  maha_pooled : log1p feature, class means (PD, Noise) on train, pooled shrinkage covariance (alpha=0.5,
                the Analyzer quality.py setting); s = d2_PD - d2_Noise
  knn5        : train-standardized log1p feature; s = mean dist to 5 nearest PD - to 5 nearest Noise
  majority    : always "Noise" (what the class imbalance alone gives)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import ShrunkCovariance
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from Diffusion.prpd_diffusion.contract import AiDataset  # noqa: E402
from PRPD_Analyzer.prpd_analyzer.features import make_features  # noqa: E402
from eval_metrics import evaluate, file_scores  # noqa: E402


def features_for(dataset: AiDataset, files: pd.DataFrame) -> np.ndarray:
    feats = np.empty((len(files), 256), dtype=np.float32)
    index = {sid: i for i, sid in enumerate(files["sample_id"])}
    meta = dataset.metadata.set_index("sample_id")
    for group, part in files.groupby("group"):
        array = dataset.group_array(str(group))
        for sid in part["sample_id"]:
            m = array[int(meta.at[sid, "tensor_index"])].astype(np.float32)  # (phase, time)
            feats[index[sid]] = make_features(m.mean(axis=1)[None], m.max(axis=1)[None])[0]
        dataset.release()
    return feats


def as_windows(files: pd.DataFrame, s: np.ndarray) -> pd.DataFrame:
    """One pseudo-window per file so eval_metrics.file_scores/evaluate can be reused unchanged."""
    return files.assign(s=s)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    config = json.loads((args.data_root / "run_config.json").read_text())
    files = pd.read_csv(args.data_root / "files.csv", dtype={"date": str})
    dataset = AiDataset.load(Path(config["ai_data_root"]))
    X = np.log1p(features_for(dataset, files).astype(np.float64))
    train = (files["split"] == "train").to_numpy()
    is_noise = (files["cls"] == "Noise").to_numpy()

    results = {}

    # Mahalanobis, pooled shrinkage covariance over the two class-centered train sets.
    means = {c: X[train & (is_noise == (c == "Noise"))].mean(axis=0) for c in ("PD", "Noise")}
    centered = np.vstack([X[train & ~is_noise] - means["PD"], X[train & is_noise] - means["Noise"]])
    precision = ShrunkCovariance(shrinkage=0.5, assume_centered=True).fit(centered).precision_
    d2 = {c: np.einsum("ij,jk,ik->i", X - means[c], precision, X - means[c]) for c in means}
    # The "file score" of a baseline is the single value; the aggregation choice is then trivially the same.
    results["maha_pooled"] = evaluate(file_scores(as_windows(files, d2["PD"] - d2["Noise"])), args.out / "maha_pooled", "256-D Mahalanobis")

    # kNN on train-standardized features.
    mu, sd = X[train].mean(axis=0), X[train].std(axis=0) + 1e-8
    Z = (X - mu) / sd
    dist = {}
    for c, mask in (("PD", train & ~is_noise), ("Noise", train & is_noise)):
        nn = NearestNeighbors(n_neighbors=5).fit(Z[mask])
        d, _ = nn.kneighbors(Z)
        # Train files are in their own reference set; only val/test are evaluated, so this is harmless.
        dist[c] = d.mean(axis=1)
    results["knn5"] = evaluate(file_scores(as_windows(files, dist["PD"] - dist["Noise"])), args.out / "knn5", "256-D kNN-5")

    # Majority class: always Noise.
    majority = {}
    for split in ("val", "test"):
        y = (files.loc[files["split"] == split, "cls"] == "Noise").to_numpy(int)
        pred = np.ones_like(y)
        majority[split] = {"auroc": 0.5, "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
                           "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "accuracy": float(y.mean())}

    summary = {
        name: {"aggregation": r["aggregation"], "val_auroc": r["val"]["auroc"], "test_auroc": r["test"]["auroc"],
               "test_macro_f1": r["test"]["macro_f1"], "test_balanced_accuracy": r["test"]["balanced_accuracy"]}
        for name, r in results.items()
    }
    summary["majority_noise"] = majority
    summary["ai_data_root"] = config["ai_data_root"]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "baselines.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

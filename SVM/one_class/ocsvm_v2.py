"""Two One-Class SVMs (PD model, Noise model) on the PatchCore v2 data — PD vs Noise.

    python SVM/one_class/ocsvm_v2.py --out Results/svm/ocsvm_v2_s42 --seed 42
    python Comparison/evaluate_two_model.py --run Results/svm/ocsvm_v2_s42 --title "One-Class SVM"

Model and features are unchanged from the earlier baseline (evaluate_svm.py): RBF One-Class SVM,
nu=0.01, StandardScaler, and the Training_SVM.py feature per phase bin
[max over time, max / (non-zero count)] -> 256-D. What changes is the data: features are computed
per 128x128 v2 window, trained on the same K windows/file as PatchCore, and every val/test window is
scored. score(window) = anomaly under the PD model - anomaly under the Noise model (> 0: Noise-like).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Comparison"))
from v2_data import WINDOW_KEYS, eval_windows, load_manifest, load_window, pick_train_windows  # noqa: E402


def window_features(paths) -> np.ndarray:
    """Training_SVM.py feature on each (128 phase, T) window: [max_t X(p,t), max_t / nnz_t] per phase."""
    out = np.empty((len(paths), 256), dtype=np.float64)
    for i, path in enumerate(paths):
        w = load_window(path).astype(np.float64)
        mx = w.max(axis=1)
        out[i, :128] = mx
        out[i, 128:] = mx / (np.count_nonzero(w, axis=1) + 1e-15)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=4, help="train windows per file (same as PatchCore)")
    p.add_argument("--nu", type=float, default=0.01)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.time()

    manifest, data_config = load_manifest()
    evals = eval_windows(manifest)
    X_eval = window_features(evals["image_path"])
    info = {"method": "one_class_svm", "created_at": datetime.now().astimezone().isoformat(),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "feature": "per phase: max over window, max/(nonzero count) -> 256-D (Training_SVM.py)",
            "ai_data_root": data_config["ai_data_root"], "raw_data_version": data_config["raw_data_version"],
            "split_seed": data_config["split_seed"], "data_run_config_sha256": data_config["run_config_sha256"],
            "sklearn": sklearn.__version__, "models": {}}

    for cls in ("pd", "noise"):
        train = pick_train_windows(manifest, cls, args.k, args.seed)
        X_train = window_features(train["image_path"])
        scaler = StandardScaler().fit(X_train)
        model = OneClassSVM(kernel="rbf", nu=args.nu).fit(scaler.transform(X_train))
        with (args.out / f"model_{cls}.pkl").open("wb") as f:
            pickle.dump({"scaler": scaler, "svm": model}, f)
        out = evals[WINDOW_KEYS].copy()
        out[f"d_{cls}"] = -model.score_samples(scaler.transform(X_eval))  # higher = more anomalous for this class
        out.to_csv(args.out / f"window_scores_{cls}.csv", index=False)
        info["models"][cls] = {"train_files": int(train["sample_id"].nunique()), "train_windows": int(len(train)),
                               "support_vectors": int(model.support_vectors_.shape[0])}
        print(f"[{cls}] {info['models'][cls]}", flush=True)

    info["seconds"] = round(time.time() - started, 1)
    (args.out / "model_info.json").write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

"""Evaluate any two-model PD-vs-Noise run (PatchCore, EfficientAD, One-Class SVM) the same way.

    python Comparison/evaluate_two_model.py --run Results/efficientad/effad_v2_s42 --title EfficientAD

Needs <run>/window_scores_pd.csv (column d_pd) and window_scores_noise.csv (d_noise): the anomaly of each
val/test window under the PD model and the Noise model. s = d_pd - d_noise (> 0: Noise-like).

Reported (plan Section 6, same protocol for all methods):
  - file-level AUROC, macro F1, Noise-class F1, balanced accuracy — aggregation (mean/median/max/min/p90)
    and threshold chosen on val, applied unchanged to test
  - window-level AUROC: every 128x128 window labelled with its file's class. This replaces pixel-wise
    AUROC, which cannot be measured (the data has no pixel-level ground truth).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "PatchCore"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_metrics import evaluate, file_scores  # noqa: E402
from v2_data import WINDOW_KEYS  # noqa: E402


def load_windows(run: Path) -> pd.DataFrame:
    a = pd.read_csv(run / "window_scores_pd.csv", dtype={"date": str})
    b = pd.read_csv(run / "window_scores_noise.csv", dtype={"date": str})
    w = a.merge(b, on=WINDOW_KEYS, how="inner", validate="one_to_one")
    if not (len(w) == len(a) == len(b)):
        raise SystemExit("the two score files cover different windows")
    w["s"] = w["d_pd"] - w["d_noise"]
    return w


def window_auroc(windows: pd.DataFrame, split: str) -> float:
    part = windows[windows["split"] == split]
    return float(roc_auc_score((part["cls"] == "Noise").astype(int), part["s"]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--out", type=Path, default=None, help="default: <run>/eval")
    args = p.parse_args()
    out = args.out or args.run / "eval"

    windows = load_windows(args.run)
    if not {"val", "test"} <= set(windows["split"]):
        raise SystemExit("run must contain val and test windows")
    result = evaluate(file_scores(windows, "s"), out, args.title)
    t = result["test"]
    summary = {
        "method": args.title, "run": str(args.run), "aggregation": result["aggregation"],
        "threshold": result["threshold"],
        "val_file_auroc": result["val"]["auroc"],
        "test_file_auroc": t["auroc"],
        "test_window_auroc": window_auroc(windows, "test"),
        "val_window_auroc": window_auroc(windows, "val"),
        "test_macro_f1": t["macro_f1"],
        "test_noise_f1": t["per_class"]["Noise"]["f1"],
        "test_pd_f1": t["per_class"]["PD"]["f1"],
        "test_balanced_accuracy": t["balanced_accuracy"],
        "test_pd_recall": t["per_class"]["PD"]["recall"],
        "test_noise_recall": t["per_class"]["Noise"]["recall"],
        "pixel_auroc": None,
        "pixel_auroc_note": "not measurable: no pixel-level ground truth; window-level AUROC reported instead",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()

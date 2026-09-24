"""Torch-free evaluation for the PD-vs-Noise task (Plans/PatchCore_training_plan.md, Sections 4.3-6).

Convention: positive class = Noise, and a larger score means "more Noise-like".
Aggregation and threshold are chosen on val only, then applied unchanged to test.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve,
)

AGGREGATIONS = {
    "mean": lambda s: s.mean(),
    "median": lambda s: s.median(),
    "max": lambda s: s.max(),
    "min": lambda s: s.min(),
    "p90": lambda s: s.quantile(0.9),
}
FILE_COLS = ["sample_id", "split", "cls", "label", "group", "domain", "date"]


def file_scores(windows: pd.DataFrame, score_col: str = "s") -> pd.DataFrame:
    """Aggregate window scores to one row per file, one column per aggregation (S_mean, S_max, ...)."""
    grouped = windows.groupby(FILE_COLS, sort=False)[score_col]
    out = pd.DataFrame({f"S_{name}": grouped.agg(fn) for name, fn in AGGREGATIONS.items()}).reset_index()
    return out


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Threshold maximizing macro F1 (midpoints between sorted unique scores)."""
    values = np.unique(score)
    candidates = np.concatenate([[values[0] - 1e-9], (values[:-1] + values[1:]) / 2, [values[-1] + 1e-9]])
    best_t, best_f1 = candidates[0], -1.0
    for t in candidates:
        f1 = f1_score(y, (score > t).astype(int), average="macro", zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return float(best_t)


def metrics_at(y: np.ndarray, score: np.ndarray, tau: float) -> dict:
    pred = (score > tau).astype(int)
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    return {
        "n_files": int(len(y)), "n_pd": int((y == 0).sum()), "n_noise": int((y == 1).sum()),
        "auroc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None,
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float((pred == y).mean()),
        "per_class": {
            "PD": {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f[0]), "support": int(n[0])},
            "Noise": {"precision": float(p[1]), "recall": float(r[1]), "f1": float(f[1]), "support": int(n[1])},
        },
        "confusion_matrix": {"rows_true_[PD,Noise]": confusion_matrix(y, pred, labels=[0, 1]).tolist()},
        "threshold": tau,
    }


def select_on_val(files: pd.DataFrame) -> tuple[str, float, dict]:
    """Pick the aggregation with the best val AUROC, then the macro-F1 threshold for it."""
    val = files[files["split"] == "val"]
    y = (val["cls"] == "Noise").to_numpy(int)
    table = {name: float(roc_auc_score(y, val[f"S_{name}"])) for name in AGGREGATIONS}
    chosen = max(table, key=table.get)
    tau = best_threshold(y, val[f"S_{chosen}"].to_numpy())
    return chosen, tau, {"val_auroc_by_aggregation": table, "val_metrics": metrics_at(y, val[f"S_{chosen}"].to_numpy(), tau)}


def breakdowns(test: pd.DataFrame, col: str, tau: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = np.where(test[col] > tau, "Noise", "PD")
    t = test.assign(pred=pred, correct=(pred == test["cls"]))
    per_label = t.groupby(["cls", "label"]).agg(files=("sample_id", "size"), correct=("correct", "sum"))
    per_label["recall"] = per_label["correct"] / per_label["files"]
    per_date = t.groupby("date").agg(files=("sample_id", "size"), n_pd=("cls", lambda c: int((c == "PD").sum())),
                                     n_noise=("cls", lambda c: int((c == "Noise").sum())), correct=("correct", "sum"))
    per_date["accuracy"] = per_date["correct"] / per_date["files"]
    return per_label.reset_index(), per_date.reset_index().sort_values("files", ascending=False)


def plots(files: pd.DataFrame, col: str, tau: float, out: Path, title: str) -> None:
    for split in ("val", "test"):
        part = files[files["split"] == split]
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for cls, color in (("PD", "#1f77b4"), ("Noise", "#d62728")):
            ax.hist(part.loc[part["cls"] == cls, col], bins=40, alpha=0.6, label=f"{cls} (n={int((part['cls'] == cls).sum())})", color=color)
        ax.axvline(tau, color="k", ls="--", lw=1, label="threshold (val)")
        ax.set_xlabel(f"file score {col}  (> threshold = Noise)"); ax.set_ylabel("files")
        ax.set_title(f"{title}: {split}"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(out / f"score_hist_{split}.png", dpi=150); plt.close(fig)

    test = files[files["split"] == "test"]
    y = (test["cls"] == "Noise").to_numpy(int)
    fpr, tpr, _ = roc_curve(y, test[col])
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.plot(fpr, tpr, label=f"AUROC {roc_auc_score(y, test[col]):.3f}")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("FPR (PD called Noise)"); ax.set_ylabel("TPR (Noise called Noise)"); ax.set_title(f"{title}: test ROC")
    ax.legend(loc="lower right"); fig.tight_layout(); fig.savefig(out / "roc_test.png", dpi=150); plt.close(fig)

    cm = confusion_matrix(y, (test[col] > tau).astype(int), labels=[0, 1])
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["PD", "Noise"]); ax.set_yticks([0, 1], ["PD", "Noise"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"{title}: test")
    fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=150); plt.close(fig)


def evaluate(files: pd.DataFrame, out: Path, title: str, extra: dict | None = None, val_only: bool = False) -> dict:
    """Full protocol: choose aggregation+threshold on val, score test once, write everything to `out`.

    val_only=True is for model-selection runs: nothing about test is computed or written.
    """
    out.mkdir(parents=True, exist_ok=True)
    chosen, tau, selection = select_on_val(files)
    col = f"S_{chosen}"
    if val_only:
        files = files[files["split"] == "val"]
        per_label, per_date = breakdowns(files, col, tau)
        files.to_csv(out / "file_scores.csv", index=False)
        per_label.to_csv(out / "val_per_label.csv", index=False)
        per_date.to_csv(out / "val_per_date.csv", index=False)
        (out / "val_selection.json").write_text(json.dumps({"aggregation": chosen, "threshold": tau, **selection, **(extra or {})}, indent=2))
        return {"aggregation": chosen, "threshold": tau, "val": selection["val_metrics"],
                "val_auroc_by_aggregation": selection["val_auroc_by_aggregation"]}
    test = files[files["split"] == "test"]
    y = (test["cls"] == "Noise").to_numpy(int)
    test_metrics = metrics_at(y, test[col].to_numpy(), tau)
    test_metrics["test_auroc_by_aggregation"] = {name: float(roc_auc_score(y, test[f"S_{name}"])) for name in AGGREGATIONS}

    per_label, per_date = breakdowns(test, col, tau)
    relabelled = files[(files["group"].str.endswith("PD")) & (files["cls"] == "Noise") & files["split"].isin(["val", "test"])]
    relabelled = relabelled.assign(pred=np.where(relabelled[col] > tau, "Noise", "PD"))[["sample_id", "split", "date", col, "pred"]]

    files.to_csv(out / "file_scores.csv", index=False)
    per_label.to_csv(out / "per_label.csv", index=False)
    per_date.to_csv(out / "per_date.csv", index=False)
    relabelled.to_csv(out / "relabelled_files.csv", index=False)
    (out / "val_selection.json").write_text(json.dumps({"aggregation": chosen, "threshold": tau, **selection}, indent=2))
    (out / "metrics_test.json").write_text(json.dumps({"aggregation": chosen, **test_metrics, **(extra or {})}, indent=2))
    plots(files, col, tau, out, title)
    return {"aggregation": chosen, "threshold": tau, "val": selection["val_metrics"], "test": test_metrics}

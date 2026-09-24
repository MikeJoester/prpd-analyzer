"""Collect val results of PatchCore v2 runs into one table (Plans/PatchCore_training_plan.md, Section 5).

    python PatchCore/summarize_runs.py Results/patchcore/v2_start_s42 Results/patchcore/e1_* ... --out Results/patchcore/v2_val_table.csv

Reads only val_selection.json and bank_info.json, so it never touches test results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def row(run: Path) -> dict:
    sel = json.loads((run / "val_selection.json").read_text())
    pd_bank = json.loads((run / "bank_pd" / "bank_info.json").read_text())
    noise_bank = json.loads((run / "bank_noise" / "bank_info.json").read_text())
    s, a, m = pd_bank["settings"], pd_bank["args"], sel["val_metrics"]
    single = {}
    for name in ("pd_bank_only", "noise_bank_only"):
        f = run / "reference" / name / "val_selection.json"
        if f.is_file():
            single[name] = json.loads(f.read_text())["val_metrics"]["auroc"]
    return {
        "run": run.name, "backbone": s["backbone"], "layers": "+".join(s["layers"]), "target_dim": s["target_dim"],
        "coreset": pd_bank["coreset_ratio"], "k": a["k"], "seed": a["seed"],
        "bank_pd": pd_bank["memory_bank_size"], "bank_noise": noise_bank["memory_bank_size"],
        "train_min": round((pd_bank["seconds"] + noise_bank["seconds"]) / 60, 1),
        "aggregation": sel["aggregation"], "val_auroc": m["auroc"], "val_macro_f1": m["macro_f1"],
        "val_bal_acc": m["balanced_accuracy"], "val_pd_recall": m["per_class"]["PD"]["recall"],
        "val_noise_recall": m["per_class"]["Noise"]["recall"],
        **{f"val_auroc_{k}": v for k, v in sel["val_auroc_by_aggregation"].items()},
        **{f"val_auroc_{k}": v for k, v in single.items()},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    table = pd.DataFrame([row(r) for r in args.runs if (r / "val_selection.json").is_file()])
    table.to_csv(args.out, index=False)
    cols = ["run", "backbone", "layers", "coreset", "k", "seed", "bank_pd", "train_min", "aggregation",
            "val_auroc", "val_macro_f1", "val_bal_acc", "val_pd_recall", "val_noise_recall"]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(table[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()

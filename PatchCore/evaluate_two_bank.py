"""Combine the two banks' window scores and evaluate (Plans/PatchCore_training_plan.md, Sections 4.3-6).

    python PatchCore/evaluate_two_bank.py --run $RUN

s(window) = d_pd - d_noise  (> 0: closer to the Noise bank). Aggregation and threshold are
chosen on val; test is scored once. Also reports each single bank on its own for reference.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_metrics import evaluate, file_scores  # noqa: E402

KEYS = ["sample_id", "window_idx", "split", "cls", "label", "group", "domain", "date"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--val-only", action="store_true", help="model-selection runs: report val only, never touch test")
    args = p.parse_args()

    pd_scores = pd.read_csv(args.run / "window_scores_pd.csv", dtype={"date": str})
    noise_scores = pd.read_csv(args.run / "window_scores_noise.csv", dtype={"date": str})
    windows = pd_scores.merge(noise_scores, on=KEYS, how="inner", validate="one_to_one")
    if len(windows) != len(pd_scores) or len(windows) != len(noise_scores):
        raise SystemExit("the two score files cover different windows")
    windows["s"] = windows["d_pd"] - windows["d_noise"]
    windows.to_csv(args.run / "window_scores.csv", index=False)

    banks = {c: json.loads((args.run / f"bank_{c}" / "bank_info.json").read_text()) for c in ("pd", "noise")}
    extra = {"banks": {c: {k: banks[c][k] for k in ("train_files", "train_windows", "memory_bank_size", "settings", "coreset_ratio")}
                       for c in banks},
             "ai_data_root": banks["pd"]["ai_data_root"], "raw_data_version": banks["pd"]["raw_data_version"]}
    files = file_scores(windows, "s")
    if args.val_only:
        if (files["split"] == "test").any():
            raise SystemExit("--val-only run has test scores; score with --splits val")
        result = evaluate(files, args.run, "PatchCore two-bank", extra, val_only=True)
        single = {}
        for col, sign, name in (("d_pd", 1, "pd_bank_only"), ("d_noise", -1, "noise_bank_only")):
            single[name] = evaluate(file_scores(windows.assign(s=sign * windows[col]), "s"),
                                    args.run / "reference" / name, name, val_only=True)["val"]["auroc"]
        v = result["val"]
        print(json.dumps({"aggregation": result["aggregation"], "val_auroc": round(v["auroc"], 4),
                          "val_macro_f1": round(v["macro_f1"], 4), "val_balanced_acc": round(v["balanced_accuracy"], 4),
                          "val_auroc_by_aggregation": {k: round(x, 4) for k, x in result["val_auroc_by_aggregation"].items()},
                          "single_bank_val_auroc": {k: round(x, 4) for k, x in single.items()}}, indent=2))
        return
    result = evaluate(files, args.run, "PatchCore two-bank", extra)

    # Reference only: each bank alone (old single-bank style). Not used for any selection.
    reference = {}
    for col, sign, name in (("d_pd", 1, "pd_bank_only"), ("d_noise", -1, "noise_bank_only")):
        single = windows.assign(s=sign * windows[col])
        reference[name] = evaluate(file_scores(single, "s"), args.run / "reference" / name, name)["test"]["auroc"]
    (args.run / "reference" / "single_bank_test_auroc.json").write_text(json.dumps(reference, indent=2))

    t = result["test"]
    print(json.dumps({"aggregation": result["aggregation"], "threshold": round(result["threshold"], 4),
                      "val_auroc": round(result["val"]["auroc"], 4), "test_auroc": round(t["auroc"], 4),
                      "test_macro_f1": round(t["macro_f1"], 4), "test_balanced_acc": round(t["balanced_accuracy"], 4),
                      "single_bank_test_auroc": {k: round(v, 4) for k, v in reference.items()}}, indent=2))


if __name__ == "__main__":
    main()

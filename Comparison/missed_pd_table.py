"""Build the per-file table of PatchCore's missed PD cases (seed 42, validation-selected threshold).

    python Comparison/missed_pd_table.py

Writes Results/patchcore/error_analysis/missed18/missed_pd_cases.csv — one row per missed PD file, joining:
  decision      score, margin to the threshold, and how many of the 10 training seeds miss the file
  phi-q-n       pulse counts, amplitudes, half-cycle correlation and asymmetry (Comparison/error_figures.py)
  bank audit    Lab/Field origin of the matched training windows and the distance to each bank (nn_audit.py)
  label quality the Analyzer suspect score (prpd_analyzer.quality)
  windows       how many of the file's 28 windows individually score on the PD side
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "PatchCore"))
from eval_metrics import best_threshold  # noqa: E402

EA = REPO / "Results/patchcore/error_analysis"
OUT = EA / "missed18"
SEEDS = tuple(range(42, 52))
REF_SEED = 42


def decide(seed: int) -> tuple[pd.DataFrame, float]:
    d = REPO / f"Results/patchcore/final_k8_s{seed}/eval"
    f = pd.read_csv(d / "file_scores.csv", dtype={"date": str})
    col = "S_" + json.loads((d / "val_selection.json").read_text())["aggregation"]
    val, test = f[f.split == "val"], f[f.split == "test"].copy()
    tau = best_threshold((val.cls == "Noise").to_numpy(int), val[col].to_numpy())
    test["score"] = test[col]
    test["margin"] = test[col] - tau
    test["pred"] = np.where(test[col] > tau, "Noise", "PD")
    test["err"] = test.pred != test.cls
    return test.set_index("sample_id"), tau


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=OUT / "missed_pd_cases.csv")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    base, tau = decide(REF_SEED)
    m = base[(base.cls == "PD") & base.err].copy()
    flags = pd.DataFrame({s: decide(s)[0].err for s in SEEDS})
    m["missed_in_n_of_10"] = flags.loc[m.index].sum(axis=1).astype(int)
    m = m.reset_index()

    meta = pd.read_parquet(REPO / "artifacts/ai_data_20260911_011704/metadata.parquet")[["sample_id", "file"]]
    m = m.merge(meta, on="sample_id", how="left")

    sus = pd.read_csv(EA / "suspect_cross_check.csv", dtype={"date": str})
    m = m.merge(sus[["sample_id", "suspect_score", "cell_percentile", "frac_neighbours_same_class"]],
                on="sample_id", how="left")

    q = pd.read_csv(EA / "phi_q_n_stats.csv")
    m = m.merge(q[["sample_id", "total_pulses", "pos_pulse_count", "neg_pulse_count", "pos_max_amplitude",
                   "neg_max_amplitude", "half_cycle_cross_correlation", "count_asymmetry"]],
                on="sample_id", how="left")

    nn = pd.read_csv(EA / "nn_audit.csv")
    m = m.merge(nn[["sample_id", "pd_nn_domain_Lab_frac", "pd_nn_top_label", "pd_nn_dist", "noise_nn_dist"]],
                on="sample_id", how="left")
    m["bank_gap"] = m.noise_nn_dist - m.pd_nn_dist

    w = pd.read_csv(REPO / f"Results/patchcore/final_k8_s{REF_SEED}/window_scores.csv", dtype={"date": str})
    w = w[(w.split == "test") & w.sample_id.isin(m.sample_id)]
    g = w.groupby("sample_id").s
    m = m.merge(pd.DataFrame({"windows_pd_like": g.apply(lambda x: int((x < tau).sum())),
                              "best_window_score": g.min(), "worst_window_score": g.max()}).reset_index(),
                on="sample_id")

    # the three failure groups identified in the report
    m["failure_group"] = np.where(m.date == "20220427", "A_textbook_void",
                          np.where(m.date == "20231023", "B_label_quality", "C_sparse"))
    m = m.sort_values(["date", "margin"]).reset_index(drop=True)
    m.insert(0, "case", m.index + 1)

    cols = ["case", "sample_id", "file", "label", "group", "date", "failure_group", "missed_in_n_of_10",
            "score", "margin", "windows_pd_like", "best_window_score", "worst_window_score",
            "total_pulses", "pos_pulse_count", "neg_pulse_count", "pos_max_amplitude", "neg_max_amplitude",
            "half_cycle_cross_correlation", "count_asymmetry", "suspect_score", "cell_percentile",
            "frac_neighbours_same_class", "pd_nn_domain_Lab_frac", "pd_nn_top_label", "pd_nn_dist",
            "noise_nn_dist", "bank_gap"]
    m[cols].to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(m)} rows, threshold {tau:+.4f}, reference seed {REF_SEED})")

    summary = m.groupby("failure_group").agg(
        files=("case", "size"), missed_seeds=("missed_in_n_of_10", "median"),
        pulses=("total_pulses", "median"), suspect=("suspect_score", "median"),
        lab_match=("pd_nn_domain_Lab_frac", "median"), bank_gap=("bank_gap", "median")).round(3)
    summary.to_csv(args.out.parent / "missed_pd_groups.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()

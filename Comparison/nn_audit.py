"""Analysis E: what training windows did PatchCore actually match each test file to?

    (on the server)  python Comparison/nn_audit.py --run Results/patchcore/nnaudit_k8_s42

For every test file's most extreme window, query both memory banks for their top-k nearest entries and
map those entries back to the training window they came from (`bank_entry_source.csv`, written by
train_two_bank.py). Answers H4: are missed PD files matching Lab windows rather than Field ones?

Output: <run>/nn_audit.csv  (one row per test file x bank)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "PatchCore"))
from two_bank_common import WindowDataset, load_bank, load_data_root, make_loader  # noqa: E402

DATA = REPO_ROOT / "PatchCore/patchcore_data/prpd_v2"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--k", type=int, default=5, help="neighbours per query patch")
    p.add_argument("--errors-only", action="store_true", help="audit only files PatchCore got wrong")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    manifest, _ = load_data_root(DATA)
    scores = pd.read_csv(REPO_ROOT / "Results/patchcore/final_k8_s42/window_scores.csv", dtype={"date": str})
    table = pd.read_csv(REPO_ROOT / "Results/patchcore/error_analysis/error_table.csv", dtype={"date": str})
    test = scores[scores.split == "test"]
    if args.errors_only:
        table = table[table.error_kind != "correct"]
    wanted = set(table.sample_id)

    # the window that drove the decision: the most Noise-like for false alarms, most PD-like otherwise
    picks = []
    for sid, g in test[test.sample_id.isin(wanted)].groupby("sample_id"):
        kind = table.loc[table.sample_id == sid, "error_kind"].iloc[0]
        row = g.loc[g.s.idxmax()] if kind == "false alarm" else g.loc[g.s.idxmin()]
        picks.append({"sample_id": sid, "window_idx": int(row.window_idx), "cls": row.cls,
                      "label": row.label, "date": row.date, "error_kind": kind})
    picks = pd.DataFrame(picks)
    rows = picks.merge(manifest, on=["sample_id", "window_idx", "cls", "label", "date"], how="left")
    print(f"auditing {len(rows)} windows")

    device = torch.device("cuda:0")
    out = picks.copy()
    for cls in ("pd", "noise"):
        bank_dir = args.run / f"bank_{cls}"
        src = pd.read_csv(bank_dir / "bank_entry_source.csv")
        model = load_bank(bank_dir, device)
        loader = make_loader(WindowDataset(DATA, rows, 224), 16, 4)
        feats = []
        for images in loader:
            with torch.no_grad():
                feats.append(np.asarray(model._embed(images.to(torch.float).to(device))))
        feats = np.concatenate(feats)
        per_window = len(feats) // len(rows)
        dists, idx = model.anomaly_scorer.nn_method.search_index.search(feats.astype("float32"), args.k)
        # per window: the patch with the largest nearest-neighbour distance decides the window score
        worst = (dists[:, 0].reshape(len(rows), per_window)).argmax(axis=1)
        flat = np.arange(len(rows)) * per_window + worst
        nb = src.iloc[idx[flat].ravel()].reset_index(drop=True)
        nb["query"] = np.repeat(np.arange(len(rows)), args.k)
        agg = nb.groupby("query").agg(
            **{f"{cls}_nn_domain_Lab_frac": ("domain", lambda d: float((d == "Lab").mean())),
               f"{cls}_nn_top_label": ("label", lambda l: l.mode().iat[0]),
               f"{cls}_nn_top_sample": ("sample_id", lambda s: s.iat[0])})
        out = pd.concat([out.reset_index(drop=True), agg.reset_index(drop=True)], axis=1)
        out[f"{cls}_nn_dist"] = dists[flat, 0]
        del model
        torch.cuda.empty_cache()

    path = args.out or args.run / "nn_audit.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}")
    print(out.groupby("error_kind")[["pd_nn_domain_Lab_frac", "pd_nn_dist", "noise_nn_dist"]].median().round(3).to_string())


if __name__ == "__main__":
    main()

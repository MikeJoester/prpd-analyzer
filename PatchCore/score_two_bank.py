"""Score val/test windows against one bank (run once per bank, e.g. on separate GPUs).

    CUDA_VISIBLE_DEVICES=0 python PatchCore/score_two_bank.py --run $RUN --cls pd    --data-root PatchCore/patchcore_data/prpd_v2
    CUDA_VISIBLE_DEVICES=1 python PatchCore/score_two_bank.py --run $RUN --cls noise --data-root PatchCore/patchcore_data/prpd_v2

Writes <run>/window_scores_<cls>.csv with the distance of every window to that bank.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from two_bank_common import WindowDataset, load_bank, load_data_root, make_loader, sha256  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--bank-run", type=Path, default=None,
                   help="reuse banks from another run (num-nn variants need no rebuild)")
    p.add_argument("--num-nn", type=int, default=None,
                   help="override the bank's k: the score becomes the mean distance to the k nearest entries")
    p.add_argument("--cls", choices=["pd", "noise"], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--max-files", type=int, default=None, help="smoke runs: score only this many files per split")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    bank_dir = (args.bank_run or args.run) / f"bank_{args.cls}"
    info = json.loads((bank_dir / "bank_info.json").read_text())
    manifest, data_config = load_data_root(args.data_root)
    # Refuse to score with a bank that was built from a different dataset (CLAUDE.md, Section 11).
    for key in ("ai_data_root", "raw_data_version", "split_seed"):
        if info[key] != data_config[key]:
            raise SystemExit(f"bank/data mismatch on {key}: {info[key]} vs {data_config[key]}")
    if info["data_run_config_sha256"] != sha256(Path(args.data_root) / "run_config.json"):
        raise SystemExit("bank was built from a different prpd_v2 build (run_config.json hash differs)")

    rows = manifest[manifest["split"].isin(args.splits)]
    if args.max_files:
        keep = (rows.drop_duplicates("sample_id").groupby(["split", "cls"], group_keys=False)
                .apply(lambda g: g.head(args.max_files // 2 or 1))["sample_id"])
        rows = rows[rows["sample_id"].isin(keep)]
    rows = rows.reset_index(drop=True)

    device = torch.device("cuda:0")
    model = load_bank(bank_dir, device)
    if args.num_nn is not None:
        # the k is captured in a closure at construction, so rebinding both is required
        scorer = model.anomaly_scorer
        scorer.n_nearest_neighbours = args.num_nn
        scorer.imagelevel_nn = lambda q, k=args.num_nn: scorer.nn_method.run(k, q)
        info = {**info, "settings": {**info["settings"], "num_nn": args.num_nn}}
    if args.bank_run is not None:            # keep the run self-describing for evaluate_two_bank
        target = args.run / f"bank_{args.cls}"
        target.mkdir(parents=True, exist_ok=True)
        (target / "bank_info.json").write_text(json.dumps(
            {**info, "reused_bank_from": str(args.bank_run)}, indent=2))
    loader = make_loader(WindowDataset(args.data_root, rows, info["settings"]["image_size"]), args.batch_size, args.workers)
    print(f"[{args.cls}] scoring {rows['sample_id'].nunique()} files / {len(rows)} windows", flush=True)
    scores = np.concatenate([np.asarray(model._predict(batch)[0], dtype=np.float64) for batch in loader])

    out = rows[["sample_id", "window_idx", "split", "cls", "label", "group", "domain", "date"]].copy()
    out[f"d_{args.cls}"] = scores
    path = args.run / f"window_scores_{args.cls}.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path} in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()

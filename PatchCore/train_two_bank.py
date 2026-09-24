"""Build one PatchCore memory bank (PD or Noise) from the v2 train split.

    CUDA_VISIBLE_DEVICES=0 python PatchCore/train_two_bank.py --cls pd    --data-root PatchCore/patchcore_data/prpd_v2 --out $RUN
    CUDA_VISIBLE_DEVICES=1 python PatchCore/train_two_bank.py --cls noise --data-root PatchCore/patchcore_data/prpd_v2 --out $RUN

Each train file contributes K windows (seeded choice from its 28). Plans/PatchCore_training_plan.md, Section 3-4.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from two_bank_common import (  # noqa: E402
    ChunkedApproxGreedyCoreset, WindowDataset, build_patchcore, fill_memory_bank_prealloc, load_data_root, make_loader,
    seed_everything, sha256,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cls", choices=["pd", "noise"], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="run folder; the bank goes to <out>/bank_<cls>")
    p.add_argument("--seed", type=int, default=42, help="training seed: window subsample + coreset init (not the split)")
    p.add_argument("--k", type=int, default=4, help="train windows per file")
    p.add_argument("--k-field", type=int, default=None, help="optional K for Field files (defaults to --k)")
    p.add_argument("--backbone", default="wideresnet50")
    p.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--pretrain-dim", type=int, default=1024)
    p.add_argument("--target-dim", type=int, default=1024)
    p.add_argument("--patchsize", type=int, default=3)
    p.add_argument("--num-nn", type=int, default=1)
    p.add_argument("--coreset", type=float, default=0.01)
    p.add_argument("--train-frac", type=float, default=1.0, help="<1 for smoke runs: fraction of train files used")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def pick_windows(rows, k: int, k_field: int, rng: np.random.Generator):
    """Choose k (Lab) / k_field (Field) of the 28 windows of each train file, reproducibly."""
    chosen = []
    for _, part in rows.groupby("sample_id", sort=True):
        n = k_field if part["domain"].iloc[0] == "Field" else k
        chosen.append(part.iloc[np.sort(rng.choice(len(part), size=min(n, len(part)), replace=False))])
    return pd.concat(chosen)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    started = time.time()

    manifest, data_config = load_data_root(args.data_root)
    train = manifest[(manifest["split"] == "train") & (manifest["cls"].str.lower() == args.cls)]
    rng = np.random.default_rng(args.seed)
    if args.train_frac < 1.0:
        files = np.sort(train["sample_id"].unique())
        keep = rng.choice(files, size=max(1, int(len(files) * args.train_frac)), replace=False)
        train = train[train["sample_id"].isin(keep)]
    k_field = args.k_field or args.k
    rows = pick_windows(train, args.k, k_field, rng).reset_index(drop=True)

    settings = {
        "backbone": args.backbone, "layers": args.layers, "image_size": args.image_size,
        "pretrain_dim": args.pretrain_dim, "target_dim": args.target_dim,
        "patchsize": args.patchsize, "num_nn": args.num_nn,
    }
    sampler = ChunkedApproxGreedyCoreset(args.coreset, device)
    model = build_patchcore(settings, device, sampler)

    loader = make_loader(WindowDataset(args.data_root, rows, args.image_size), args.batch_size, args.workers)
    print(f"[{args.cls}] {rows['sample_id'].nunique()} files, {len(rows)} windows -> fitting", flush=True)
    pool_idx, patches_per_window = fill_memory_bank_prealloc(model, loader, len(rows))  # = model.fit(loader), half the RAM

    bank_dir = args.out / f"bank_{args.cls}"
    bank_dir.mkdir(parents=True, exist_ok=False)
    model.save_to_path(str(bank_dir))
    rows[["image_path", "sample_id", "window_idx", "domain", "label"]].to_csv(bank_dir / "train_windows.csv", index=False)
    if pool_idx is not None:
        # Each bank entry came from one patch of one training window: pool row // patches-per-window.
        src = rows.iloc[pool_idx // patches_per_window][["sample_id", "window_idx", "domain", "label"]]
        src.insert(0, "bank_entry", np.arange(len(src)))
        src.to_csv(bank_dir / "bank_entry_source.csv", index=False)

    bank_size = int(model.anomaly_scorer.nn_method.search_index.ntotal)
    info = {
        "cls": args.cls,
        "created_at": datetime.now().astimezone().isoformat(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "settings": settings,
        "coreset_ratio": args.coreset,
        "train_files": int(rows["sample_id"].nunique()),
        "train_files_by_domain": rows.drop_duplicates("sample_id")["domain"].value_counts().to_dict(),
        "train_windows": int(len(rows)),
        "memory_bank_size": bank_size,
        "data_root": str(Path(args.data_root).resolve()),
        "data_run_config_sha256": sha256(Path(args.data_root) / "run_config.json"),
        "ai_data_root": data_config["ai_data_root"],
        "raw_data_version": data_config["raw_data_version"],
        "split_seed": data_config["split_seed"],
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "peak_host_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
        "seconds": round(time.time() - started, 1),
    }
    (bank_dir / "bank_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: info[k] for k in ("train_files", "train_windows", "memory_bank_size", "peak_gpu_mem_gb", "seconds")}), flush=True)


if __name__ == "__main__":
    main()

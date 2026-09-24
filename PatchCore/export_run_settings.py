"""Flatten a PatchCore v2 run's settings into one CSV.

    python PatchCore/export_run_settings.py --run Results/patchcore/final_k8_s42

Writes <run>/training_settings.csv. Everything is read from what the run itself recorded
(bank_*/bank_info.json, val_selection.json, command.txt, git_head.txt) — nothing is hard-coded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROWS = [  # (setting, group, key in bank_info, note)
    ("backbone", "model", ("settings", "backbone"), "ImageNet-pretrained feature extractor"),
    ("feature_layers", "model", ("settings", "layers"), "backbone layers the patch features come from"),
    ("input_size", "model", ("settings", "image_size"), "window resized to this square size (no crop)"),
    ("patchsize", "model", ("settings", "patchsize"), "feature cells pooled into one patch feature"),
    ("pretrain_embed_dim", "model", ("settings", "pretrain_dim"), "feature dim before aggregation"),
    ("target_embed_dim", "model", ("settings", "target_dim"), "feature dim stored in the memory bank"),
    ("num_nearest_neighbours", "model", ("settings", "num_nn"), "k in the kNN distance used as the score"),
    ("coreset_ratio", "training", ("coreset_ratio",), "fraction of patch features kept (approx. greedy coreset)"),
    ("train_windows_per_file_K", "training", ("args", "k"), "windows sampled per training file"),
    ("train_windows_per_file_K_field", "training", ("args", "k_field"), "K for Field files (None = same as K)"),
    ("seed", "training", ("args", "seed"), "window subsample + coreset init (not the data split)"),
    ("batch_size", "training", ("args", "batch_size"), "feature-extraction batch size"),
    ("train_files", "data", ("train_files",), "files contributing to this bank"),
    ("train_files_by_domain", "data", ("train_files_by_domain",), "Lab / Field split of those files"),
    ("train_windows", "data", ("train_windows",), "windows = files x K"),
    ("memory_bank_size", "result", ("memory_bank_size",), "patch features kept after the coreset"),
    ("build_seconds", "result", ("seconds",), "wall-clock build time"),
    ("peak_gpu_mem_gb", "result", ("peak_gpu_mem_gb",), "peak GPU memory"),
    ("peak_host_rss_gb", "result", ("peak_host_rss_gb",), "peak host RAM"),
    ("data_root", "data", ("data_root",), "v2 window dataset"),
    ("ai_data_root", "data", ("ai_data_root",), "source ai_data run"),
    ("raw_data_version", "data", ("raw_data_version",), "axis convention of the raw tensors"),
    ("split_seed", "data", ("split_seed",), "date-split seed of the dataset build"),
    ("data_run_config_sha256", "data", ("data_run_config_sha256",), "checksum tying the run to that build"),
    ("torch", "environment", ("torch",), ""),
    ("cuda", "environment", ("cuda",), ""),
    ("gpu", "environment", ("gpu",), ""),
    ("python", "environment", ("python",), ""),
    ("created_at", "environment", ("created_at",), ""),
]


def dig(info: dict, keys: tuple):
    for k in keys:
        info = info.get(k) if isinstance(info, dict) else None
        if info is None:
            return None
    return info


def fmt(v):
    if isinstance(v, (list, dict)):
        return json.dumps(v, separators=(",", " "))
    return v


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="default: <run>/training_settings.csv")
    args = p.parse_args()
    banks = {c: json.loads((args.run / f"bank_{c}" / "bank_info.json").read_text()) for c in ("pd", "noise")}

    rows = [{"setting": name, "group": group, "pd_bank": fmt(dig(banks["pd"], keys)),
             "noise_bank": fmt(dig(banks["noise"], keys)), "note": note} for name, group, keys, note in ROWS]

    sel = args.run / "val_selection.json"
    if sel.is_file():
        s = json.loads(sel.read_text())
        rows += [{"setting": "window_aggregation", "group": "scoring", "pd_bank": s["aggregation"],
                  "noise_bank": s["aggregation"], "note": "window->file score, chosen on val"},
                 {"setting": "decision_threshold", "group": "scoring", "pd_bank": round(s["threshold"], 6),
                  "noise_bank": round(s["threshold"], 6), "note": "s = d_pd - d_noise > threshold => Noise; chosen on val"}]
    for label, path in (("run_command", "command.txt"), ("git_head", "git_head.txt")):
        f = args.run / path
        if f.is_file():
            rows.append({"setting": label, "group": "provenance", "pd_bank": f.read_text().strip(),
                         "noise_bank": "", "note": ""})

    out = args.out or args.run / "training_settings.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}")
    print(pd.DataFrame(rows)[["group", "setting", "pd_bank", "noise_bank"]].to_string(index=False))


if __name__ == "__main__":
    main()

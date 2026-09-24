"""Build the PatchCore v2 dataset (Plans/PatchCore_data_plan.md).

    python PatchCore/build_prpd_v2.py --ai-data-root artifacts/ai_data_20260911_011704

Train = all Lab PD/Noise + ~70% Field PD/Noise; val/test = ~15% Field each.
Field is split by measurement date (a date belongs to exactly one split), class
comes from the `label` column (PD vs Noise), and every file is tiled into
non-overlapping 128x128 windows with raw uint8 values (no per-image rescale).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from Diffusion.prpd_diffusion.contract import AiDataset  # noqa: E402

SPLITS = ("train", "val", "test")
FIELD_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
CLASSES = ("PD", "Noise")
# val/test must contain these Field PD labels (Corona only if it can be arranged).
REQUIRED_LABELS = ("Floating", "Void")
UNKNOWN_DATE = "unknown_date"  # Analyzer value when no date could be parsed from the path
PREFERRED_LABELS = ("Corona",)


class SplitError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ai-data-root", type=Path, required=True, help="ai_data_* folder (no default on purpose)")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "PatchCore" / "patchcore_data" / "prpd_v2")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--exclude-groups", nargs="*", default=["Synthetic"])
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


# ------------------------------------------------------------------ table

def build_file_table(dataset: AiDataset, exclude_groups: list[str]) -> pd.DataFrame:
    table = dataset.metadata.copy()
    table = table[~table["group"].isin(exclude_groups)].reset_index(drop=True)
    table["domain"] = np.where(table["group"].str.startswith("Lab"), "Lab", "Field")
    unknown = ~table["group"].str.startswith(("Lab", "Field"))
    if unknown.any():
        raise SplitError(f"groups with no Lab/Field prefix: {sorted(table.loc[unknown, 'group'].unique())}")
    table["cls"] = np.where(table["label"] == "Noise", "Noise", "PD")
    table["date"] = table["date"].astype(str)
    return table


# ------------------------------------------------------------------ split

def assign_field_dates(field: pd.DataFrame, forced_train: set[str], seed: int) -> dict[str, str]:
    per_date = field.pivot_table(index="date", columns="cls", values="sample_id", aggfunc="count", fill_value=0)
    per_date = per_date.reindex(columns=list(CLASSES), fill_value=0)
    totals = {c: int(per_date[c].sum()) for c in CLASSES}
    targets = {s: {c: FIELD_RATIOS[s] * totals[c] for c in CLASSES} for s in SPLITS}
    current = {s: {c: 0 for c in CLASSES} for s in SPLITS}
    assignment: dict[str, str] = {}

    def place(date: str, split: str) -> None:
        assignment[date] = split
        for c in CLASSES:
            current[split][c] += int(per_date.loc[date, c])

    for date in sorted(forced_train & set(per_date.index)):
        place(date, "train")

    rng = np.random.default_rng(seed)
    remaining = [d for d in per_date.index if d not in assignment]
    tie_break = dict(zip(remaining, rng.permutation(len(remaining))))
    remaining.sort(key=lambda d: (-int(per_date.loc[d].sum()), tie_break[d]))

    for date in remaining:
        size = int(per_date.loc[date].sum())
        best, best_score = None, -np.inf
        for split in SPLITS:
            # Deficit relative to the class total, weighted by this date's class mix.
            score = sum(
                (per_date.loc[date, c] / size) * (targets[split][c] - current[split][c]) / max(totals[c], 1)
                for c in CLASSES
            )
            if score > best_score:
                best, best_score = split, score
        place(date, best)

    enforce_coverage(field, assignment)
    refine(field, assignment, set(forced_train) & set(per_date.index), per_date, totals)
    return assignment


def split_cost(field: pd.DataFrame, assignment: dict[str, str], per_date: pd.DataFrame, totals: dict) -> float:
    """Squared deviation from the target ratios (both classes) + penalties for label coverage."""
    cost = 0.0
    for c in CLASSES:
        for s in SPLITS:
            got = sum(int(per_date.loc[d, c]) for d, sp in assignment.items() if sp == s)
            cost += (got / max(totals[c], 1) - FIELD_RATIOS[s]) ** 2
    pd_rows = field[field["cls"] == "PD"]
    for s in ("val", "test"):
        dates = {d for d, sp in assignment.items() if sp == s}
        labels = set(pd_rows.loc[pd_rows["date"].isin(dates), "label"])
        cost += 10.0 * sum(l not in labels for l in REQUIRED_LABELS)
        cost += 0.01 * sum(l not in labels for l in PREFERRED_LABELS)
    return cost


def refine(field: pd.DataFrame, assignment: dict[str, str], forced: set[str], per_date: pd.DataFrame, totals: dict) -> None:
    """Best-improvement hill climb: move one non-forced date to another split while the cost drops."""
    movable = sorted(d for d in assignment if d not in forced)
    best = split_cost(field, assignment, per_date, totals)
    while True:
        best_move = None
        for date in movable:
            original = assignment[date]
            for split in SPLITS:
                if split == original:
                    continue
                assignment[date] = split
                cost = split_cost(field, assignment, per_date, totals)
                if cost < best - 1e-12:
                    best, best_move = cost, (date, split)
                assignment[date] = original
        if best_move is None:
            return
        assignment[best_move[0]] = best_move[1]


def enforce_coverage(field: pd.DataFrame, assignment: dict[str, str]) -> None:
    """Make sure val and test each hold the required Field PD labels, moving small dates out of train."""
    pd_rows = field[field["cls"] == "PD"]
    date_sizes = field.groupby("date").size()

    def labels_in(split: str) -> set[str]:
        dates = {d for d, s in assignment.items() if s == split}
        return set(pd_rows.loc[pd_rows["date"].isin(dates), "label"])

    for split in ("val", "test"):
        for label in REQUIRED_LABELS + PREFERRED_LABELS:
            if label in labels_in(split):
                continue
            train_dates = {d for d, s in assignment.items() if s == "train"}
            donors = sorted(set(pd_rows.loc[(pd_rows["label"] == label) & pd_rows["date"].isin(train_dates), "date"]))
            # Never take train's last date of a label; prefer the smallest date so ratios move the least.
            if len(donors) < 2:
                if label in REQUIRED_LABELS:
                    raise SplitError(f"cannot give {split} a Field PD '{label}' date without emptying train")
                continue
            donor = min(donors, key=lambda d: int(date_sizes[d]))
            assignment[donor] = split


def apply_split(table: pd.DataFrame, assignment: dict[str, str]) -> pd.DataFrame:
    table = table.copy()
    table["split"] = np.where(table["domain"] == "Lab", "train", table["date"].map(assignment))
    return table


def check_split(table: pd.DataFrame, lab_dates: set[str]) -> list[str]:
    """Hard checks from data plan 4.4. Returns report lines; raises on any failure."""
    lines = []
    if table["split"].isna().any():
        raise SplitError(f"{int(table['split'].isna().sum())} files have no split")
    if table["sample_id"].duplicated().any():
        raise SplitError("sample_id appears more than once")
    field = table[table["domain"] == "Field"]
    date_splits = field.groupby("date")["split"].nunique()
    if (date_splits > 1).any():
        raise SplitError(f"Field dates in more than one split: {list(date_splits[date_splits > 1].index)}")
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                shared = set(table.loc[table["split"] == a, "date"]) & set(table.loc[table["split"] == b, "date"])
                if shared:
                    raise SplitError(f"dates shared by {a} and {b}: {sorted(shared)}")
    leak = lab_dates & set(table.loc[table["split"].isin(["val", "test"]), "date"])
    if leak:
        raise SplitError(f"Lab dates appear in val/test: {sorted(leak)}")
    if not (table.loc[table["domain"] == "Lab", "split"] == "train").all():
        raise SplitError("some Lab files are not in train")
    lines.append("leakage checks: PASS (no date or sample_id in two splits; Lab dates not in val/test)")

    for split in ("val", "test"):
        labels = set(field[(field["split"] == split) & (field["cls"] == "PD")]["label"])
        missing = [l for l in REQUIRED_LABELS if l not in labels]
        if missing:
            raise SplitError(f"{split} is missing required Field PD labels {missing}")
        note = "" if all(l in labels for l in PREFERRED_LABELS) else f" (preferred label missing: {[l for l in PREFERRED_LABELS if l not in labels]})"
        lines.append(f"coverage {split}: Field PD labels {sorted(labels)}{note}")
    return lines


# ------------------------------------------------------------------ export

def tile(matrix: np.ndarray, window: int) -> list[tuple[int, int, np.ndarray]]:
    n = matrix.shape[1] // window
    return [(k, k * window, matrix[:, k * window:(k + 1) * window]) for k in range(n)]


def export_images(dataset: AiDataset, table: pd.DataFrame, out: Path, window: int, workers: int) -> pd.DataFrame:
    rows = []
    jobs = []
    for group, part in table.groupby("group"):
        array = dataset.group_array(str(group))  # (n_group, 128, 3600) uint8
        for rec in part.itertuples(index=False):
            matrix = array[int(rec.tensor_index)]
            if matrix.shape != (128, 3600) or matrix.dtype != np.uint8:
                raise SplitError(f"{rec.sample_id}: unexpected tensor {matrix.shape} {matrix.dtype}")
            folder = out / rec.split / rec.cls.lower()
            for k, t0, win in tile(matrix, window):
                rel = Path(rec.split) / rec.cls.lower() / f"{rec.sample_id}_w{k:02d}.png"
                jobs.append((out / rel, np.ascontiguousarray(win)))
                rows.append({
                    "image_path": rel.as_posix(), "sample_id": rec.sample_id, "file": rec.file,
                    "window_idx": k, "t_start": t0, "t_end": t0 + window, "group": rec.group,
                    "label": rec.label, "cls": rec.cls, "domain": rec.domain, "date": rec.date,
                    "split": rec.split,
                })
            folder.mkdir(parents=True, exist_ok=True)
        # Write this group's images before loading the next group's tensor.
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(lambda job: Image.fromarray(job[1], mode="L").save(job[0], optimize=False), jobs))
        jobs.clear()
        dataset.release()  # free this group's tensor before the next one
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ reports

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_report(table: pd.DataFrame) -> str:
    counts = table.groupby(["split", "domain", "cls", "label"]).size().rename("files")
    return counts.to_string()


def achieved_ratios(table: pd.DataFrame) -> dict:
    field = table[table["domain"] == "Field"]
    result = {}
    for c in CLASSES:
        sub = field[field["cls"] == c]
        result[c] = {s: round(float((sub["split"] == s).mean()), 4) for s in SPLITS}
        result[c]["files"] = {s: int((sub["split"] == s).sum()) for s in SPLITS}
        result[c]["dates"] = {s: int(sub.loc[sub["split"] == s, "date"].nunique()) for s in SPLITS}
    return result


def main() -> None:
    args = parse_args()
    out: Path = args.out
    if out.exists():
        raise SystemExit(f"refusing to overwrite existing folder: {out}")

    ai_root = args.ai_data_root.resolve()
    dataset = AiDataset.load(ai_root)
    table = build_file_table(dataset, args.exclude_groups)
    lab_dates = set(table.loc[table["domain"] == "Lab", "date"])
    field = table[table["domain"] == "Field"]
    # Undated files cannot be date-split safely, so they never go to val/test.
    # Field dates that also occur in Lab are forced to train as well (all Lab is train).
    forced = (lab_dates & set(field["date"])) | ({UNKNOWN_DATE} & set(field["date"]))

    assignment = assign_field_dates(field, forced, args.split_seed)
    table = apply_split(table, assignment)
    check_lines = check_split(table, lab_dates)
    ratios = achieved_ratios(table)

    out.mkdir(parents=True)
    manifest = export_images(dataset, table, out, args.window, args.workers)
    expected = len(table) * (3600 // args.window)
    if len(manifest) != expected:
        raise SplitError(f"exported {len(manifest)} images, expected {expected}")

    manifest.to_csv(out / "manifest.csv", index=False)
    file_cols = ["sample_id", "file", "group", "label", "cls", "domain", "date", "split"]
    table[file_cols].to_csv(out / "files.csv", index=False)

    split_dates = {
        "field_date_to_split": dict(sorted(assignment.items())),
        "forced_train_dates": sorted(forced),
        "lab_dates": sorted(lab_dates),
        "achieved_field_ratios": ratios,
    }
    (out / "split_dates.json").write_text(json.dumps(split_dates, indent=2, ensure_ascii=False), encoding="utf-8")

    report = "\n".join([
        "PatchCore v2 split report", "",
        *check_lines, "",
        f"forced-to-train Field dates (undated, or also a Lab date): {sorted(forced)}", "",
        "Achieved Field ratios (target 70/15/15):",
        json.dumps(ratios, indent=2), "",
        "Files per split x domain x cls x label:",
        count_report(table), "",
        f"Images: {len(manifest)} ({3600 // args.window} windows/file, {args.window}x{args.window})",
    ])
    (out / "split_report.txt").write_text(report + "\n", encoding="utf-8")

    run_config = {
        "created_at": datetime.now().astimezone().isoformat(),
        "script": "PatchCore/build_prpd_v2.py",
        "ai_data_root": str(ai_root),
        "raw_data_version": dataset.run_config.get("raw_data_version"),
        "split_seed": args.split_seed,
        "field_ratios_target": FIELD_RATIOS,
        "split_unit": "Field measurement date (one date -> one split); all Lab -> train; undated (unknown_date) Field files -> train",
        "class_rule": "cls = Noise if label == 'Noise' else PD",
        "exclude_groups": args.exclude_groups,
        "window": [128, args.window],
        "windows_per_file": 3600 // args.window,
        "dropped_tail_cycles": 3600 % args.window,
        "pixel_values": "raw uint8, no per-image rescaling; saved as 8-bit grayscale PNG (L)",
        "n_files": int(len(table)),
        "n_images": int(len(manifest)),
    }
    (out / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding="utf-8")

    by_split = table.groupby(["split", "cls"]).size().unstack(fill_value=0)
    description = "\n".join([
        "PatchCore v2 dataset (PD vs Noise, two memory banks)",
        f"Created: {run_config['created_at']}",
        f"Source: {ai_root} ({run_config['raw_data_version']}); groups excluded: {args.exclude_groups}",
        "Class: PD = Corona/Floating/Particle/Void labels; Noise = label 'Noise' (includes files relabelled by the 260909 change log that stay in PD group folders).",
        "Train = 100% Lab PD + Lab Noise + ~70% Field PD + Field Noise. Val/test = ~15% Field each.",
        "Field is split by measurement date so that no date appears in two splits; Field dates that also occur in Lab are forced into train.",
        f"Each file (128 phase x 3600 cycles) is tiled into {3600 // args.window} non-overlapping 128x{args.window} windows; the last {3600 % args.window} cycles are dropped.",
        "Pixel values are the raw uint8 PRPD amplitudes, with no per-image rescaling.",
        "",
        "Files per split and class:",
        by_split.to_string(),
        "",
        f"Total files: {len(table)}   Total images: {len(manifest)}",
        f"Split seed: {args.split_seed}",
    ])
    (out / "dataset_description.txt").write_text(description + "\n", encoding="utf-8")

    listed = ["manifest.csv", "files.csv", "split_dates.json", "split_report.txt", "run_config.json", "dataset_description.txt"]
    manifest_json = {
        "artifacts": [
            {"path": name, "bytes": (out / name).stat().st_size, "sha256": sha256(out / name)} for name in listed
        ] + [{"path": "manifest.json", "note": "this file"}],
        "image_count": int(len(manifest)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest_json, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

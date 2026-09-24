"""Shared v2 data access for the SVM / EfficientAD / PatchCore comparison (torch-free).

All three methods read the same PatchCore v2 build (PatchCore/patchcore_data/prpd_v2): same files,
same date split, same 128x128 raw-uint8 windows, and the same seeded choice of K train windows per file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "PatchCore" / "patchcore_data" / "prpd_v2"
WINDOW_KEYS = ["sample_id", "window_idx", "split", "cls", "label", "group", "domain", "date"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(data_root: Path = DATA_ROOT) -> tuple[pd.DataFrame, dict]:
    """Manifest + run_config of the v2 build, after verifying the recorded checksums."""
    data_root = Path(data_root)
    listing = json.loads((data_root / "manifest.json").read_text())
    for entry in listing["artifacts"]:
        if "sha256" in entry and _sha256(data_root / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"checksum mismatch in {data_root / entry['path']}")
    manifest = pd.read_csv(data_root / "manifest.csv", dtype={"date": str})
    config = json.loads((data_root / "run_config.json").read_text())
    config["run_config_sha256"] = _sha256(data_root / "run_config.json")
    return manifest, config


def pick_train_windows(manifest: pd.DataFrame, cls: str, k: int, seed: int) -> pd.DataFrame:
    """K seeded windows per train file of one class — same rule as PatchCore/train_two_bank.py."""
    train = manifest[(manifest["split"] == "train") & (manifest["cls"].str.lower() == cls)]
    rng = np.random.default_rng(seed)
    chosen = [part.iloc[np.sort(rng.choice(len(part), size=min(k, len(part)), replace=False))]
              for _, part in train.groupby("sample_id", sort=True)]
    return pd.concat(chosen).reset_index(drop=True)


def eval_windows(manifest: pd.DataFrame, splits=("val", "test")) -> pd.DataFrame:
    return manifest[manifest["split"].isin(splits)].reset_index(drop=True)


def load_window(image_path: str, data_root: Path = DATA_ROOT) -> np.ndarray:
    """(128 phase, 128 cycles) uint8, exactly as exported (no rescaling)."""
    return np.asarray(Image.open(Path(data_root) / image_path), dtype=np.uint8)

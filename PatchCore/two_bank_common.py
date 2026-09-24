"""Shared pieces for the PatchCore v2 two-bank (PD vs Noise) pipeline.

See Plans/PatchCore_training_plan.md. Run all scripts from the repo root with
PYTHONPATH including PatchCore/patchcore-inspection/src.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.sampler

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASSES = ("pd", "noise")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_data_root(data_root: Path) -> tuple[pd.DataFrame, dict]:
    """Read the v2 manifest after checking the checksums recorded by build_prpd_v2.py."""
    data_root = Path(data_root)
    listing = json.loads((data_root / "manifest.json").read_text())
    for entry in listing["artifacts"]:
        if "sha256" in entry and sha256(data_root / entry["path"]) != entry["sha256"]:
            raise RuntimeError(f"checksum mismatch in {data_root / entry['path']}")
    config = json.loads((data_root / "run_config.json").read_text())
    manifest = pd.read_csv(data_root / "manifest.csv", dtype={"date": str})
    if len(manifest) != listing["image_count"]:
        raise RuntimeError(f"manifest has {len(manifest)} rows, manifest.json says {listing['image_count']}")
    return manifest, config


class WindowDataset(torch.utils.data.Dataset):
    """128x128 raw-uint8 PRPD windows -> 3x224x224 ImageNet-normalized tensors (resize only, no crop)."""

    def __init__(self, data_root: Path, rows: pd.DataFrame, image_size: int = 224):
        self.data_root = Path(data_root)
        self.paths = rows["image_path"].tolist()
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.data_root / self.paths[idx]).convert("RGB")
        return self.transform(image)


def make_loader(dataset: WindowDataset, batch_size: int, workers: int) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                       num_workers=workers, pin_memory=True)


class ChunkedApproxGreedyCoreset(patchcore.sampler.ApproximateGreedyCoresetSampler):
    """Same algorithm as the library sampler, but never puts the full-dimension bank on the GPU.

    The library's _reduce_features moves all N x D features to the GPU before the random
    projection. For the PD bank (~5.6M x 1024 fp32 = ~23 GB) that does not fit on a 24 GB card
    next to the backbone, so project in chunks and keep only the N x 128 projection on the GPU.
    """

    chunk = 500_000
    last_indices = None  # pool rows kept by the coreset, so bank entries can be traced to training windows

    def run(self, features):
        import numpy as _np
        feats = torch.from_numpy(features) if isinstance(features, _np.ndarray) else features
        reduced = self._reduce_features(feats)
        idx = self._compute_greedy_coreset_indices(reduced)
        self.last_indices = _np.asarray(idx)
        return features[self.last_indices]

    def _reduce_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[1] == self.dimension_to_project_features_to:
            return features.to(self.device)
        mapper = torch.nn.Linear(features.shape[1], self.dimension_to_project_features_to, bias=False).to(self.device)
        with torch.no_grad():
            parts = [mapper(features[i:i + self.chunk].to(self.device)) for i in range(0, len(features), self.chunk)]
        return torch.cat(parts)


def build_patchcore(settings: dict, device: torch.device, sampler=None) -> patchcore.patchcore.PatchCore:
    backbone = patchcore.backbones.load(settings["backbone"])
    backbone.name = settings["backbone"]
    model = patchcore.patchcore.PatchCore(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=settings["layers"],
        device=device,
        input_shape=(3, settings["image_size"], settings["image_size"]),
        pretrain_embed_dimension=settings["pretrain_dim"],
        target_embed_dimension=settings["target_dim"],
        patchsize=settings["patchsize"],
        anomaly_score_num_nn=settings["num_nn"],
        featuresampler=sampler or patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(on_gpu=True, num_workers=8),
    )
    return model


def load_bank(bank_dir: Path, device: torch.device) -> patchcore.patchcore.PatchCore:
    model = patchcore.patchcore.PatchCore(device)
    model.load_from_path(str(bank_dir), device, patchcore.common.FaissNN(on_gpu=True, num_workers=8))
    return model


def fill_memory_bank_prealloc(model: patchcore.patchcore.PatchCore, loader, n_windows: int) -> None:
    """Drop-in for PatchCore._fill_memory_bank with half the peak host RAM.

    The library appends per-batch arrays to a list and then np.concatenate()s them, so the whole
    feature pool exists twice at the peak. Here the pool is preallocated once (size known after the
    first batch: patches per window x windows) and filled in place.
    """
    _ = model.forward_modules.eval()
    pool, row = None, 0
    for images in loader:
        with torch.no_grad():
            feats = np.asarray(model._embed(images.to(torch.float).to(model.device)))
        if pool is None:
            per_window = feats.shape[0] // images.shape[0]
            pool = np.empty((n_windows * per_window, feats.shape[1]), dtype=np.float32)
            print(f"feature pool {pool.shape} = {pool.nbytes / 1e9:.1f} GB", flush=True)
        pool[row:row + len(feats)] = feats
        row += len(feats)
    if row != len(pool):
        raise RuntimeError(f"filled {row} of {len(pool)} feature rows")
    features = model.featuresampler.run(pool)
    patches_per_window = len(pool) // n_windows
    del pool
    model.anomaly_scorer.fit(detection_features=[features])
    return getattr(model.featuresampler, "last_indices", None), patches_per_window


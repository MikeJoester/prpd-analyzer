"""EfficientAD, two models (PD model, Noise model) on the PatchCore v2 data — PD vs Noise.

    CUDA_VISIBLE_DEVICES=0 python EfficientAD/efficientad_v2.py --cls pd    --out Results/efficientad/effad_v2_s42 --seed 42
    CUDA_VISIBLE_DEVICES=1 python EfficientAD/efficientad_v2.py --cls noise --out Results/efficientad/effad_v2_s42 --seed 42
    python Comparison/evaluate_two_model.py --run Results/efficientad/effad_v2_s42 --title EfficientAD

Architecture, losses and training settings are those of train_efficient_ad.py (PDN teacher/student,
64-ch bottleneck autoencoder, hard-feature ST loss at the 99.9th percentile, AE and ST-AE MSE losses,
Adam lr 1e-4, batch 8, 5 epochs, input /255). Changes, all needed for v2:
  - input is the v2 128x128 window (was a random 128x256 crop); the AE bottleneck resize follows the input
  - trained on the same K windows/file as PatchCore, one model per class
  - the frozen random teacher is SEEDED AND SAVED. The old evaluate_efficient_ad.py built a fresh random
    teacher at test time, so the student was scored against a teacher it never learned to imitate.
Window score = max over the map 0.5*(T-S)^2 + 0.5*(AE-S_ae)^2 (unchanged); pixel maps are not evaluated
because the data has no pixel-level ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "Comparison"))
from EfficientAD.train_efficient_ad import PDN  # noqa: E402  (unchanged architecture)
from v2_data import WINDOW_KEYS, eval_windows, load_manifest, load_window, pick_train_windows  # noqa: E402

T_CH = 384


class Autoencoder(nn.Module):
    """train_efficient_ad.Autoencoder with the bottleneck resize tied to the input size (was fixed to 4x8 for 128x256)."""

    def __init__(self, out_channels: int = T_CH):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1), nn.ReLU(), nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(), nn.Conv2d(64, 64, 8, 1, 0))
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(), nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(64, 64, 4, 1, 2), nn.ReLU(), nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(64, out_channels, 3, 1, 1))

    def forward(self, x):
        z = F.interpolate(self.encoder(x), size=(x.shape[2] // 32, x.shape[3] // 32), mode="bilinear")
        return self.decoder(z)


class Windows(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return torch.from_numpy(load_window(self.paths[i]).astype(np.float32) / 255.0).unsqueeze(0)


def maps(teacher, student, ae, x):
    t = teacher(x)
    s = student(x)
    a = F.interpolate(ae(x), size=t.shape[2:], mode="bilinear")
    s_ae = F.interpolate(s[:, T_CH:], size=t.shape[2:], mode="bilinear")
    return t, s[:, :T_CH], a, s_ae


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cls", choices=["pd", "noise"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    started = time.time()
    model_dir = args.out / f"model_{args.cls}"
    model_dir.mkdir(parents=True, exist_ok=False)

    manifest, data_config = load_manifest()
    train = pick_train_windows(manifest, args.cls, args.k, args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    loader = torch.utils.data.DataLoader(Windows(train["image_path"]), batch_size=args.batch_size, shuffle=True,
                                         generator=gen, num_workers=args.workers, drop_last=False)

    teacher = PDN(out_channels=T_CH).to(device).eval()
    for q in teacher.parameters():
        q.requires_grad = False
    student = PDN(out_channels=2 * T_CH).to(device)
    ae = Autoencoder(T_CH).to(device)
    opt = torch.optim.Adam(list(student.parameters()) + list(ae.parameters()), lr=args.lr)

    history = []
    for epoch in range(args.epochs):
        student.train(); ae.train()
        total = 0.0
        for x in loader:
            x = x.to(device)
            with torch.no_grad():
                t = teacher(x)
            s = student(x)
            s_t = s[:, :T_CH]
            a = F.interpolate(ae(x), size=t.shape[2:], mode="bilinear")
            s_ae = F.interpolate(s[:, T_CH:], size=t.shape[2:], mode="bilinear")
            d = (t - s_t) ** 2
            q = torch.quantile(d.flatten(2), 0.999, dim=2, keepdim=True).unsqueeze(-1)
            loss = d[d >= q].mean() + F.mse_loss(a, t) + F.mse_loss(s_ae, a.detach())
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        history.append(total / len(loader))
        print(f"[{args.cls}] epoch {epoch + 1}/{args.epochs} loss {history[-1]:.5f}", flush=True)

    torch.save({"teacher": teacher.state_dict(), "student": student.state_dict(), "autoencoder": ae.state_dict()},
               model_dir / "weights.pt")

    # Score every val/test window with this class's model.
    evals = eval_windows(manifest)
    student.eval(); ae.eval()
    scores = []
    with torch.no_grad():
        for x in torch.utils.data.DataLoader(Windows(evals["image_path"]), batch_size=64, num_workers=args.workers):
            t, s_t, a, s_ae = maps(teacher, student, ae, x.to(device))
            m = 0.5 * ((t - s_t) ** 2).mean(1) + 0.5 * ((a - s_ae) ** 2).mean(1)
            scores.append(m.flatten(1).max(1).values.cpu().numpy())
    out = evals[WINDOW_KEYS].copy()
    out[f"d_{args.cls}"] = np.concatenate(scores)
    out.to_csv(args.out / f"window_scores_{args.cls}.csv", index=False)

    info = {"method": "efficientad", "cls": args.cls, "created_at": datetime.now().astimezone().isoformat(),
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "train_files": int(train["sample_id"].nunique()), "train_windows": int(len(train)),
            "loss_history": history, "ai_data_root": data_config["ai_data_root"],
            "raw_data_version": data_config["raw_data_version"], "split_seed": data_config["split_seed"],
            "data_run_config_sha256": data_config["run_config_sha256"], "torch": torch.__version__,
            "seconds": round(time.time() - started, 1)}
    (model_dir / "model_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: info[k] for k in ("train_windows", "seconds")}), flush=True)


if __name__ == "__main__":
    main()

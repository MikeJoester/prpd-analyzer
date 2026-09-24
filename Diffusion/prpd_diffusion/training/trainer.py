"""조건부 diffusion denoiser 학습 루프.

재현성 규칙: seed, config, 데이터 split, raw_data_version을 run 폴더에 함께 저장한다.
checkpoint는 EMA 가중치를 함께 저장하고, 샘플링에는 EMA 가중치를 사용한다.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..diffusion.gaussian import GaussianDiffusion


class EMA:
    """지수이동평균 가중치. diffusion 학습에서 샘플 품질을 크게 좌우한다."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for shadow_param, param in zip(self.shadow.parameters(), model.parameters()):
            shadow_param.mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)
        for shadow_buffer, buffer in zip(self.shadow.buffers(), model.buffers()):
            shadow_buffer.copy_(buffer)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        diffusion: GaussianDiffusion,
        train_dataset,
        val_dataset,
        run_folder: Path,
        config,  # runs.config.DenoiseConfig
        device: str | None = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.config = config
        self.run_folder = Path(run_folder)
        self.checkpoint_folder = self.run_folder / "checkpoints"
        self.checkpoint_folder.mkdir(parents=True, exist_ok=True)

        seed_everything(config.seed)
        self.model = model.to(self.device)
        self.diffusion = diffusion.to(self.device)
        self.ema = EMA(self.model, config.train.ema_decay)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.train.learning_rate, weight_decay=0.01
        )

        loader_kwargs = dict(
            batch_size=config.train.batch_size,
            num_workers=config.train.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
        )
        self.train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        self.val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.log_path = self.run_folder / "train_log.jsonl"
        self.step = 0

    # ------------------------------------------------------------------ 로그

    def _log(self, record: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------ 학습/검증

    def train(self) -> Path:
        best_val = float("inf")
        for epoch in range(self.config.train.epochs):
            self.train_dataset.set_epoch(epoch)
            self.val_dataset.set_epoch(0)  # 검증 쌍은 epoch 간 고정해 비교 가능하게 둔다
            started = time.perf_counter()
            train_loss = self._run_epoch(epoch)
            val_loss = self.validate()
            record = {
                "epoch": epoch,
                "step": self.step,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "seconds": round(time.perf_counter() - started, 1),
            }
            self._log(record)
            print(
                f"epoch {epoch:03d} | train {train_loss:.4f} | val {val_loss:.4f} "
                f"| {record['seconds']:.0f}s"
            )

            self.save_checkpoint(self.checkpoint_folder / "last.pt", epoch, val_loss)
            if val_loss < best_val:
                best_val = val_loss
                self.save_checkpoint(self.checkpoint_folder / "best.pt", epoch, val_loss)
        return self.checkpoint_folder / "best.pt"

    def _run_epoch(self, epoch: int) -> float:
        self.model.train()
        losses: list[float] = []
        for batch in self.train_loader:
            clean = batch["clean"].to(self.device, non_blocking=True)
            noisy = batch["noisy"].to(self.device, non_blocking=True)

            loss, _ = self.diffusion.training_loss(self.model, clean, noisy)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
            self.optimizer.step()
            self.ema.update(self.model)

            self.step += 1
            losses.append(float(loss.detach()))
            if self.config.train.max_steps and self.step >= self.config.train.max_steps:
                break
        return float(np.mean(losses)) if losses else float("nan")

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        losses: list[float] = []
        generator = torch.Generator(device=self.device).manual_seed(self.config.seed)
        for batch in self.val_loader:
            clean = batch["clean"].to(self.device)
            noisy = batch["noisy"].to(self.device)
            # 검증 손실을 비교 가능하게 하려면 t와 noise도 고정해야 한다.
            t = self.diffusion.sample_timesteps(clean.shape[0], self.device, generator=generator)
            noise = torch.randn(clean.shape, device=self.device, generator=generator)
            loss, _ = self.diffusion.training_loss(self.model, clean, noisy, t=t, noise=noise)
            losses.append(float(loss))
        return float(np.mean(losses)) if losses else float("nan")

    # ------------------------------------------------------------ checkpoint

    def save_checkpoint(self, path: Path, epoch: int, val_loss: float) -> Path:
        torch.save(
            {
                "model": self.model.state_dict(),
                "ema": self.ema.shadow.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch,
                "step": self.step,
                "val_loss": val_loss,
                "config": asdict(self.config),
            },
            path,
        )
        return path


def load_checkpoint(path: Path, model: torch.nn.Module, use_ema: bool = True) -> dict:
    """저장된 가중치를 모델에 적재한다. 기본은 EMA 가중치."""
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["ema" if use_ema and "ema" in payload else "model"])
    return payload

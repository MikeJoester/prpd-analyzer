"""Gaussian diffusion 스케줄. numpy만 사용하므로 torch 없이 검증할 수 있다.

표기는 Ho et al.(2020) DDPM을 따른다.

    q(x_t | x_0) = N(sqrt(ab_t) x_0, (1 - ab_t) I),   ab_t = prod_{s<=t} (1 - beta_s)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCHEDULE_KINDS = ("linear", "cosine")


def make_betas(kind: str, timesteps: int) -> np.ndarray:
    """beta 스케줄. 값은 (0, 1) 범위의 증가 수열."""
    if timesteps < 2:
        raise ValueError("timesteps는 2 이상이어야 합니다.")
    if kind == "linear":
        scale = 1000.0 / timesteps  # DDPM 기본값(T=1000)을 다른 T로 옮길 때의 보정
        return np.linspace(scale * 1e-4, scale * 0.02, timesteps, dtype=np.float64)
    if kind == "cosine":
        # Nichol & Dhariwal (2021), improved DDPM
        steps = np.arange(timesteps + 1, dtype=np.float64) / timesteps
        alphas_cumprod = np.cos((steps + 0.008) / 1.008 * np.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return np.clip(betas, 1e-8, 0.999)
    raise ValueError(f"알 수 없는 schedule: {kind} (가능: {SCHEDULE_KINDS})")


@dataclass(frozen=True)
class DiffusionSchedule:
    """학습·샘플링에 필요한 계수를 미리 계산해 둔다."""

    kind: str
    timesteps: int
    betas: np.ndarray
    alphas_cumprod: np.ndarray
    alphas_cumprod_prev: np.ndarray
    sqrt_alphas_cumprod: np.ndarray
    sqrt_one_minus_alphas_cumprod: np.ndarray
    sqrt_recip_alphas_cumprod: np.ndarray
    sqrt_recipm1_alphas_cumprod: np.ndarray
    posterior_variance: np.ndarray
    posterior_log_variance_clipped: np.ndarray
    posterior_mean_coef1: np.ndarray
    posterior_mean_coef2: np.ndarray

    @classmethod
    def create(cls, kind: str = "cosine", timesteps: int = 1000) -> "DiffusionSchedule":
        betas = make_betas(kind, timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        return cls(
            kind=kind,
            timesteps=timesteps,
            betas=betas,
            alphas_cumprod=alphas_cumprod,
            alphas_cumprod_prev=alphas_cumprod_prev,
            sqrt_alphas_cumprod=np.sqrt(alphas_cumprod),
            sqrt_one_minus_alphas_cumprod=np.sqrt(1.0 - alphas_cumprod),
            sqrt_recip_alphas_cumprod=np.sqrt(1.0 / alphas_cumprod),
            sqrt_recipm1_alphas_cumprod=np.sqrt(1.0 / alphas_cumprod - 1.0),
            posterior_variance=posterior_variance,
            # t=0에서 분산이 0이 되어 log가 발산하므로 t=1 값으로 clip한다.
            posterior_log_variance_clipped=np.log(
                np.append(posterior_variance[1], posterior_variance[1:])
            ),
            posterior_mean_coef1=betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
            posterior_mean_coef2=(1.0 - alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - alphas_cumprod),
        )

    def q_sample(self, x_start: np.ndarray, t: int, noise: np.ndarray) -> np.ndarray:
        """numpy 버전 forward diffusion. 스케줄 검증용."""
        return (
            self.sqrt_alphas_cumprod[t] * x_start
            + self.sqrt_one_minus_alphas_cumprod[t] * noise
        )

    def to_torch(self, device=None, dtype=None):
        """torch 텐서 dict으로 변환한다(학습·샘플링에서 사용)."""
        import torch  # 지연 import: numpy만 쓰는 경로에서는 torch가 필요 없다.

        dtype = dtype or torch.float32
        names = (
            "betas",
            "alphas_cumprod",
            "alphas_cumprod_prev",
            "sqrt_alphas_cumprod",
            "sqrt_one_minus_alphas_cumprod",
            "sqrt_recip_alphas_cumprod",
            "sqrt_recipm1_alphas_cumprod",
            "posterior_variance",
            "posterior_log_variance_clipped",
            "posterior_mean_coef1",
            "posterior_mean_coef2",
        )
        return {
            name: torch.as_tensor(getattr(self, name), device=device, dtype=dtype)
            for name in names
        }

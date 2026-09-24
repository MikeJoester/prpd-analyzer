"""조건부 Gaussian diffusion 학습 목적함수 (torch).

노이즈 제거는 **조건부 생성**으로 푼다. 조건 y(노이즈 섞인 PRPS)를 채널로 concat해
모델이 clean x0에 대한 diffusion을 학습한다 (Palette / SR3와 같은 구성).

    model(cat[x_t, y], t) -> eps_hat
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .schedule import DiffusionSchedule


def _extract(values: torch.Tensor, t: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """timestep별 계수를 배치 차원에 맞춰 broadcast 한다."""
    out = values.gather(0, t)
    return out.reshape(t.shape[0], *([1] * (len(shape) - 1)))


class GaussianDiffusion:
    """스케줄 텐서를 들고 forward/loss/posterior를 계산한다. 파라미터는 없다."""

    def __init__(self, schedule: DiffusionSchedule, device=None, loss_type: str = "l2") -> None:
        if loss_type not in ("l1", "l2", "huber"):
            raise ValueError(f"알 수 없는 loss_type: {loss_type}")
        self.schedule = schedule
        self.timesteps = schedule.timesteps
        self.loss_type = loss_type
        self.coefficients = schedule.to_torch(device=device)

    def to(self, device) -> "GaussianDiffusion":
        self.coefficients = {name: value.to(device) for name, value in self.coefficients.items()}
        return self

    def sample_timesteps(self, batch_size: int, device, generator=None) -> torch.Tensor:
        return torch.randint(
            0, self.timesteps, (batch_size,), device=device, generator=generator, dtype=torch.long
        )

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            _extract(self.coefficients["sqrt_alphas_cumprod"], t, x_start.shape) * x_start
            + _extract(self.coefficients["sqrt_one_minus_alphas_cumprod"], t, x_start.shape) * noise
        )

    def predict_start_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        return (
            _extract(self.coefficients["sqrt_recip_alphas_cumprod"], t, x_t.shape) * x_t
            - _extract(self.coefficients["sqrt_recipm1_alphas_cumprod"], t, x_t.shape) * noise
        )

    def q_posterior(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = (
            _extract(self.coefficients["posterior_mean_coef1"], t, x_t.shape) * x_start
            + _extract(self.coefficients["posterior_mean_coef2"], t, x_t.shape) * x_t
        )
        log_variance = _extract(self.coefficients["posterior_log_variance_clipped"], t, x_t.shape)
        return mean, log_variance

    def p_mean_variance(
        self,
        model,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor,
        clip_denoised: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        predicted_noise = model(torch.cat([x_t, condition], dim=1), t)
        x_start = self.predict_start_from_noise(x_t, t, predicted_noise)
        if clip_denoised:
            x_start = x_start.clamp(-1.0, 1.0)  # 입력 규약이 [-1, 1]이므로 항상 유효
        mean, log_variance = self.q_posterior(x_start, x_t, t)
        return mean, log_variance, x_start

    def training_loss(
        self,
        model,
        x_start: torch.Tensor,
        condition: torch.Tensor,
        t: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """epsilon 예측 손실. `(loss, 로그용 dict)`를 반환한다."""
        if t is None:
            t = self.sample_timesteps(x_start.shape[0], x_start.device)
        if noise is None:
            noise = torch.randn_like(x_start)

        x_t = self.q_sample(x_start, t, noise)
        predicted_noise = model(torch.cat([x_t, condition], dim=1), t)

        if self.loss_type == "l2":
            loss = F.mse_loss(predicted_noise, noise)
        elif self.loss_type == "l1":
            loss = F.l1_loss(predicted_noise, noise)
        else:
            loss = F.smooth_l1_loss(predicted_noise, noise)

        return loss, {"loss": float(loss.detach()), "t_mean": float(t.float().mean())}

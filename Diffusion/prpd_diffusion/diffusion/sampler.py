"""조건부 샘플링 (torch). DDPM 전체 스텝과 DDIM 축약 스텝.

노이즈 제거 추론에서는 조건 y가 강한 정보를 주므로 보통 DDIM 20~50 스텝이면 충분하다.
샘플링은 확률적이므로 재현을 위해 generator seed를 항상 기록한다.
"""

from __future__ import annotations

import torch

from .gaussian import GaussianDiffusion, _extract


@torch.no_grad()
def ddpm_sample(
    diffusion: GaussianDiffusion,
    model,
    condition: torch.Tensor,
    generator: torch.Generator | None = None,
    clip_denoised: bool = True,
    progress=None,
) -> torch.Tensor:
    """조건 y에서 clean x0를 복원한다. 반환 범위는 `[-1, 1]`."""
    device = condition.device
    x_t = torch.randn(condition.shape, device=device, generator=generator)
    steps = range(diffusion.timesteps - 1, -1, -1)
    if progress is not None:
        steps = progress(steps)

    for step in steps:
        t = torch.full((condition.shape[0],), step, device=device, dtype=torch.long)
        mean, log_variance, _ = diffusion.p_mean_variance(model, x_t, t, condition, clip_denoised)
        if step > 0:
            noise = torch.randn(x_t.shape, device=device, generator=generator)
            x_t = mean + (0.5 * log_variance).exp() * noise
        else:
            x_t = mean
    return x_t


@torch.no_grad()
def ddim_sample(
    diffusion: GaussianDiffusion,
    model,
    condition: torch.Tensor,
    steps: int = 50,
    eta: float = 0.0,
    generator: torch.Generator | None = None,
    clip_denoised: bool = True,
) -> torch.Tensor:
    """DDIM 축약 샘플링. `eta=0`이면 결정적(초기 노이즈만 무작위)."""
    device = condition.device
    timestep_sequence = torch.linspace(
        diffusion.timesteps - 1, 0, steps, device=device
    ).long().tolist()

    x_t = torch.randn(condition.shape, device=device, generator=generator)
    alphas_cumprod = diffusion.coefficients["alphas_cumprod"]

    for index, step in enumerate(timestep_sequence):
        t = torch.full((condition.shape[0],), step, device=device, dtype=torch.long)
        predicted_noise = model(torch.cat([x_t, condition], dim=1), t)
        x_start = diffusion.predict_start_from_noise(x_t, t, predicted_noise)
        if clip_denoised:
            x_start = x_start.clamp(-1.0, 1.0)
            # x0를 clip 했으므로 eps도 다시 맞춰 준다(불일치 시 누적 오차 발생).
            predicted_noise = (
                _extract(diffusion.coefficients["sqrt_recip_alphas_cumprod"], t, x_t.shape) * x_t
                - x_start
            ) / _extract(diffusion.coefficients["sqrt_recipm1_alphas_cumprod"], t, x_t.shape)

        next_step = timestep_sequence[index + 1] if index + 1 < len(timestep_sequence) else -1
        alpha_next = (
            alphas_cumprod[next_step]
            if next_step >= 0
            else torch.ones((), device=device, dtype=alphas_cumprod.dtype)
        )
        alpha_current = alphas_cumprod[step]

        sigma = eta * torch.sqrt(
            (1 - alpha_next) / (1 - alpha_current) * (1 - alpha_current / alpha_next)
        )
        direction = torch.sqrt((1 - alpha_next - sigma**2).clamp(min=0.0)) * predicted_noise
        x_t = torch.sqrt(alpha_next) * x_start + direction
        if eta > 0 and next_step >= 0:
            x_t = x_t + sigma * torch.randn(x_t.shape, device=device, generator=generator)

    return x_t


@torch.no_grad()
def denoise_full_file(
    diffusion: GaussianDiffusion,
    model,
    matrix,
    crop_width: int = 256,
    batch_size: int = 8,
    steps: int = 50,
    eta: float = 0.0,
    device=None,
    seed: int = 0,
):
    """`(128, 3600)` uint8 한 파일을 window 단위로 복원해 다시 uint8로 이어붙인다."""
    import numpy as np

    from ..data.representation import (
        from_model_output,
        stitch_time_windows,
        tile_time_windows,
        to_model_input,
    )

    windows, pad = tile_time_windows(np.asarray(matrix), crop_width)
    generator = torch.Generator(device=device or "cpu").manual_seed(int(seed))

    outputs = []
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        condition = torch.from_numpy(to_model_input(chunk)).unsqueeze(1).to(device)
        restored = ddim_sample(
            diffusion, model, condition, steps=steps, eta=eta, generator=generator
        )
        outputs.append(restored.squeeze(1).cpu().numpy())

    return from_model_output(stitch_time_windows(np.concatenate(outputs, axis=0), pad))

"""torch가 설치된 뒤에 U-Net과 diffusion 경로의 형태를 검증한다.

skip connection 개수/채널이 어긋나면 여기서 바로 드러난다.
torch 미설치 환경에서는 전체가 skip된다.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _skip import Skip


def _torch():
    try:
        import torch
    except ModuleNotFoundError:
        raise Skip("torch 미설치 — requirements-diffusion.txt 설치 후 실행하세요.")
    return torch


def _model(torch, **overrides):
    from Diffusion.prpd_diffusion.models.unet import ConditionalUNet

    defaults = dict(base_channels=8, channel_multipliers=(1, 2), blocks_per_stage=1, attention_stages=())
    defaults.update(overrides)
    return ConditionalUNet(**defaults)


def test_unet_output_shape_matches_input():
    torch = _torch()
    model = _model(torch)
    x = torch.randn(2, 2, 128, 128)
    t = torch.randint(0, 100, (2,))
    assert model(x, t).shape == (2, 1, 128, 128)


def test_unet_handles_default_crop_width():
    torch = _torch()
    model = _model(torch, channel_multipliers=(1, 2, 4), attention_stages=(2,))
    x = torch.randn(1, 2, 128, 256)
    t = torch.zeros(1, dtype=torch.long)
    assert model(x, t).shape == (1, 1, 128, 256)


def test_phase_axis_padding_is_circular():
    """위상축을 순환 이동하면 출력도 같은 만큼 순환 이동해야 한다."""
    torch = _torch()
    model = _model(torch).eval()
    x = torch.randn(1, 2, 128, 128)
    t = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        rolled_output = model(torch.roll(x, 16, dims=2), t)
        output_rolled = torch.roll(model(x, t), 16, dims=2)
    assert torch.allclose(rolled_output, output_rolled, atol=1e-4)


def test_training_loss_is_finite_and_backpropagates():
    torch = _torch()
    from Diffusion.prpd_diffusion.diffusion.gaussian import GaussianDiffusion
    from Diffusion.prpd_diffusion.diffusion.schedule import DiffusionSchedule

    model = _model(torch)
    diffusion = GaussianDiffusion(DiffusionSchedule.create("cosine", 50))
    clean = torch.rand(2, 1, 128, 128) * 2 - 1
    noisy = torch.rand(2, 1, 128, 128) * 2 - 1

    loss, log = diffusion.training_loss(model, clean, noisy)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert 0 <= log["t_mean"] < 50


def test_ddim_sampling_returns_valid_range():
    torch = _torch()
    from Diffusion.prpd_diffusion.diffusion.gaussian import GaussianDiffusion
    from Diffusion.prpd_diffusion.diffusion.sampler import ddim_sample
    from Diffusion.prpd_diffusion.diffusion.schedule import DiffusionSchedule

    model = _model(torch).eval()
    diffusion = GaussianDiffusion(DiffusionSchedule.create("cosine", 50))
    condition = torch.rand(1, 1, 128, 128) * 2 - 1
    generator = torch.Generator().manual_seed(0)
    restored = ddim_sample(diffusion, model, condition, steps=5, generator=generator)
    assert restored.shape == condition.shape
    assert torch.isfinite(restored).all()


def test_denoise_full_file_returns_uint8_original_shape():
    torch = _torch()
    import numpy as np

    from Diffusion.prpd_diffusion.diffusion.gaussian import GaussianDiffusion
    from Diffusion.prpd_diffusion.diffusion.sampler import denoise_full_file
    from Diffusion.prpd_diffusion.diffusion.schedule import DiffusionSchedule

    model = _model(torch).eval()
    diffusion = GaussianDiffusion(DiffusionSchedule.create("cosine", 20))
    matrix = np.random.default_rng(0).integers(0, 256, (128, 3600), dtype=np.uint8)
    restored = denoise_full_file(
        diffusion, model, matrix, crop_width=128, batch_size=4, steps=2
    )
    assert restored.shape == (128, 3600)
    assert restored.dtype == np.uint8

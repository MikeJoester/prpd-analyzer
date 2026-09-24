from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Diffusion.prpd_diffusion.diffusion.schedule import SCHEDULE_KINDS, DiffusionSchedule, make_betas


def test_betas_are_increasing_and_bounded():
    for kind in SCHEDULE_KINDS:
        betas = make_betas(kind, 1000)
        assert betas.shape == (1000,)
        assert betas.min() > 0.0 and betas.max() < 1.0
        assert np.all(np.diff(betas) > 0)


def test_alphas_cumprod_decreases_to_near_zero():
    for kind in SCHEDULE_KINDS:
        schedule = DiffusionSchedule.create(kind, 1000)
        alphas = schedule.alphas_cumprod
        assert np.all(np.diff(alphas) < 0)
        assert alphas[0] > 0.99
        assert alphas[-1] < 0.01          # t=T에서 사실상 순수 노이즈


def test_q_sample_endpoints():
    schedule = DiffusionSchedule.create("cosine", 1000)
    rng = np.random.default_rng(0)
    x_start = rng.normal(size=(128, 64)).astype(np.float32)
    noise = rng.normal(size=x_start.shape).astype(np.float32)

    early = schedule.q_sample(x_start, 0, noise)
    assert np.corrcoef(early.ravel(), x_start.ravel())[0, 1] > 0.99

    late = schedule.q_sample(x_start, schedule.timesteps - 1, noise)
    assert np.corrcoef(late.ravel(), noise.ravel())[0, 1] > 0.99


def test_posterior_terms_are_finite_and_nonnegative():
    for kind in SCHEDULE_KINDS:
        schedule = DiffusionSchedule.create(kind, 200)
        assert np.all(schedule.posterior_variance >= 0)
        assert np.all(np.isfinite(schedule.posterior_log_variance_clipped))
        assert np.all(np.isfinite(schedule.posterior_mean_coef1))
        assert np.all(np.isfinite(schedule.posterior_mean_coef2))


def test_posterior_mean_reconstructs_x_start_at_t_zero():
    schedule = DiffusionSchedule.create("cosine", 500)
    rng = np.random.default_rng(1)
    x_start = rng.normal(size=(16,)).astype(np.float64)
    x_t = schedule.q_sample(x_start, 0, rng.normal(size=(16,)))
    mean = (
        schedule.posterior_mean_coef1[0] * x_start + schedule.posterior_mean_coef2[0] * x_t
    )
    assert np.allclose(mean, x_start, atol=1e-6)


def test_unknown_schedule_raises():
    try:
        make_betas("bogus", 100)
    except ValueError:
        return
    raise AssertionError("알 수 없는 schedule은 ValueError여야 합니다.")

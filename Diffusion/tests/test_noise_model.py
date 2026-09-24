from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Diffusion.prpd_diffusion.data.noise_model import (
    NoiseAugmentConfig,
    augment_noise,
    mix,
)


def _pair(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    clean = rng.integers(0, 200, size=(128, 256), dtype=np.uint8)
    noise = rng.integers(0, 200, size=(128, 256), dtype=np.uint8)
    return clean, noise


def test_maximum_mix_dominates_both_inputs():
    clean, noise = _pair()
    mixed = mix(clean, noise, "maximum")
    assert mixed.dtype == np.uint8
    assert np.all(mixed >= clean) and np.all(mixed >= noise)


def test_additive_mix_clips_at_255():
    clean = np.full((4, 4), 200, dtype=np.uint8)
    noise = np.full((4, 4), 200, dtype=np.uint8)
    mixed = mix(clean, noise, "additive")
    assert mixed.max() == 255          # uint8 wrap-around(=145)이 아니어야 한다
    assert mixed.dtype == np.uint8


def test_quadrature_mix_is_bounded():
    clean, noise = _pair(1)
    mixed = mix(clean, noise, "quadrature")
    assert mixed.min() >= 0 and mixed.max() <= 255
    assert np.all(mixed >= np.maximum(clean, noise) - 1)  # 반올림 오차 허용


def test_shape_mismatch_raises():
    try:
        mix(np.zeros((4, 4), np.uint8), np.zeros((4, 5), np.uint8))
    except ValueError:
        return
    raise AssertionError("shape 불일치는 ValueError여야 합니다.")


def test_unknown_mode_raises():
    clean, noise = _pair()
    for call in (lambda: mix(clean, noise, "bogus"), lambda: NoiseAugmentConfig(mode="bogus").validate()):
        try:
            call()
        except ValueError:
            continue
        raise AssertionError("알 수 없는 mode는 ValueError여야 합니다.")


def test_augment_is_reproducible_for_same_seed():
    _, noise = _pair(2)
    config = NoiseAugmentConfig()
    left = augment_noise(noise, config, np.random.default_rng(7))
    right = augment_noise(noise, config, np.random.default_rng(7))
    assert np.array_equal(left, right)
    assert left.shape == noise.shape and left.dtype == np.uint8


def test_gain_scales_amplitude():
    noise = np.full((8, 8), 100, dtype=np.uint8)
    config = NoiseAugmentConfig(gain_min=2.0, gain_max=2.0, phase_roll=False, time_roll=False)
    scaled = augment_noise(noise, config, np.random.default_rng(0))
    assert np.all(scaled == 200)


def test_config_validation_rejects_bad_ranges():
    for bad in (
        NoiseAugmentConfig(gain_min=0.0),
        NoiseAugmentConfig(gain_min=2.0, gain_max=1.0),
        NoiseAugmentConfig(dropout_probability=1.5),
    ):
        try:
            bad.validate()
        except ValueError:
            continue
        raise AssertionError(f"잘못된 설정을 통과시켰습니다: {bad}")

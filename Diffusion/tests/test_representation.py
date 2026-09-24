from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Diffusion.prpd_diffusion.data import representation as repr_module


def _matrix(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(128, 3600), dtype=np.uint8)


def test_value_roundtrip_is_exact():
    values = np.arange(256, dtype=np.uint8).reshape(1, 256)
    restored = repr_module.from_model_output(repr_module.to_model_input(values))
    assert np.array_equal(restored, values)


def test_model_input_range():
    scaled = repr_module.to_model_input(_matrix())
    assert scaled.dtype == np.float32
    assert scaled.min() >= -1.0 and scaled.max() <= 1.0


def test_random_crop_shape_and_bounds():
    matrix = _matrix()
    rng = np.random.default_rng(1)
    crop = repr_module.random_time_crop(matrix, 256, rng)
    assert crop.shape == (128, 256)
    # 같은 seed면 같은 구간이 나오고, 그 구간은 원본의 연속 slice여야 한다.
    start = int(np.random.default_rng(1).integers(0, 3600 - 256 + 1))
    assert np.array_equal(crop, matrix[:, start : start + 256])


def test_crop_wider_than_signal_raises():
    try:
        repr_module.random_time_crop(_matrix(), 5000, np.random.default_rng(0))
    except ValueError:
        return
    raise AssertionError("crop width가 시간 길이보다 크면 ValueError여야 합니다.")


def test_tile_and_stitch_roundtrip():
    matrix = _matrix(2)
    windows, pad = repr_module.tile_time_windows(matrix, 256)
    assert windows.shape == (15, 128, 256)   # 3600 = 256*14 + 16 → 마지막 window는 padding
    assert pad == 240
    restored = repr_module.stitch_time_windows(windows, pad)
    assert np.array_equal(restored, matrix)


def test_tile_exact_multiple_has_no_padding():
    matrix = _matrix(3)[:, :3584]            # 3584 = 256 * 14
    windows, pad = repr_module.tile_time_windows(matrix, 256)
    assert pad == 0
    assert np.array_equal(repr_module.stitch_time_windows(windows, pad), matrix)


def test_time_pool_reduces_time_axis():
    matrix = _matrix(4)
    pooled = repr_module.time_pool(matrix, 30, mode="max")
    assert pooled.shape == (128, 120)
    assert pooled.max() <= 255


def test_phase_profiles_match_analyzer_definition():
    matrix = _matrix(5)
    mean_profile, max_profile = repr_module.phase_profiles(matrix)
    assert mean_profile.shape == (128,) and max_profile.shape == (128,)
    assert np.allclose(mean_profile, matrix.mean(axis=1))
    assert np.array_equal(max_profile, matrix.max(axis=1))

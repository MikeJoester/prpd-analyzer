"""Composite Industrial Noise 생성기 테스트 (`ARDD-2025` A4).

성분별로 "정말 그 성분인가"를 스펙트럼으로 확인한다. 진폭만 맞고 스펙트럼이 틀리면
백색 잡음 세 개를 더한 것과 다를 바 없다.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Diffusion.prpd_diffusion.data.cin_noise import (
    CinNoiseConfig,
    apply_burst_missing,
    harmonic_field,
    measured_snr_db,
    noise_gain_for_snr,
    pink_field,
    sample_cin_field,
    sample_cin_matrix,
)

SHAPE = (128, 600)  # 시간축을 줄여 테스트를 빠르게. 축 의미는 동일하다.


def test_config_validation_rejects_bad_settings():
    CinNoiseConfig().validate()
    for bad in (
        CinNoiseConfig(white_sigma=-1.0),
        CinNoiseConfig(harmonic_amplitudes=(1.0, 2.0), harmonic_orders=(1,)),
        CinNoiseConfig(harmonic_axis="frequency"),
        CinNoiseConfig(pink_axis="frequency"),
        CinNoiseConfig(harmonic_orders=(0, 3, 5)),
        CinNoiseConfig(white_sigma=0.0, pink_sigma=0.0, harmonic_amplitudes=(0.0, 0.0, 0.0)),
    ):
        try:
            bad.validate()
        except ValueError:
            continue
        raise AssertionError(f"잘못된 설정을 거부해야 합니다: {bad}")


def test_same_seed_reproduces_identical_noise():
    config = CinNoiseConfig()
    first = sample_cin_matrix(np.random.default_rng(7), config, SHAPE)
    second = sample_cin_matrix(np.random.default_rng(7), config, SHAPE)
    assert np.array_equal(first, second)
    other = sample_cin_matrix(np.random.default_rng(8), config, SHAPE)
    assert not np.array_equal(first, other)


def test_pink_spectrum_slope_is_close_to_minus_one():
    """`1/f` 성분의 PSD 기울기는 log-log에서 약 −1이어야 한다."""
    values = pink_field(np.random.default_rng(0), (64, 4096), axis="time")
    spectrum = np.fft.rfft(values, axis=1)
    power = (np.abs(spectrum) ** 2).mean(axis=0)
    # DC와 최고주파 부근은 제외하고 중간 대역만 적합한다.
    low, high = 4, power.size // 4
    frequencies = np.arange(power.size)[low:high]
    slope = np.polyfit(np.log(frequencies), np.log(power[low:high]), 1)[0]
    assert -1.35 < slope < -0.65, f"pink 기울기가 −1에서 너무 멉니다: {slope:.3f}"


def test_white_spectrum_is_flat():
    config = CinNoiseConfig(white_sigma=5.0, pink_sigma=0.0, harmonic_amplitudes=(0.0, 0.0, 0.0))
    values = sample_cin_field(np.random.default_rng(1), config, (64, 4096))
    power = (np.abs(np.fft.rfft(values, axis=1)) ** 2).mean(axis=0)
    low, high = 4, power.size // 4
    frequencies = np.arange(power.size)[low:high]
    slope = np.polyfit(np.log(frequencies), np.log(power[low:high]), 1)[0]
    assert abs(slope) < 0.2, f"백색 성분의 기울기가 0이 아닙니다: {slope:.3f}"


def test_harmonics_land_on_phase_axis_orders():
    """기본파·3·5고조파는 128 위상 bin에서 정확히 1/3/5 주기여야 한다."""
    values = harmonic_field(
        np.random.default_rng(2),
        SHAPE,
        amplitudes=(10.0, 4.0, 2.0),
        orders=(1, 3, 5),
        axis="phase",
    )
    spectrum = np.abs(np.fft.rfft(values[:, 0]))
    occupied = {int(index) for index in np.argsort(spectrum)[-3:]}
    assert occupied == {1, 3, 5}, f"에너지가 실린 위상 주파수: {sorted(occupied)}"
    # 시간축으로는 일정해야 한다(전원주파수는 위상에 고정된다).
    assert np.allclose(values[:, 0], values[:, -1])


def test_harmonic_axis_can_be_switched_to_time():
    values = harmonic_field(
        np.random.default_rng(3), SHAPE, (10.0,), (3,), axis="time", randomize_phase=False
    )
    spectrum = np.abs(np.fft.rfft(values[0]))
    assert int(np.argmax(spectrum)) == 3
    assert np.allclose(values[0], values[-1])


def test_matrix_is_nonnegative_uint8():
    matrix = sample_cin_matrix(np.random.default_rng(4), CinNoiseConfig(), SHAPE)
    assert matrix.dtype == np.uint8
    assert matrix.shape == SHAPE
    assert matrix.max() <= 255


def test_noise_gain_reaches_target_snr():
    rng = np.random.default_rng(5)
    clean = rng.integers(0, 256, size=SHAPE).astype(np.float64)
    noise = sample_cin_field(np.random.default_rng(6), CinNoiseConfig(), SHAPE)
    for target in (3.0, 0.0, -6.0, -12.0):
        gain = noise_gain_for_snr(clean, noise, target)
        assert abs(measured_snr_db(clean, noise * gain) - target) < 1e-6


def test_target_snr_requires_clean_matrix():
    config = CinNoiseConfig(target_snr_db=-6.0)
    try:
        sample_cin_matrix(np.random.default_rng(0), config, SHAPE)
    except ValueError:
        return
    raise AssertionError("target_snr_db 사용 시 clean 없이 호출하면 실패해야 합니다.")


def test_burst_missing_zeroes_contiguous_time_window():
    matrix = np.full(SHAPE, 7, dtype=np.uint8)
    result = apply_burst_missing(matrix, np.random.default_rng(9), 0.10)
    assert matrix.max() == 7, "입력을 변경하면 안 됩니다."
    zero_columns = np.flatnonzero((result == 0).all(axis=0))
    assert zero_columns.size == int(round(SHAPE[1] * 0.10))
    assert np.array_equal(zero_columns, np.arange(zero_columns[0], zero_columns[-1] + 1))


def test_burst_missing_zero_fraction_is_identity():
    matrix = np.full(SHAPE, 3, dtype=np.uint8)
    assert np.array_equal(apply_burst_missing(matrix, np.random.default_rng(0), 0.0), matrix)

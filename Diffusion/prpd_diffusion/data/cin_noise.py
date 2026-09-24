"""Composite Industrial Noise (CIN) — 파라메트릭 노이즈 생성기. numpy만 사용.

`DIFFUSION.md` 18.3의 `cin_noise` 후보, 18.4의 실험 A4. 근거는 `ARDD-2025`
(Chen et al., Sci. Rep. 15:42848)의 로버스트니스 실험이며, 논문은 측정 노이즈가 아니라
다음 세 성분을 합성해 주입한다.

    1) 백색 잡음          — 배경 열잡음
    2) pink(1/f) 잡음     — 저주파 전자기기 잡음
    3) 전원주파수 협대역   — 기본파 + 3·5고조파

왜 필요한가: ARDD 1D 트랙은 config에서 `Lab PD`/`Field PD`만 선택하므로 노이즈 그룹을
쓰지 않는다. 실측 pool 합성(`noise_model.NoiseBank`) 대신 파라메트릭 CIN을 쓰며,
이는 우회가 아니라 **논문이 실제로 쓴 방식**이다.

축 매핑 — 논문의 신호는 시간 파형 1개지만 우리 raw는 `(위상 128, 시간 3600)`이고 두 축의
물리적 의미가 다르다. `128 위상 bin = AC 1주기`, `3600 시간 열 ≈ 60 Hz × 60 s = 1분치 사이클`이다.

| 성분 | 논문 | 우리 축 | 근거 |
|---|---|---|---|
| 백색 | 시간축 | 양축 i.i.d. | 배경 열잡음 |
| pink(1/f) | 시간축 | 시간축(사이클 간 드리프트) | 저주파 전자기기 잡음 |
| 기본파 | 50 Hz | 위상축 1 cycle | 128 bin = AC 1주기이므로 전원주파수는 위상축에 고정 |
| 3·5고조파 | 150/250 Hz | 위상축 3·5 cycle | 동일 |

이 매핑은 설계 판단이므로 `run_config.json`과 리포트에 기록한다. 대안(`harmonic_axis="time"`,
`pink_axis="phase"`)도 config로 선택할 수 있어 비교 항목으로 남길 수 있다.

부호 처리: 세 성분의 합은 부호 있는 실수지만, PRPS 한 bin 값은 해당 위상 window에서 검출된
**피크 진폭**이라 음수 개념이 없다. 따라서 `|·|`을 취해 포락선으로 만든 뒤 `[0, 255]`로 clip해
uint8 노이즈 행렬을 만든다. 이 결정은 `noise_model.mix`의 `maximum` 규약과 짝을 이룬다.

**주의 — `|·|`이 고조파 차수를 바꾼다 (2026-08-25 실측).** 위 표는 `sample_cin_field`가
만드는 **부호 있는** 장의 성질이다. `sample_cin_matrix`가 취하는 전파 정류 `|·|`은 주기를
절반으로 줄이므로, 모델이 실제로 보는 위상 profile에서 1/3/5 차 성분은 **정확히 0**이 되고
에너지가 2/4/6차로 옮겨간다. `harmonic_amplitudes=(10, 4, 2)`, 백색·pink 0일 때 측정값:

```text
부호 있는 장   k1 0.625  k3 0.250  k5 0.125   (k2/k4/k6 = 0)
|·| 후 profile k1 0.000  k3 0.000  k5 0.000   k2 0.489  k4 0.293  k6 0.053
```

`tests/test_cin_noise.py`는 `harmonic_field`(부호 있는 성분)만 검증하므로 이 간극을
잡지 못한다. 실측 노이즈 그룹의 위상 profile은 1/3/5차 에너지 비중이 중앙값 약 0.06
(`Lab Noise` 0.062, `Field Noise` 0.063)으로 애초에 낮으니, 정류된 CIN이 실측과 크게
어긋난다고 단정할 수는 없다. 다만 **문서의 축 매핑 표를 "모델이 보는 노이즈"의 설명으로
읽으면 안 된다.** 차수를 보존하려면 `|·|` 대신 offset(예: `field + c`)을 쓰는 대안을
비교 항목으로 두어야 하며, 이는 아직 결정되지 않았다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .representation import PHASE_BINS, TIME_SAMPLES

AXES = ("phase", "time")
_AXIS_INDEX = {"phase": 0, "time": 1}
CIN_VERSION = "cin_v1"


@dataclass(frozen=True)
class CinNoiseConfig:
    """CIN 성분별 진폭과 축 배치. 진폭 단위는 uint8 스케일(0~255)이다."""

    white_sigma: float = 6.0
    pink_sigma: float = 4.0
    harmonic_amplitudes: tuple[float, ...] = (10.0, 4.0, 2.0)
    harmonic_orders: tuple[int, ...] = (1, 3, 5)
    harmonic_axis: str = "phase"
    pink_axis: str = "time"
    randomize_harmonic_phase: bool = True
    target_snr_db: float | None = None  # 지정하면 clean 대비 SNR로 전체 진폭을 재조정

    def validate(self) -> None:
        if self.white_sigma < 0 or self.pink_sigma < 0:
            raise ValueError("white_sigma / pink_sigma는 0 이상이어야 합니다.")
        if len(self.harmonic_amplitudes) != len(self.harmonic_orders):
            raise ValueError(
                "harmonic_amplitudes와 harmonic_orders의 길이가 다릅니다: "
                f"{len(self.harmonic_amplitudes)} vs {len(self.harmonic_orders)}"
            )
        if any(amplitude < 0 for amplitude in self.harmonic_amplitudes):
            raise ValueError("harmonic_amplitudes는 0 이상이어야 합니다.")
        if any(order < 1 for order in self.harmonic_orders):
            raise ValueError("harmonic_orders는 1 이상의 정수여야 합니다.")
        for name, axis in (("harmonic_axis", self.harmonic_axis), ("pink_axis", self.pink_axis)):
            if axis not in AXES:
                raise ValueError(f"알 수 없는 {name}: {axis} (가능: {AXES})")
        if self.white_sigma == 0 and self.pink_sigma == 0 and not any(self.harmonic_amplitudes):
            raise ValueError("모든 성분의 진폭이 0입니다. 노이즈가 생성되지 않습니다.")

    def to_dict(self) -> dict:
        return {
            "version": CIN_VERSION,
            "white_sigma": self.white_sigma,
            "pink_sigma": self.pink_sigma,
            "harmonic_amplitudes": list(self.harmonic_amplitudes),
            "harmonic_orders": list(self.harmonic_orders),
            "harmonic_axis": self.harmonic_axis,
            "pink_axis": self.pink_axis,
            "randomize_harmonic_phase": self.randomize_harmonic_phase,
            "target_snr_db": self.target_snr_db,
        }


# ------------------------------------------------------------------ 성분 생성


def pink_field(rng: np.random.Generator, shape: tuple[int, int], axis: str) -> np.ndarray:
    """지정한 축을 따라 `1/f` 스펙트럼을 가진 잡음장. 단위 표준편차로 정규화한다.

    백색 잡음을 rfft한 뒤 진폭에 `1/sqrt(f)`를 곱하고 역변환한다. 다른 축의 각 인덱스는
    독립 실현이므로 축 방향으로만 상관이 생긴다.
    """
    if axis not in AXES:
        raise ValueError(f"알 수 없는 axis: {axis} (가능: {AXES})")
    index = _AXIS_INDEX[axis]
    length = shape[index]
    white = rng.standard_normal(shape)
    spectrum = np.fft.rfft(white, axis=index)
    frequencies = np.arange(spectrum.shape[index], dtype=np.float64)
    scale = np.ones_like(frequencies)
    scale[1:] = 1.0 / np.sqrt(frequencies[1:])
    scale[0] = 0.0  # DC 제거 — 일정한 오프셋은 드리프트가 아니다
    broadcast = scale.reshape(-1, 1) if index == 0 else scale.reshape(1, -1)
    shaped = np.fft.irfft(spectrum * broadcast, n=length, axis=index)
    deviation = float(shaped.std())
    if deviation > 0:
        shaped = shaped / deviation
    return shaped.astype(np.float32)


def harmonic_field(
    rng: np.random.Generator,
    shape: tuple[int, int],
    amplitudes: tuple[float, ...],
    orders: tuple[int, ...],
    axis: str,
    randomize_phase: bool = True,
) -> np.ndarray:
    """전원주파수 기본파와 고조파의 합. 지정 축에서 `order` 주기를 갖는다.

    위상 오프셋을 무작위로 뽑아 노이즈가 항상 같은 위상 구간에만 실리는 편향을 막는다.
    """
    if axis not in AXES:
        raise ValueError(f"알 수 없는 axis: {axis} (가능: {AXES})")
    index = _AXIS_INDEX[axis]
    length = shape[index]
    positions = np.arange(length, dtype=np.float64) / float(length)
    total = np.zeros(length, dtype=np.float64)
    for amplitude, order in zip(amplitudes, orders):
        if amplitude == 0:
            continue
        offset = float(rng.uniform(0.0, 2.0 * np.pi)) if randomize_phase else 0.0
        total += amplitude * np.sin(2.0 * np.pi * order * positions + offset)
    broadcast = total.reshape(-1, 1) if index == 0 else total.reshape(1, -1)
    return np.broadcast_to(broadcast, shape).astype(np.float32)


def sample_cin_field(
    rng: np.random.Generator,
    config: CinNoiseConfig,
    shape: tuple[int, int] = (PHASE_BINS, TIME_SAMPLES),
) -> np.ndarray:
    """세 성분을 합친 **부호 있는** 잡음장 `(128, 3600)` float32."""
    config.validate()
    total = np.zeros(shape, dtype=np.float32)
    if config.white_sigma > 0:
        total += (config.white_sigma * rng.standard_normal(shape)).astype(np.float32)
    if config.pink_sigma > 0:
        total += config.pink_sigma * pink_field(rng, shape, config.pink_axis)
    if any(config.harmonic_amplitudes):
        total += harmonic_field(
            rng,
            shape,
            config.harmonic_amplitudes,
            config.harmonic_orders,
            config.harmonic_axis,
            config.randomize_harmonic_phase,
        )
    return total


def sample_cin_matrix(
    rng: np.random.Generator,
    config: CinNoiseConfig,
    shape: tuple[int, int] = (PHASE_BINS, TIME_SAMPLES),
    clean: np.ndarray | None = None,
) -> np.ndarray:
    """`noise_model.mix`에 넣을 uint8 노이즈 행렬.

    부호 있는 잡음장의 포락선 `|·|`을 취해 `[0, 255]`로 clip한다(모듈 docstring 참조).
    `config.target_snr_db`가 지정되고 `clean`이 주어지면, clip 전에 전체 진폭을 조정해
    목표 SNR을 맞춘다(SNR 스윕용).
    """
    field_values = sample_cin_field(rng, config, shape)
    if config.target_snr_db is not None:
        if clean is None:
            raise ValueError("target_snr_db를 쓰려면 clean 행렬이 필요합니다.")
        field_values = field_values * noise_gain_for_snr(clean, field_values, config.target_snr_db)
    return np.clip(np.abs(field_values), 0.0, 255.0).astype(np.uint8)


# ------------------------------------------------------------------ SNR / 결측


def signal_power(array: np.ndarray) -> float:
    """평균 제곱 전력."""
    values = np.asarray(array, dtype=np.float64)
    return float(np.mean(values * values))


def noise_gain_for_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> float:
    """`10·log10(P_clean / P_(gain·noise)) == snr_db`가 되는 gain.

    논문의 로버스트니스 프로토콜(+3 → −12 dB, 3 dB 간격)에 그대로 쓴다.
    """
    noise_power = signal_power(noise)
    if noise_power <= 0:
        raise ValueError("노이즈 전력이 0이라 SNR을 맞출 수 없습니다.")
    clean_power = signal_power(clean)
    if clean_power <= 0:
        raise ValueError("clean 전력이 0이라 SNR을 맞출 수 없습니다.")
    target_noise_power = clean_power / (10.0 ** (snr_db / 10.0))
    return float(np.sqrt(target_noise_power / noise_power))


def measured_snr_db(clean: np.ndarray, noise: np.ndarray) -> float:
    """실제로 달성된 SNR(dB). 테스트와 리포트용."""
    return float(10.0 * np.log10(signal_power(clean) / signal_power(noise)))


def apply_burst_missing(
    matrix: np.ndarray,
    rng: np.random.Generator,
    fraction: float,
) -> np.ndarray:
    """시간축의 연속 구간을 0으로 만든다(논문의 burst 결측 실험, 5~15%).

    반환은 새 배열이며 입력은 변경하지 않는다.
    """
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction은 0 이상 1 미만이어야 합니다.")
    result = np.asarray(matrix).copy()
    width = int(round(result.shape[1] * fraction))
    if width == 0:
        return result
    start = int(rng.integers(0, result.shape[1] - width + 1))
    result[:, start : start + width] = 0
    return result

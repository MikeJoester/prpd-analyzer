"""raw `(128, 3600)` uint8 ↔ 모델 입력 표현 변환.

`128 x 3600`을 통째로 U-Net에 넣는 것은 비현실적이므로(샘플당 460,800 픽셀, 종횡비 28:1)
기본 표현은 **시간축 crop**이다.

    crop_v1 : 학습은 (128, CROP_WIDTH) 무작위 window,
              추론은 전체 파일을 window로 타일링 후 다시 이어붙임 → 정보 손실 없음

노이즈 제거는 입력과 같은 해상도의 출력을 요구하므로 다운샘플 표현을 기본으로 쓰지 않는다.
`time_pool`과 `prpd_histogram`은 시각화·분포 평가용 보조 표현이다.

값 범위 규약: 모델 입력/출력은 항상 float32 `[-1, 1]`. 저장은 항상 uint8 `[0, 255]`.
"""

from __future__ import annotations

import numpy as np

PHASE_BINS = 128
TIME_SAMPLES = 3600
REPRESENTATION_VERSION = "crop_v1"
DEFAULT_CROP_WIDTH = 256


# ------------------------------------------------------------------ 값 변환


def to_model_input(matrix: np.ndarray) -> np.ndarray:
    """uint8 `[0, 255]` → float32 `[-1, 1]`."""
    return matrix.astype(np.float32) / 127.5 - 1.0


def from_model_output(array: np.ndarray) -> np.ndarray:
    """float32 `[-1, 1]` → uint8 `[0, 255]` (범위를 벗어난 값은 clip)."""
    scaled = (np.asarray(array, dtype=np.float32) + 1.0) * 127.5
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


# --------------------------------------------------------------- 시간축 crop


def random_time_crop(matrix: np.ndarray, width: int, rng: np.random.Generator) -> np.ndarray:
    """시간축에서 길이 `width` window를 무작위로 잘라낸다."""
    total = matrix.shape[1]
    if width > total:
        raise ValueError(f"crop width {width} > 시간 길이 {total}")
    start = int(rng.integers(0, total - width + 1))
    return matrix[:, start : start + width]


def tile_time_windows(matrix: np.ndarray, width: int) -> tuple[np.ndarray, int]:
    """전체 파일을 겹치지 않는 window로 나눈다.

    Returns:
        `(windows, pad)` — windows shape `(n_windows, 128, width)`,
        `pad`는 마지막 window를 채우기 위해 덧붙인 열 수(복원 시 잘라냄).
    """
    total = matrix.shape[1]
    remainder = (-total) % width
    if remainder:
        matrix = np.pad(matrix, ((0, 0), (0, remainder)), mode="edge")
    windows = matrix.reshape(matrix.shape[0], -1, width).transpose(1, 0, 2)
    return np.ascontiguousarray(windows), remainder


def stitch_time_windows(windows: np.ndarray, pad: int) -> np.ndarray:
    """`tile_time_windows` 결과를 원래 시간 길이로 복원한다."""
    merged = windows.transpose(1, 0, 2).reshape(windows.shape[1], -1)
    return merged[:, : merged.shape[1] - pad] if pad else merged


# ------------------------------------------------------------- 보조 표현


def time_pool(matrix: np.ndarray, factor: int, mode: str = "max") -> np.ndarray:
    """시간축을 `factor`배 축소한다. 시각화·빠른 실험용."""
    total = matrix.shape[1]
    if total % factor:
        raise ValueError(f"시간 길이 {total}이 factor {factor}로 나누어떨어지지 않습니다.")
    blocks = matrix.reshape(matrix.shape[0], total // factor, factor)
    if mode == "max":
        return blocks.max(axis=2)
    if mode == "mean":
        return blocks.mean(axis=2)
    raise ValueError(f"알 수 없는 pooling mode: {mode}")


def prpd_histogram(matrix: np.ndarray, amplitude_bins: int = 128) -> np.ndarray:
    """`(128, 3600)` PRPS → `(phase, amplitude)` 카운트 히스토그램.

    분포 비교·리포트 그림용. 진폭 0(=검출 없음)은 제외한다.
    """
    edges = np.linspace(0, 256, amplitude_bins + 1)
    histogram = np.zeros((matrix.shape[0], amplitude_bins), dtype=np.int32)
    for phase in range(matrix.shape[0]):
        values = matrix[phase]
        counts, _ = np.histogram(values[values > 0], bins=edges)
        histogram[phase] = counts
    return histogram


def phase_profiles(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Analyzer와 동일한 위상별 평균/최대 profile(각 128차원)."""
    return matrix.mean(axis=1).astype(np.float32), matrix.max(axis=1).astype(np.float32)

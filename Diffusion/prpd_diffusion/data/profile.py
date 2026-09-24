"""ARDD 1D 트랙의 표현: raw `(128, 3600)` → 위상 profile `(2, 128)`.

`DIFFUSION.md` 19절. 2D baseline(`representation.crop_v1`)과 달리 이 트랙은 시간축을
mean/max로 접어 **1차원 신호**를 만든다. 논문(ARDD-2025)이 1D 시간파형을 입력으로 쓴 것을
우리 데이터에 옮긴 형태다.

    profile_v1 : (128, 3600) uint8 → (2, 128) float32
                 채널 0 = 위상별 시간 평균, 채널 1 = 위상별 시간 최대

두 채널을 이어붙이면 Analyzer의 256차원 mean/max feature와 **정확히 같은 벡터**가 된다
(`to_feature_vector` ↔ `prpd_analyzer.features.make_features`). 그래야 unpaired 분포 지표를
Analyzer 정의 그대로 쓸 수 있다.

알아 둘 것 — 이 표현은 비가역이다:

* `3600 → 1` 축소이므로 복원된 profile을 `128×3600`으로 되돌릴 수 없다. 따라서 8절의
  raw-space `mae`/`psnr`은 이 트랙에 적용되지 않고, profile 공간 지표로 대체한다.
* mean 채널은 3,600 사이클 평균이라 시간축 노이즈에 원래 둔감하다. 반대로 max 채널은
  노이즈 피크에 지배된다. **채널별 개선폭을 반드시 분리해 보고할 것.**
* `claude.md` 11절은 256 feature를 딥러닝 입력으로 쓰지 않는다고 규정한다. 이 트랙은
  그 규칙의 명시적 예외이며, 각 run 산출물에 `feature_input_exception`으로 기록한다.

값 범위 규약: 모델 입력/출력은 float32 `[-1, 1]`. profile의 자연 범위는 `[0, 255]`이며
**uint8로 양자화하지 않는다** — mean 채널은 본질적으로 소수값이므로 반올림하면 Analyzer의
`make_features`와 값이 달라져 분포 지표 비교가 성립하지 않는다.
"""

from __future__ import annotations

import numpy as np

from .representation import PHASE_BINS, phase_profiles

PROFILE_VERSION = "profile_v1"
PROFILE_KINDS = ("profile_v1", "phase_row_v1")
CHANNELS = ("mean", "max")
PROFILE_CHANNELS = len(CHANNELS)
VALUE_RANGE = 255.0


# ------------------------------------------------------------------ raw → profile


def profile_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """`(128, 3600)` uint8 → `(2, 128)` float32, 값 범위 `[0, 255]`.

    Analyzer(`prpd_analyzer.io`)와 같은 정의: 채널 0 = `matrix.mean(axis=1)`,
    채널 1 = `matrix.max(axis=1)`.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != PHASE_BINS:
        raise ValueError(f"raw 행렬 shape이 (128, T)가 아닙니다: {matrix.shape}")
    mean_profile, max_profile = phase_profiles(matrix)
    return np.stack((mean_profile, max_profile), axis=0).astype(np.float32)


def profiles_from_matrices(matrices: np.ndarray) -> np.ndarray:
    """`(N, 128, 3600)` → `(N, 2, 128)` float32. 한 장이면 `(1, 2, 128)`."""
    matrices = np.asarray(matrices)
    if matrices.ndim == 2:
        matrices = matrices[None, ...]
    if matrices.ndim != 3:
        raise ValueError(f"입력 shape이 (N, 128, T)가 아닙니다: {matrices.shape}")
    return np.stack([profile_from_matrix(item) for item in matrices], axis=0)


# ------------------------------------------------------------------ 값 변환


def to_model_input(profile: np.ndarray) -> np.ndarray:
    """profile `[0, 255]` → float32 `[-1, 1]`."""
    return np.asarray(profile, dtype=np.float32) / (VALUE_RANGE / 2.0) - 1.0


def from_model_output(array: np.ndarray) -> np.ndarray:
    """모델 출력 `[-1, 1]` → profile `[0, 255]` float32 (범위 밖은 clip).

    uint8로 반올림하지 않는다(모듈 docstring 참조).
    """
    scaled = (np.asarray(array, dtype=np.float32) + 1.0) * (VALUE_RANGE / 2.0)
    return np.clip(scaled, 0.0, VALUE_RANGE).astype(np.float32)


def to_profile(matrix: np.ndarray) -> np.ndarray:
    """raw 행렬 → 모델 입력 `(2, 128)` float32 `[-1, 1]`."""
    return to_model_input(profile_from_matrix(matrix))


# ------------------------------------------------------------------ Analyzer 호환


def to_feature_vector(profile: np.ndarray) -> np.ndarray:
    """profile `(..., 2, 128)` `[0, 255]` → Analyzer의 256차원 feature `(..., 256)`.

    `prpd_analyzer.features.make_features(mean, max)`와 **정확히 같은 값·같은 순서**여야 한다
    (mean 128개 다음 max 128개). 다르면 Fréchet/MMD 비교가 무의미해진다.
    """
    array = np.asarray(profile, dtype=np.float32)
    if array.shape[-2:] != (PROFILE_CHANNELS, PHASE_BINS):
        raise ValueError(f"profile shape이 (..., 2, 128)이 아닙니다: {array.shape}")
    return array.reshape(*array.shape[:-2], PROFILE_CHANNELS * PHASE_BINS)


def feature_matrix(profiles: np.ndarray) -> np.ndarray:
    """`(N, 2, 128)` `[0, 255]` → `(N, 256)` float32."""
    array = np.asarray(profiles, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    return to_feature_vector(array)


# ------------------------------------------------------------------ 표현 선택


def resolve_kind(kind: str) -> str:
    """표현 종류를 검증한다.

    `phase_row_v1`(위상 행 1개 = 길이 3600, 논문 길이와 정확히 일치하고 무손실)은 아직
    구현하지 않았다. profile 표현의 비가역성이 병목으로 드러나면 여기서 전환한다.
    """
    if kind == "profile_v1":
        return kind
    if kind == "phase_row_v1":
        raise NotImplementedError(
            "phase_row_v1은 아직 구현하지 않았습니다. DIFFUSION.md 19절의 전환 계획을 참조하세요."
        )
    raise ValueError(f"알 수 없는 representation kind: {kind} (가능: {PROFILE_KINDS})")

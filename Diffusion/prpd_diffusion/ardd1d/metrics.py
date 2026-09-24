"""ARDD 1D 트랙의 지표. numpy만 사용.

`DIFFUSION.md` 8절의 raw-space `mae`/`psnr`은 이 트랙에 적용되지 않는다. profile 표현이
비가역(`3600 → 1`)이라 `128×3600`을 복원할 수 없기 때문이다. 대신 **profile 공간**에서
같은 역할의 지표를 정의한다.

핵심 규칙 — **mean 채널과 max 채널을 합치지 않는다.**
mean 채널은 3,600 사이클 평균이라 시간축 노이즈에 원래 둔감하고, max 채널은 노이즈 피크에
지배된다. 두 채널을 평균 내면 "max는 크게 좋아졌는데 mean은 그대로"인 정상적인 결과가
"절반만 좋아짐"으로 뭉개진다.

unpaired 분포 지표는 2D 트랙과 **완전히 같은 함수**(`evaluation.metrics`)를 쓴다. 이 트랙은
모델 출력이 곧 Analyzer의 256차원 feature이므로 변환 없이 그대로 넣을 수 있다.
"""

from __future__ import annotations

import numpy as np

from ..data.profile import CHANNELS, PHASE_BINS, PROFILE_CHANNELS, feature_matrix
from ..evaluation.metrics import distribution_shift, frechet_distance, mmd_rbf, standardize

__all__ = [
    "channel_errors",
    "paired_profile_metrics",
    "distribution_shift",
    "feature_matrix",
    "frechet_distance",
    "mmd_rbf",
    "standardize",
]

VALUE_RANGE = 255.0


def _as_batch(profiles: np.ndarray) -> np.ndarray:
    array = np.asarray(profiles, dtype=np.float64)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[1:] != (PROFILE_CHANNELS, PHASE_BINS):
        raise ValueError(f"profile shape이 (N, 2, 128)이 아닙니다: {array.shape}")
    return array


def _psnr(mse: float, data_range: float = VALUE_RANGE) -> float:
    if mse <= 0:
        return float("inf")
    return float(10.0 * np.log10(data_range**2 / mse))


def channel_errors(restored: np.ndarray, clean: np.ndarray) -> dict[str, float]:
    """채널별 MAE와 PSNR. 입력은 `(N, 2, 128)` 또는 `(2, 128)`, 값 범위 `[0, 255]`."""
    restored_batch = _as_batch(restored)
    clean_batch = _as_batch(clean)
    if restored_batch.shape != clean_batch.shape:
        raise ValueError(
            f"shape 불일치: restored {restored_batch.shape} vs clean {clean_batch.shape}"
        )
    difference = restored_batch - clean_batch
    metrics: dict[str, float] = {}
    for index, name in enumerate(CHANNELS):
        channel = difference[:, index, :]
        metrics[f"{name}_profile_mae"] = float(np.mean(np.abs(channel)))
        metrics[f"{name}_profile_psnr"] = _psnr(float(np.mean(channel**2)))
    return metrics


def paired_profile_metrics(
    restored: np.ndarray,
    clean: np.ndarray,
    noisy: np.ndarray | None = None,
) -> dict[str, float]:
    """복원 profile의 paired 지표.

    `noisy`를 주면 "아무것도 하지 않았을 때"(=`identity` baseline)의 오차와 채널별 개선폭을
    함께 계산한다. `DIFFUSION.md` 18.1이 요구하는 무처리 기준선이 항상 같은 표에 있어야 한다.
    개선폭이 양수가 아니면 그 채널에서는 모델이 도움이 되지 않은 것이다.
    """
    metrics = channel_errors(restored, clean)
    if noisy is not None:
        baseline = channel_errors(noisy, clean)
        for name in CHANNELS:
            key = f"{name}_profile_mae"
            metrics[f"baseline_{key}"] = baseline[key]
            metrics[f"{key}_improvement"] = baseline[key] - metrics[key]
            metrics[f"baseline_{name}_profile_psnr"] = baseline[f"{name}_profile_psnr"]
    return metrics


def improved_channels(metrics: dict[str, float]) -> list[str]:
    """개선폭이 양수인 채널 목록. 채택 판단(18.6) 보조."""
    return [name for name in CHANNELS if metrics.get(f"{name}_profile_mae_improvement", 0.0) > 0.0]

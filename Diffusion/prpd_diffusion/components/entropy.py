"""국소 정보 엔트로피 `H(x, t)` — `ARDD-2025` Eq.(7).

    H(x, t) = -Σ_i p_i(x, t) · log p_i(x, t)

`p_i`는 위치 `t` 주변 window 안에서 값이 진폭 bin `i`에 들어갈 확률이다.
논문은 정의만 주고 window 크기·bin 수를 밝히지 않았으므로 둘 다 설정으로 노출한다.

구현 방식: 값을 `bins`개로 양자화 → one-hot → window 평균(`avg_pool1d`)으로 `p_i`를 얻는다.
window 안의 표본이 `window`개뿐이므로 실제로 채워질 수 있는 bin은 최대 `min(bins, window)`개다.
따라서 `H_max = log(min(bins, window))`로 정규화해 결과가 `[0, 1]`에 놓이게 한다.

학습 파라미터가 없고 입력의 결정적 함수이므로 `no_grad`로 계산해 그래프에서 분리한다.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

DEFAULT_WINDOW = 9
DEFAULT_BINS = 16


def _pad_1d(x: torch.Tensor, pad: int, circular: bool) -> torch.Tensor:
    if pad == 0:
        return x
    if circular and pad < x.shape[-1]:
        return F.pad(x, (pad, pad), mode="circular")
    return F.pad(x, (pad, pad), mode="replicate")


@torch.no_grad()
def local_entropy(
    x: torch.Tensor,
    window: int = DEFAULT_WINDOW,
    bins: int = DEFAULT_BINS,
    circular: bool = True,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> torch.Tensor:
    """`(B, C, L)` → `(B, 1, L)` 국소 엔트로피, `[0, 1]`로 정규화.

    채널이 여러 개면 채널 평균 신호에서 계산한다. 엔트로피는 "이 위치 주변이 얼마나
    복잡한가"를 재는 것이므로 채널별로 따로 재서 다시 합칠 이유가 없다.

    Args:
        circular: 위상축은 순환 구조이므로 기본값 True. 시간축이나 flat256 표현에서는 False.
        value_range: 양자화 기준 범위. 모델 입력 규약은 `[-1, 1]`이다.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError(f"window는 1 이상의 홀수여야 합니다: {window}")
    if bins < 2:
        raise ValueError(f"bins는 2 이상이어야 합니다: {bins}")
    if x.dim() != 3:
        raise ValueError(f"입력 shape이 (B, C, L)이 아닙니다: {tuple(x.shape)}")

    signal = x.mean(dim=1, keepdim=True)
    low, high = value_range
    scaled = (signal - low) / max(high - low, 1e-12)
    indices = torch.clamp((scaled * bins).long(), 0, bins - 1)  # (B, 1, L)

    one_hot = F.one_hot(indices.squeeze(1), num_classes=bins)   # (B, L, bins)
    one_hot = one_hot.permute(0, 2, 1).to(x.dtype)              # (B, bins, L)

    padded = _pad_1d(one_hot, window // 2, circular)
    probabilities = F.avg_pool1d(padded, kernel_size=window, stride=1)  # (B, bins, L), Σ=1

    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=1, keepdim=True)
    maximum = math.log(min(bins, window)) if min(bins, window) > 1 else 1.0
    return (entropy / maximum).clamp(0.0, 1.0)

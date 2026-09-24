"""형태학적 gradient attention — `ARDD-2025` Eq.(9)~(15). 실험 id `A3` (`DIFFUSION.md` 18.4).

    dilation_s(x)(t) = max_{u∈[-s,s]} x(t+u)         Eq.(9)
    erosion_s(x)(t)  = min_{u∈[-s,s]} x(t+u)         Eq.(10)
    G_s(x)(t) = dilation_s − erosion_s               Eq.(11)
    a_base(t) = max_{s∈S} G_s(x)(t)                  Eq.(14)
    a(t) = σ(a_base(t))                              Eq.(15)

PD 펄스의 **폭과 가장자리 구조**를 강조하는 것이 목적이다. 평탄한 구간에서는 팽창과 침식이
같아 `G=0`, 펄스 경계에서는 `G`가 커진다. 학습 파라미터가 없다(parameter-free attention).

**scales 기본값이 논문과 다른 이유**: 논문의 신호는 길이 3,600이지만 우리 축은 위상 128 bin
(=AC 1주기)이다. 논문 규모의 s를 그대로 쓰면 window가 위상 1주기의 대부분을 덮어 위치 정보가
사라진다. 기본 `(1, 3, 7)`은 window 3/7/15로 위상축의 2.3% / 5.5% / 11.7%에 해당한다.
이 차이는 리포트에 명시한다.

**정규화 옵션**: `a_base ≥ 0`이므로 Eq.(15)를 문자 그대로 적용하면 `σ(a_base) ∈ [0.5, 1)`이고,
`a_base`가 크면 1로 포화해 attention이 사실상 사라진다. 기본값은 논문 그대로(`none`)이되,
샘플별 z-score 후 sigmoid를 적용하는 `instance`와 Otsu 임계(Eq.12~13)를 sigmoid 중심으로
쓰는 `otsu`를 선택할 수 있게 했다. 어느 쪽이 나은지는 ablation으로 판단한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NORMALIZE_MODES = ("none", "instance", "otsu")
DEFAULT_SCALES = (1, 3, 7)


def _pad_1d(x: torch.Tensor, pad: int, circular: bool) -> torch.Tensor:
    if pad == 0:
        return x
    if circular and pad < x.shape[-1]:
        return F.pad(x, (pad, pad), mode="circular")
    return F.pad(x, (pad, pad), mode="replicate")


def morphological_gradient(x: torch.Tensor, scale: int, circular: bool = True) -> torch.Tensor:
    """`G_s(x) = dilation_s(x) − erosion_s(x)`. 항상 0 이상이다."""
    if scale < 1:
        raise ValueError(f"scale은 1 이상이어야 합니다: {scale}")
    window = 2 * scale + 1
    padded = _pad_1d(x, scale, circular)
    dilation = F.max_pool1d(padded, kernel_size=window, stride=1)
    erosion = -F.max_pool1d(-padded, kernel_size=window, stride=1)
    return dilation - erosion


@torch.no_grad()
def otsu_threshold(values: torch.Tensor, bins: int = 64) -> torch.Tensor:
    """샘플별 Otsu 임계 `τ` — `ARDD-2025` Eq.(13)의 클래스 간 분산 최대화.

    Returns:
        `(B, 1, 1)` 임계값.
    """
    batch = values.shape[0]
    flat = values.reshape(batch, -1)
    low = flat.min(dim=1, keepdim=True).values
    high = flat.max(dim=1, keepdim=True).values
    span = (high - low).clamp_min(1e-12)
    normalized = (flat - low) / span

    indices = torch.clamp((normalized * bins).long(), 0, bins - 1)
    histogram = torch.zeros(batch, bins, device=values.device, dtype=values.dtype)
    histogram.scatter_add_(1, indices, torch.ones_like(normalized))
    probability = histogram / histogram.sum(dim=1, keepdim=True).clamp_min(1e-12)

    levels = (torch.arange(bins, device=values.device, dtype=values.dtype) + 0.5) / bins
    weight0 = torch.cumsum(probability, dim=1)
    weight1 = 1.0 - weight0
    mean_total = (probability * levels).sum(dim=1, keepdim=True)
    mean0 = torch.cumsum(probability * levels, dim=1) / weight0.clamp_min(1e-12)
    mean1 = (mean_total - torch.cumsum(probability * levels, dim=1)) / weight1.clamp_min(1e-12)

    between = weight0 * weight1 * (mean0 - mean1) ** 2
    best = between.argmax(dim=1, keepdim=True)
    threshold = torch.gather(levels.expand(batch, bins), 1, best)
    return (low + threshold * span).unsqueeze(-1)


class MorphologicalAttention1d(nn.Module):
    """다중 스케일 형태학 gradient 기반 attention 계수. 파라미터 없음."""

    def __init__(
        self,
        scales: tuple[int, ...] = DEFAULT_SCALES,
        circular: bool = True,
        normalize: str = "none",
    ) -> None:
        super().__init__()
        if not scales:
            raise ValueError("scales가 비어 있습니다.")
        if normalize not in NORMALIZE_MODES:
            raise ValueError(f"알 수 없는 normalize: {normalize} (가능: {NORMALIZE_MODES})")
        self.scales = tuple(int(scale) for scale in scales)
        self.circular = bool(circular)
        self.normalize = normalize

    def extra_repr(self) -> str:
        return f"scales={self.scales}, circular={self.circular}, normalize={self.normalize}"

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        """`a_base` — 스케일 간 max 융합 결과 `(B, 1, L)`."""
        signal = x.mean(dim=1, keepdim=True)
        gradients = [morphological_gradient(signal, scale, self.circular) for scale in self.scales]
        return torch.stack(gradients, dim=0).amax(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`(B, C, L)` → attention 계수 `(B, 1, L)`, 값 범위 `(0, 1)`."""
        if x.dim() != 3:
            raise ValueError(f"입력 shape이 (B, C, L)이 아닙니다: {tuple(x.shape)}")
        base = self.gradient(x)
        if self.normalize == "instance":
            mean = base.mean(dim=-1, keepdim=True)
            std = base.std(dim=-1, keepdim=True).clamp_min(1e-6)
            base = (base - mean) / std
        elif self.normalize == "otsu":
            base = base - otsu_threshold(base)
        return torch.sigmoid(base)

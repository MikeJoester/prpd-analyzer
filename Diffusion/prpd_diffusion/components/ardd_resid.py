"""적응형 잔차 연결 — `ARDD-2025` Eq.(5)~(8). 실험 id `A2` (`DIFFUSION.md` 18.4).

    x̂(t) = f(x, t) + a(x, t) · F(x, t)          Eq.(5)
    a(x, t) = α · (1 + H(x, t) / H_max)          Eq.(6)

`f`는 그대로 유지되는 원본 특징(U-Net ResidualBlock의 skip 경로), `F`는 convolution과
비선형 사상으로 얻은 특징 보충분(잔차 경로)이다. 즉 잔차 경로의 세기를 국소 엔트로피로
조절한다. 학습 파라미터는 없다.

**논문 본문과 수식의 방향이 서로 다르다.** 본문은 "노이즈가 많은 영역에서 잔차 특징 의존을
줄인다"고 설명하지만, Eq.(6)은 `H`가 클수록 `a`가 **커진다**(노이즈는 보통 엔트로피가 높다).
여기서는 **수식을 그대로 구현**하고, 본문 쪽 해석을 시험할 수 있도록 `invert` 옵션을 둔다.
어느 쪽이 우리 데이터에서 나은지는 ablation으로 판단하고 근거를 리포트에 남긴다.

Lipschitz 연속성(Eq.8): `a ∈ [α, 2α]`로 유계이므로 잔차 경로의 이득이 제한되고,
입력 섭동에 대한 출력 변동이 발산하지 않는다. `alpha=0.5`가 기본값인 이유는 이때
`a ∈ [0.5, 1.0]`이라 표준 잔차(=1.0)를 넘어 증폭하지 않아 초기 학습이 안정적이기 때문이다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveResidualGate(nn.Module):
    """엔트로피 맵으로 잔차 경로 세기를 조절한다. 파라미터 없음."""

    def __init__(self, alpha: float = 0.5, invert: bool = False) -> None:
        super().__init__()
        if alpha <= 0:
            raise ValueError(f"alpha는 0보다 커야 합니다: {alpha}")
        self.alpha = float(alpha)
        self.invert = bool(invert)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}, invert={self.invert}"

    def coefficient(self, entropy: torch.Tensor) -> torch.Tensor:
        """`a(x, t) = α · (1 + H/H_max)`. `invert=True`면 `α · (2 − H/H_max)`.

        두 경우 모두 `[α, 2α]` 범위이므로 Lipschitz 상한은 같다.
        """
        normalized = entropy.clamp(0.0, 1.0)
        if self.invert:
            normalized = 1.0 - normalized
        return self.alpha * (1.0 + normalized)

    def forward(
        self,
        residual: torch.Tensor,
        retained: torch.Tensor,
        entropy: torch.Tensor,
    ) -> torch.Tensor:
        """`retained + a · residual`.

        Args:
            residual: 잔차 경로 출력 `F(x, t)` — `(B, C, L)`
            retained: skip 경로 출력 `f(x, t)` — `(B, C, L)`
            entropy: `(B, 1, L')` 엔트로피 맵. 길이가 다르면 현재 해상도로 맞춘다.
        """
        if entropy.shape[-1] != residual.shape[-1]:
            entropy = F.adaptive_avg_pool1d(entropy, residual.shape[-1])
        return retained + self.coefficient(entropy) * residual

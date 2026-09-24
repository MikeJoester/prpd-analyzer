"""PRPD denoising diffusion package.

Analyzer가 생성한 `artifacts/ai_data_YYYYMMDD_HHMMSS/` 산출물만 입력으로 사용한다.
원본 `.dat`를 다시 해석하지 않으며, 날짜별/유형별 결과를 혼합하지 않는다.

torch에 의존하는 모듈(`models`, `training`, `diffusion.gaussian`, `diffusion.sampler`)은
필요한 시점에만 import한다. contract/data/evaluation은 numpy·pandas만으로 동작한다.
"""

from __future__ import annotations

REPO_VERSION = "denoise_v1"
"""이 패키지가 생성하는 산출물의 버전 태그. 학습/평가 결과에 함께 기록한다."""

__all__ = ["REPO_VERSION"]

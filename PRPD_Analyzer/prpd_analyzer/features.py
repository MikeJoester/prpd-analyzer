"""256차원 mean/max feature 생성.

    [mean_phase_001 ... mean_phase_128, max_phase_001 ... max_phase_128]

`claude.md` 3절의 정의이며 순서가 계약이다. `prpd_diffusion.data.profile.to_feature_vector`가
같은 벡터를 만들어야 분포 지표 비교가 성립한다(`tests/test_profile.py`에서 단언).

category 품질 점검(의심 점수)은 이 모듈이 아니라 `quality.py`에 있다.
"""

from __future__ import annotations

import numpy as np


def make_features(mean_profiles: np.ndarray, max_profiles: np.ndarray) -> np.ndarray:
    return np.concatenate((mean_profiles, max_profiles), axis=1).astype(np.float32)

"""ARDD 1D 트랙 — `ARDD-2025` 논문 기법을 위상 profile(1차원) 위에서 재현한다.

`DIFFUSION.md` 18.4의 A1~A5를 구성요소 on/off로 구현한 독립 트랙이다.
2D baseline(`prpd_diffusion.models.unet`, `prpd_diffusion.cli`)은 수정하지 않으며,
같은 비교 규약(18.1)으로 나란히 놓을 수 있게 지표·run 산출물 형식을 공유한다.

torch가 필요한 모듈: `dataset`(DataLoader), `cli`의 train/sample.
torch 없이 동작: `config`, `metrics`, `cli`의 prepare/evaluate.
"""

from __future__ import annotations

TRACK_VERSION = "ardd1d_v1"

"""논문 기법을 켜고 끄는 단위 구성요소 (`DIFFUSION.md` 18.5).

논문의 제안 모델을 통째로 이식하지 않고 구성요소로 쪼개 하나씩 켜고 끄며 비교한다.
그래야 개선이 어느 요소에서 왔는지 알 수 있다(18.4).

| 모듈 | 실험 id | 논문 근거 |
|---|---|---|
| `entropy` + `ardd_resid` | A2 | `ARDD-2025` Eq.(5)~(8) |
| `morph_attn` | A3 | `ARDD-2025` Eq.(9)~(15) |

모두 torch가 필요하다. 학습 파라미터는 없다(둘 다 parameter-free).
"""

from __future__ import annotations

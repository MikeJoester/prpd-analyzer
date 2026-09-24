"""전제 데이터가 없을 때 테스트를 건너뛰기 위한 공용 예외.

pytest가 설치되어 있으면 pytest의 skip 예외를 상속해 `python -m pytest tests`에서도
실패가 아니라 skip으로 집계된다. 없으면 평범한 예외로 동작하고
`tests/run_tests.py`가 이를 skip으로 처리한다.
"""

from __future__ import annotations

try:  # pytest가 있으면 skip 결과로 인식되게 한다.
    from _pytest.outcomes import Skipped as _Base
except ModuleNotFoundError:  # pragma: no cover - pytest 미설치 환경
    _Base = Exception  # type: ignore[misc, assignment]


class Skip(_Base):  # type: ignore[valid-type, misc]
    """이 테스트의 전제 조건(예: artifacts 폴더)이 없어 건너뜀."""

"""pytest 없이도 테스트를 돌리기 위한 최소 러너.

    python tests/run_tests.py            # 전체
    python tests/run_tests.py splits     # 이름에 'splits'가 포함된 모듈만

pytest를 설치했다면 `python -m pytest tests` 도 그대로 동작한다.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _skip import Skip  # noqa: E402 - sys.path 설정 후에 import 해야 한다


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(pattern: str | None = None) -> int:
    passed = failed = skipped = 0
    for path in sorted(Path(__file__).parent.glob("test_*.py")):
        if pattern and pattern not in path.stem:
            continue
        module = _load_module(path)
        for name in sorted(dir(module)):
            if not name.startswith("test_"):
                continue
            function = getattr(module, name)
            if not callable(function):
                continue
            try:
                function()
            except Skip as reason:
                skipped += 1
                print(f"SKIP {path.stem}.{name}: {reason}")
            except Exception:  # noqa: BLE001 - 테스트 러너이므로 모두 잡는다
                failed += 1
                print(f"FAIL {path.stem}.{name}")
                traceback.print_exc()
            else:
                passed += 1
                print(f"ok   {path.stem}.{name}")

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))

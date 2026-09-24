"""AI 학습용 raw dataset 실행 폴더를 CLI에서 생성한다.

Streamlit UI(`app.py`의 "AI 데이터셋" 탭)와 **같은 함수**를 호출한다. 새 로직은 없다.

    discover_files → load_samples → generate_ai_dataset

3,172개 x 460,800 B ≈ 1.5 GB를 한 번에 올리는 작업이라 UI 세션보다 스크립트가
재현·기록에 유리하다. UI 경로는 그대로 남겨 둔다.

사용 예:

    python scripts/build_ai_dataset.py
    python scripts/build_ai_dataset.py --groups "Lab PD" "Field PD" --raw-version raw_v2

`--root`는 `Data/by_type` 또는 `Data/by_date` 중 **하나만** 지정한다. 두 폴더는 같은
데이터의 다른 뷰일 뿐이므로, 한 번만 읽으면 모든 파일이 올바르게 맵핑된다. 기본값은 `Data/by_type`이다.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent  # PRPD_Analyzer/
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from PRPD_Analyzer.prpd_analyzer.dataset import generate_ai_dataset  # noqa: E402
from PRPD_Analyzer.prpd_analyzer.io import discover_files, load_samples  # noqa: E402
from PRPD_Analyzer.prpd_analyzer.metadata import GROUPS  # noqa: E402

SELECTABLE_GROUPS = GROUPS[:-1]  # "Unknown" 제외


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(ROOT / "Data" / "by_type"), help="기준 root (by_type 또는 by_date 중 하나)")
    parser.add_argument("--groups", nargs="+", default=list(SELECTABLE_GROUPS), choices=list(SELECTABLE_GROUPS))
    parser.add_argument("--output", default=str(REPO_ROOT / "artifacts"), help="ai_data_* 폴더를 만들 위치")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-version", default="raw_v2", help="raw_data_version 문자열")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root)
    if not root.is_dir():
        print(f"root 폴더를 찾을 수 없습니다: {root}")
        return 1

    files = discover_files(root)
    print(f"root         : {root}")
    print(f"검색된 파일  : {len(files):,}개")
    print(f"선택 그룹    : {', '.join(args.groups)}")
    if not files:
        print("읽을 .dat 파일이 없습니다.")
        return 1

    print("파일을 읽는 중... (수 분 걸릴 수 있습니다)")
    table = load_samples(str(root), tuple(str(path) for path in files))
    valid_count = int(table["valid"].sum())
    print(f"정상 파일    : {valid_count:,}개 / 오류 {len(table) - valid_count:,}개")

    in_scope = table[table["valid"] & table["group"].isin(args.groups)]
    print("그룹별 수량  :")
    print(in_scope.groupby("group").size().to_string())

    folder = generate_ai_dataset(
        table,
        root,
        args.groups,
        Path(args.output),
        args.seed,
        raw_data_version=args.raw_version,
    )
    print(f"\n생성 완료    : {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

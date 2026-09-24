"""날짜 단위 train/validation/test 분할.

CLAUDE.md 규칙: 같은 날짜의 데이터가 train과 test에 섞이지 않아야 한다.
파일 단위 무작위 분할은 같은 측정 세션이 양쪽에 들어가 성능을 과대평가하므로 쓰지 않는다.

한 날짜는 전 그룹에 걸쳐 **정확히 하나의 split**에 속한다(`scope="global"`).
날짜 수가 적은 그룹은 특정 split에서 비게 될 수 있으므로 `coverage_report`로 확인한다.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_NAMES = ("train", "val", "test")
DEFAULT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


@dataclass
class DateSplit:
    """날짜 → split 배정 결과."""

    assignment: dict[str, str]
    ratios: dict[str, float]
    seed: int
    scope: str = "global"

    def dates(self, split: str) -> tuple[str, ...]:
        return tuple(sorted(date for date, name in self.assignment.items() if name == split))

    def apply(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """metadata에 `split` 컬럼을 채워 반환한다(원본은 수정하지 않음)."""
        table = metadata.copy()
        table["split"] = table["date"].map(self.assignment).fillna("unassigned")
        return table

    def to_dict(self) -> dict:
        return {
            "scope": self.scope,
            "seed": self.seed,
            "ratios": self.ratios,
            "split_dates": {name: list(self.dates(name)) for name in SPLIT_NAMES},
        }


def make_date_splits(
    metadata: pd.DataFrame,
    ratios: dict[str, float] | None = None,
    seed: int = 42,
) -> DateSplit:
    """(group, label) 셀 커버리지를 고려한 결정적 greedy 날짜 분할.

    날짜를 샘플 수 내림차순으로 처리하면서, 아직 부족한 (group, label) 셀을 가장 많이
    채워 주는 split에 배정한다. 같은 metadata·seed면 항상 같은 결과가 나온다.
    """
    ratios = dict(ratios or DEFAULT_RATIOS)
    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        raise ValueError("ratio 합이 0보다 커야 합니다.")
    ratios = {name: value / total_ratio for name, value in ratios.items()}

    table = metadata.copy()
    table["cell"] = table["group"].astype(str) + " | " + table["label"].astype(str)
    cell_totals = table.groupby("cell").size().to_dict()
    per_date = table.groupby(["date", "cell"]).size().unstack(fill_value=0)

    rng = np.random.default_rng(seed)
    order = list(per_date.index)
    rng.shuffle(order)  # 동점 날짜의 순서를 seed로 고정
    order.sort(key=lambda date: int(per_date.loc[date].sum()), reverse=True)

    assigned = {name: {cell: 0 for cell in cell_totals} for name in ratios}
    totals = {name: 0 for name in ratios}
    assignment: dict[str, str] = {}

    for date in order:
        counts = per_date.loc[date]
        best_name, best_score, best_deficit = None, -1.0, -1.0
        for name in SPLIT_NAMES:
            if name not in ratios:
                continue
            score = 0.0
            for cell, cell_total in cell_totals.items():
                need = ratios[name] * cell_total - assigned[name][cell]
                if need > 0:
                    score += min(float(counts.get(cell, 0)), need)
            deficit = ratios[name] * len(table) - totals[name]
            if (score, deficit) > (best_score, best_deficit):
                best_name, best_score, best_deficit = name, score, deficit
        assignment[str(date)] = str(best_name)
        for cell in cell_totals:
            assigned[best_name][cell] += int(counts.get(cell, 0))
        totals[best_name] += int(counts.sum())

    return DateSplit(assignment=assignment, ratios=ratios, seed=seed)


def coverage_report(metadata: pd.DataFrame, split: DateSplit) -> pd.DataFrame:
    """split × group × label 샘플 수 표. 빈 셀이 있으면 학습/평가 설계를 재검토한다."""
    table = split.apply(metadata)
    report = (
        table.groupby(["split", "group", "label"])
        .agg(samples=("sample_id", "size"), dates=("date", "nunique"))
        .reset_index()
    )
    return report.sort_values(["group", "label", "split"]).reset_index(drop=True)


def check_no_date_leakage(split: DateSplit) -> list[str]:
    """한 날짜가 두 split에 배정되지 않았는지 확인한다(빈 리스트면 정상)."""
    seen: dict[str, str] = {}
    problems: list[str] = []
    for name in SPLIT_NAMES:
        for date in split.dates(name):
            if date in seen:
                problems.append(f"날짜 {date}가 {seen[date]}와 {name}에 중복 배정됨")
            seen[date] = name
    return problems


def save_split_config(path: Path, split: DateSplit, metadata: pd.DataFrame) -> Path:
    payload = split.to_dict()
    payload["counts"] = (
        split.apply(metadata).groupby("split").size().astype(int).to_dict()
    )
    payload["coverage"] = coverage_report(metadata, split).to_dict(orient="records")
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return Path(path)


def load_split_config(path: Path) -> DateSplit:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assignment = {
        date: name
        for name, dates in payload["split_dates"].items()
        for date in dates
    }
    return DateSplit(
        assignment=assignment,
        ratios=payload["ratios"],
        seed=int(payload["seed"]),
        scope=payload.get("scope", "global"),
    )

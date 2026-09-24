"""split별 clean / noise / 실제 noisy pool 구성.

    clean pool      : 학습 목표 x0. 기본 `Lab PD`
    noise pool      : 합성에 쓸 실측 노이즈. 기본 `Lab Noise`, `Field Noise`
    real noisy pool : 라벨 없는 평가 전용. 기본 `Field PD`
                      (같은 시점의 clean 정답이 없으므로 학습에 쓰지 않고
                       분포 기반 지표로만 평가한다)

`Synthetic` 그룹은 기본적으로 어디에도 포함하지 않는다. 필요하면 config에서 명시한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .splits import SPLIT_NAMES, DateSplit


@dataclass(frozen=True)
class PoolConfig:
    clean_groups: tuple[str, ...] = ("Lab PD",)
    noise_groups: tuple[str, ...] = ("Lab Noise", "Field Noise")
    real_noisy_groups: tuple[str, ...] = ("Field PD",)
    clean_labels: tuple[str, ...] | None = None  # None이면 clean 그룹의 모든 고장 종류


@dataclass
class Pools:
    """한 split의 pool 3종."""

    split: str
    clean: pd.DataFrame = field(default_factory=pd.DataFrame)
    noise: pd.DataFrame = field(default_factory=pd.DataFrame)
    real_noisy: pd.DataFrame = field(default_factory=pd.DataFrame)

    def counts(self) -> dict[str, int]:
        return {
            "clean": int(len(self.clean)),
            "noise": int(len(self.noise)),
            "real_noisy": int(len(self.real_noisy)),
        }


def build_pools(
    metadata: pd.DataFrame,
    date_split: DateSplit,
    config: PoolConfig | None = None,
) -> dict[str, Pools]:
    """split 이름 → `Pools`."""
    config = config or PoolConfig()
    table = date_split.apply(metadata)
    pools: dict[str, Pools] = {}
    for split in SPLIT_NAMES:
        rows = table[table["split"] == split]
        clean = rows[rows["group"].isin(config.clean_groups)]
        if config.clean_labels is not None:
            clean = clean[clean["label"].isin(config.clean_labels)]
        pools[split] = Pools(
            split=split,
            clean=clean.reset_index(drop=True),
            noise=rows[rows["group"].isin(config.noise_groups)].reset_index(drop=True),
            real_noisy=rows[rows["group"].isin(config.real_noisy_groups)].reset_index(drop=True),
        )
    return pools


def describe_pools(pools: dict[str, Pools]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"split": split, **pool.counts()} for split, pool in pools.items()]
    )


def check_pools(pools: dict[str, Pools], required: tuple[str, ...] = ("train", "val")) -> list[str]:
    """학습에 필요한 pool이 비어 있는지 확인한다(빈 리스트면 정상)."""
    problems: list[str] = []
    for split in required:
        pool = pools.get(split)
        if pool is None:
            problems.append(f"split '{split}' 없음")
            continue
        if pool.clean.empty:
            problems.append(f"split '{split}'의 clean pool이 비어 있음")
        if pool.noise.empty:
            problems.append(f"split '{split}'의 noise pool이 비어 있음")
    return problems

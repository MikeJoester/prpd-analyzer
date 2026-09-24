from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Diffusion.prpd_diffusion.data.pairs import PoolConfig, build_pools, check_pools
from Diffusion.prpd_diffusion.data.splits import (
    check_no_date_leakage,
    coverage_report,
    load_split_config,
    make_date_splits,
    save_split_config,
)


def _metadata(dates_per_group: int = 12) -> pd.DataFrame:
    rows = []
    groups = [("Lab PD", "Corona"), ("Lab PD", "Void"), ("Field PD", "Void"), ("Lab Noise", "Noise"), ("Field Noise", "Noise")]
    for group, label in groups:
        for day in range(dates_per_group):
            date = f"2024{(day % 12) + 1:02d}{day + 1:02d}"
            for index in range(4):
                rows.append(
                    {
                        "sample_id": f"{group}-{label}-{date}-{index}",
                        "group": group,
                        "label": label,
                        "date": date,
                    }
                )
    return pd.DataFrame(rows)


def test_no_date_appears_in_two_splits():
    metadata = _metadata()
    split = make_date_splits(metadata, seed=42)
    assert check_no_date_leakage(split) == []

    applied = split.apply(metadata)
    per_date_splits = applied.groupby("date")["split"].nunique()
    assert int(per_date_splits.max()) == 1


def test_every_sample_is_assigned():
    metadata = _metadata()
    applied = make_date_splits(metadata, seed=1).apply(metadata)
    assert not (applied["split"] == "unassigned").any()
    assert set(applied["split"].unique()) <= {"train", "val", "test"}


def test_split_is_deterministic_for_same_seed():
    metadata = _metadata()
    left = make_date_splits(metadata, seed=7).assignment
    right = make_date_splits(metadata, seed=7).assignment
    assert left == right


def test_ratios_are_approximately_respected():
    metadata = _metadata(dates_per_group=20)
    applied = make_date_splits(metadata, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=3).apply(metadata)
    shares = applied.groupby("split").size() / len(applied)
    assert 0.55 <= shares.get("train", 0) <= 0.85
    assert shares.get("test", 0) > 0.02


def test_coverage_report_has_all_cells():
    metadata = _metadata()
    report = coverage_report(metadata, make_date_splits(metadata, seed=5))
    assert set(report.columns) == {"split", "group", "label", "samples", "dates"}
    assert report["samples"].sum() == len(metadata)


def test_split_config_roundtrip():
    metadata = _metadata()
    split = make_date_splits(metadata, seed=11)
    with tempfile.TemporaryDirectory() as folder:
        path = save_split_config(Path(folder) / "split_config.json", split, metadata)
        restored = load_split_config(path)
    assert restored.assignment == split.assignment
    assert restored.seed == split.seed


def test_pools_are_built_per_split():
    metadata = _metadata()
    split = make_date_splits(metadata, seed=13)
    pools = build_pools(metadata, split, PoolConfig())
    assert check_pools(pools) == []
    for pool in pools.values():
        assert set(pool.clean["group"].unique()) <= {"Lab PD"}
        assert set(pool.noise["group"].unique()) <= {"Lab Noise", "Field Noise"}
        assert set(pool.real_noisy["group"].unique()) <= {"Field PD"}

    # 노이즈 pool의 날짜도 split 간에 겹치면 안 된다(노이즈 누수 방지).
    train_noise_dates = set(pools["train"].noise["date"])
    test_noise_dates = set(pools["test"].noise["date"])
    assert not (train_noise_dates & test_noise_dates)

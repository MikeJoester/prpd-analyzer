"""실제 `artifacts/ai_data_*` 폴더를 대상으로 하는 계약 검증.

Analyzer 산출물이 없으면 건너뛴다. metadata 행과 raw tensor index의 대응이 어긋나면
라벨과 데이터가 뒤섞이므로, 원본 `.dat` 한 개와 직접 대조해 확인한다.
(원본 재해석은 이 검증 목적에만 사용하고 학습 경로에서는 하지 않는다.)
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _skip import Skip
from Diffusion.prpd_diffusion.contract import AiDataset, find_run_folders

ROOT = Path(__file__).resolve().parent.parent


def _dataset() -> AiDataset:
    folders = find_run_folders(ROOT.parent / "artifacts")
    if not folders:
        raise Skip("artifacts/ai_data_* 폴더가 없습니다.")
    return AiDataset.load(folders[-1])


def test_dataset_loads_and_group_counts_match():
    dataset = _dataset()
    assert len(dataset.metadata) > 0
    assert "tensor_index" in dataset.metadata.columns
    # AiDataset.load가 그룹별 개수를 이미 검증한다. 여기서는 인덱스 범위를 본다.
    for group, table in dataset.metadata.groupby("group"):
        assert table["tensor_index"].min() == 0
        assert table["tensor_index"].max() == len(table) - 1


def test_sample_ids_are_unique():
    dataset = _dataset()
    assert dataset.metadata["sample_id"].is_unique


def test_matrix_shape_and_dtype():
    dataset = _dataset()
    sample_id = str(dataset.metadata["sample_id"].iloc[0])
    matrix = dataset.matrix(sample_id)
    assert matrix.shape == (128, 3600)
    assert matrix.dtype == np.uint8


def test_matrix_matches_original_dat_file():
    """metadata 행 ↔ raw tensor index 대응이 실제로 맞는지 원본과 대조한다."""
    from PRPD_Analyzer.prpd_analyzer.io import read_dat

    dataset = _dataset()
    checked = 0
    # 그룹마다 첫/마지막 행을 확인하면 index 오프셋 오류를 잡을 수 있다.
    for _, table in dataset.metadata.groupby("group"):
        for position in (0, len(table) - 1):
            row = table.iloc[position]
            source = Path(str(row["path"]))
            if not source.is_file():
                continue
            assert np.array_equal(dataset.matrix(str(row["sample_id"])), read_dat(source))
            checked += 1
    if checked == 0:
        raise Skip("원본 .dat 파일 경로를 찾을 수 없습니다.")


def test_select_filters_by_group_and_label():
    dataset = _dataset()
    groups = dataset.groups[:1]
    selected = dataset.select(groups=tuple(groups))
    assert set(selected["group"].unique()) == set(groups)
    assert len(selected) <= len(dataset.metadata)


def test_memmap_cache_is_separated_per_ai_data_run():
    """캐시 경로에 실행 이름이 없으면 dataset을 바꿔도 이전 실행의 데이터를 읽게 된다."""
    folders = find_run_folders(ROOT.parent / "artifacts")
    if len(folders) < 2:
        raise Skip("ai_data 실행 폴더가 2개 미만이라 분리 여부를 확인할 수 없습니다.")
    cache_root = Path("artifacts") / "repr_cache"
    first = AiDataset.load(folders[0]).cache_folder(cache_root)
    second = AiDataset.load(folders[-1]).cache_folder(cache_root)
    assert first != second
    assert first.name == folders[0].name
    assert second.name == folders[-1].name

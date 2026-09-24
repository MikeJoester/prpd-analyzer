"""Analyzer AI dataset(`ai_data_YYYYMMDD_HHMMSS/`) 읽기 전용 로더.

`src/prpd_analyzer/dataset.py`가 저장한 구조를 그대로 소비한다.

    ai_data_YYYYMMDD_HHMMSS/
    ├── metadata.parquet          # valid 행 순서 그대로 저장됨
    ├── raw_tensors_v1/{group}.npz   # key "raw", shape (그룹 파일 수, 128, 3600), uint8
    ├── run_config.json
    └── manifest.json

metadata 행과 npz 배열의 대응 규칙:
    dataset.py는 `valid.groupby("group")` 순서로 각 그룹 배열을 stack 하므로,
    한 그룹 안에서의 배열 index는 metadata 저장 순서 기준 그룹 내 누적 순번과 같다.
    → `metadata.groupby("group").cumcount()`
이 규칙이 깨지면 라벨과 raw tensor가 어긋나므로 로드 시점에 그룹별 개수를 검증한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PHASE_BINS = 128
TIME_SAMPLES = 3600
RAW_DTYPE = np.uint8

REQUIRED_COLUMNS = (
    "sample_id",
    "file",
    "path",
    "group",
    "label",
    "date",
    "raw_tensor_path",
    "data_quality_status",
)


class ContractError(RuntimeError):
    """AI dataset 폴더가 기대한 계약을 만족하지 않을 때 발생한다."""


def find_run_folders(artifacts_root: Path) -> list[Path]:
    """`artifacts/` 아래의 ai_data 실행 폴더를 이름(=생성 시각) 순으로 반환한다."""
    if not artifacts_root.is_dir():
        return []
    return sorted(path for path in artifacts_root.glob("ai_data_*") if path.is_dir())


def latest_run_folder(artifacts_root: Path) -> Path:
    folders = find_run_folders(artifacts_root)
    if not folders:
        raise ContractError(f"ai_data_* 폴더를 찾을 수 없습니다: {artifacts_root}")
    return folders[-1]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class AiDataset:
    """하나의 ai_data 실행 폴더를 나타낸다. 원본은 절대 수정하지 않는다."""

    root: Path
    metadata: pd.DataFrame
    run_config: dict
    _arrays: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _memmap_root: Path | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ 로드

    @classmethod
    def load(cls, root: Path) -> "AiDataset":
        root = Path(root)
        metadata_path = root / "metadata.parquet"
        if not metadata_path.is_file():
            raise ContractError(f"metadata.parquet 없음: {root}")

        metadata = pd.read_parquet(metadata_path)
        missing = [name for name in REQUIRED_COLUMNS if name not in metadata.columns]
        if missing:
            raise ContractError(f"metadata 필수 컬럼 누락: {missing}")

        metadata = metadata.reset_index(drop=True)
        metadata["tensor_index"] = metadata.groupby("group").cumcount()

        duplicated = metadata["sample_id"].duplicated()
        if duplicated.any():
            raise ContractError(
                f"sample_id 중복 {int(duplicated.sum())}건 — 날짜별/유형별 혼합 여부를 확인하세요."
            )

        config_path = root / "run_config.json"
        run_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}

        dataset = cls(root=root, metadata=metadata, run_config=run_config)
        dataset._verify_group_counts()
        return dataset

    def _verify_group_counts(self) -> None:
        """metadata의 그룹별 행 수와 npz 배열 길이가 일치하는지 확인한다."""
        for group, count in self.metadata.groupby("group").size().items():
            path = self._tensor_path(str(group))
            if not path.is_file():
                raise ContractError(f"raw tensor 없음: {path}")
            with np.load(path) as archive:
                stored = archive["raw"].shape[0]
            if stored != int(count):
                raise ContractError(
                    f"그룹 '{group}' 개수 불일치: metadata {count}행 vs raw tensor {stored}개"
                )

    def _tensor_path(self, group: str) -> Path:
        rows = self.metadata[self.metadata["group"] == group]
        if rows.empty:
            raise ContractError(f"metadata에 없는 그룹: {group}")
        return self.root / str(rows["raw_tensor_path"].iloc[0])

    # -------------------------------------------------------------- 조회 API

    @property
    def groups(self) -> list[str]:
        return sorted(self.metadata["group"].unique().tolist())

    @property
    def dates(self) -> list[str]:
        return sorted(self.metadata["date"].unique().tolist())

    def select(
        self,
        groups: tuple[str, ...] | None = None,
        labels: tuple[str, ...] | None = None,
        dates: tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        table = self.metadata
        if groups is not None:
            table = table[table["group"].isin(groups)]
        if labels is not None:
            table = table[table["label"].isin(labels)]
        if dates is not None:
            table = table[table["date"].isin(dates)]
        return table.reset_index(drop=True)

    def group_array(self, group: str) -> np.ndarray:
        """그룹 전체 raw tensor. memmap 캐시가 있으면 mmap으로, 없으면 npz 전체를 올린다."""
        if group in self._arrays:
            return self._arrays[group]

        if self._memmap_root is not None:
            memmap_path = self._memmap_root / f"{group}.npy"
            if memmap_path.is_file():
                array = np.load(memmap_path, mmap_mode="r")
                self._arrays[group] = array
                return array

        with np.load(self._tensor_path(group)) as archive:
            array = archive["raw"]
        self._arrays[group] = array
        return array

    def matrix(self, sample_id: str) -> np.ndarray:
        """sample_id 하나의 `(128, 3600)` uint8 행렬을 반환한다(복사본 아님, 읽기 전용 취급)."""
        rows = self.metadata[self.metadata["sample_id"] == sample_id]
        if rows.empty:
            raise KeyError(f"알 수 없는 sample_id: {sample_id}")
        row = rows.iloc[0]
        matrix = self.group_array(str(row["group"]))[int(row["tensor_index"])]
        if matrix.shape != (PHASE_BINS, TIME_SAMPLES):
            raise ContractError(f"raw shape 오류 {matrix.shape}: {sample_id}")
        return matrix

    def matrices(self, table: pd.DataFrame) -> np.ndarray:
        """metadata 부분집합에 대응하는 `(N, 128, 3600)` 배열을 만든다(메모리 주의)."""
        return np.stack([self.matrix(str(sample_id)) for sample_id in table["sample_id"]])

    # ------------------------------------------------------------ 캐시/검증

    def cache_folder(self, cache_root: Path) -> Path:
        """이 ai_data 실행 전용 캐시 폴더 경로 `{cache_root}/{실행 폴더 이름}`.

        실행 이름을 넣지 않으면 서로 다른 `ai_data_*` 실행이 같은 `{group}.npy`를 공유해,
        dataset을 바꿔도 이전 실행의 데이터를 그대로 읽게 된다(`DIFFUSION.md` 15·17절).
        """
        return Path(cache_root) / self.root.name

    def materialize_memmap(self, cache_root: Path, groups: tuple[str, ...] | None = None) -> Path:
        """압축 npz를 그룹별 `.npy`로 풀어 학습 중 mmap 읽기가 가능하게 한다.

        npz(deflate)는 매 접근마다 전체 압축 해제가 필요해 학습 루프에 부적합하다.
        원본 npz는 그대로 두고 캐시만 생성한다.

        캐시는 **ai_data 실행별 하위 폴더**에 만든다(`cache_folder` 참조).
        """
        cache_root = self.cache_folder(cache_root)
        cache_root.mkdir(parents=True, exist_ok=True)
        for group in groups or self.groups:
            target = cache_root / f"{group}.npy"
            if not target.is_file():
                with np.load(self._tensor_path(group)) as archive:
                    np.save(target, archive["raw"])
            self._arrays.pop(group, None)
        self._memmap_root = cache_root
        return cache_root

    def release(self) -> None:
        """열려 있는 배열·memmap 참조를 해제한다.

        Windows에서는 memmap이 열려 있는 동안 캐시 파일을 지울 수 없으므로,
        캐시를 재생성하거나 삭제하기 전에 호출한다.
        """
        self._arrays.clear()

    def verify_manifest(self) -> list[str]:
        """manifest.json의 checksum을 재계산해 불일치 목록을 반환한다(빈 리스트면 정상)."""
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            return ["manifest.json 없음"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        problems: list[str] = []
        for entry in manifest.get("files", []):
            expected = entry.get("sha256")
            if not expected:
                continue
            path = self.root / str(entry["path"])
            if not path.is_file():
                problems.append(f"파일 없음: {entry['path']}")
            elif _file_sha256(path) != expected:
                problems.append(f"checksum 불일치: {entry['path']}")
        return problems

    def summary(self) -> pd.DataFrame:
        return (
            self.metadata.groupby(["group", "label"])
            .agg(count=("sample_id", "size"), dates=("date", "nunique"))
            .reset_index()
        )

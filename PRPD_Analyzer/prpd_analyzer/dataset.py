from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .io import PHASE_BINS, PAYLOAD_LAYOUT, TIME_SAMPLES


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_ai_dataset(
    table: pd.DataFrame,
    source_root: Path,
    selected_groups: Iterable[str],
    output_root: Path,
    seed: int,
    raw_data_version: str = "raw_v1",
) -> Path:
    selected_groups = tuple(selected_groups)
    valid = table[(table["valid"]) & table["group"].isin(selected_groups)].copy().reset_index(drop=True)
    if valid.empty:
        raise ValueError("선택한 그룹에 유효한 raw data가 없습니다.")

    created_at = datetime.now().astimezone()
    run_folder = output_root / f"ai_data_{created_at:%Y%m%d_%H%M%S}"
    run_folder.mkdir(parents=True, exist_ok=False)
    tensor_folder = run_folder / "raw_tensors_v1"
    tensor_folder.mkdir()

    valid["sample_id"] = valid["path"].map(lambda value: hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16])
    valid["split"] = "unspecified"
    valid["raw_shape"] = "(128, 3600)"
    valid["raw_dtype"] = "uint8"
    valid["raw_data_version"] = raw_data_version
    valid["normalization"] = "none"
    valid["data_quality_status"] = "valid"
    valid["quality_reasons"] = ""
    valid["raw_tensor_path"] = valid["group"].map(lambda group: f"raw_tensors_v1/{group}.npz")

    for group, group_table in valid.groupby("group"):
        matrices = np.stack(group_table["matrix"].to_numpy()).astype(np.uint8, copy=False)
        np.savez_compressed(tensor_folder / f"{group}.npz", raw=matrices)

    metadata_columns = [
        "sample_id", "file", "path", "group", "label", "date", "raw_shape",
        "raw_dtype", "raw_tensor_path", "normalization", "data_quality_status",
        "quality_reasons", "split", "raw_data_version",
    ]
    metadata = valid[metadata_columns]
    metadata.to_parquet(run_folder / "metadata.parquet", index=False)
    metadata.to_parquet(run_folder / "ai_raw_dataset_v1.parquet", index=False)
    metadata.to_csv(run_folder / "ai_raw_dataset_v1.csv", index=False, encoding="utf-8-sig")

    run_config = {
        "created_at": created_at.isoformat(),
        "source_root_type": source_root.name,
        "source_root_path": str(source_root.resolve()),
        "selected_groups": list(selected_groups),
        "total_selected_files": int(len(table[table["group"].isin(selected_groups)])),
        "valid_file_count": int(len(valid)),
        "excluded_file_count": int(len(table[table["group"].isin(selected_groups)]) - len(valid)),
        "raw_shape": [PHASE_BINS, TIME_SAMPLES],
        "payload_layout": PAYLOAD_LAYOUT,
        "raw_data_version": raw_data_version,
        "normalization": "none",
        "split": "unspecified; decided by AI model stage",
        "seed": seed,
    }
    (run_folder / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    description = f"""PRPD AI raw dataset description

생성 날짜 및 시간: {created_at.isoformat()}
생성 데이터 종류: 원본 128x3600 raw tensor, metadata, raw data 목록
원본 데이터 경로: {source_root.resolve()}
원본 root 설명: {source_root.name} 기준을 사용했으며 날짜별/유형별 중 하나만 분석함
선택 그룹: {', '.join(selected_groups)}
그룹별 파일 수:\n{valid.groupby('group').size().to_string()}
label별 파일 수:\n{valid.groupby('label').size().to_string()}
전체 선택 파일 수: {run_config['total_selected_files']}
정상 파일 수: {len(valid)}
제외 파일 수: {run_config['excluded_file_count']}
raw tensor shape: (128, 3600)  # (phase, time)
raw tensor dtype: uint8
payload layout: {PAYLOAD_LAYOUT}
normalization: none (원본 raw 보존)
split 방법: 미지정 (AI model 단계에서 결정)
random seed: {seed}
raw data version: {raw_data_version}
"""
    (run_folder / "dataset_description.txt").write_text(description, encoding="utf-8")

    manifest: dict[str, object] = {"created_at": created_at.isoformat(), "files": []}
    for path in sorted(run_folder.rglob("*")):
        if path.is_file():
            manifest["files"].append({
                "path": str(path.relative_to(run_folder)),
                "size": path.stat().st_size,
                "sha256": _file_sha256(path),
            })
    manifest["files"].append({
        "path": "manifest.json",
        "size": None,
        "sha256": None,
        "note": "manifest self-entry; checksum omitted to avoid recursive hashing",
    })
    (run_folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_folder

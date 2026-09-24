"""`.dat` (PRPS) 파일 읽기와 파일 목록 수집.

payload 레이아웃 — 장비는 **cycle 단위 레코드**로 저장한다:

    [227 또는 167 byte header]
    [cycle 0: phase 0..127][cycle 1: phase 0..127] ... [cycle 3599: phase 0..127]

즉 연속 축은 위상(128)이고 레코드 축이 시간(3600 cycle = 60 Hz x 60 s)이다.
따라서 payload를 `(3600, 128)`로 reshape한 뒤 transpose해야 위상 행이 만들어진다.

`reshape(128, 3600)`으로 읽으면 `3600 % 128 = 16`이라 각 행이 위상 16 bin씩 밀리며
섞여, 위상별 평균/최대가 같은 위상의 값이 아니게 된다. 실측 확인(2026-08-25):
payload 자기상관이 lag 128에서 0.978, lag 3600에서 0.192 — 위상이 연속 축이다.

모듈 밖으로 나가는 행렬은 항상 `(phase=128, time=3600)` 규약을 따른다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metadata import infer_metadata

HEADER_BYTES = 227
ALTERNATE_HEADER_BYTES = 167
PHASE_BINS = 128
TIME_SAMPLES = 3600
PAYLOAD_SIZE = PHASE_BINS * TIME_SAMPLES

# 산출물에 기록해, 축이 뒤집힌 옛 실행(raw_v1)과 파일만 보고 구별할 수 있게 한다.
PAYLOAD_LAYOUT = "3600 cycles x 128 phase bins (phase contiguous), transposed to (128, 3600)"


def discover_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.dat"))


def read_dat(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < ALTERNATE_HEADER_BYTES:
        raise ValueError(f"헤더보다 짧은 파일: {path.name}")
    if len(raw) == HEADER_BYTES + PAYLOAD_SIZE:
        payload = raw[HEADER_BYTES:]
    elif len(raw) == ALTERNATE_HEADER_BYTES + PAYLOAD_SIZE:
        payload = raw[ALTERNATE_HEADER_BYTES:]
    else:
        payload_size = len(raw) - HEADER_BYTES
        raise ValueError(f"payload 크기 {payload_size:,} (예상 {PAYLOAD_SIZE:,}): {path.name}")
    # payload는 (cycle, phase) 순서다. 모듈 docstring 참조.
    # ascontiguousarray: .T는 non-contiguous read-only view라 stack/savez/memmap에 부적합하다.
    records = np.frombuffer(payload, dtype=np.uint8).reshape(TIME_SAMPLES, PHASE_BINS)
    return np.ascontiguousarray(records.T)


def load_samples(root_text: str, selected_files: tuple[str, ...]) -> pd.DataFrame:
    root = Path(root_text)
    records: list[dict[str, object]] = []
    for file_text in selected_files:
        path = Path(file_text)
        try:
            matrix = read_dat(path)
        except ValueError as error:
            records.append({"path": str(path), "valid": False, "error": str(error)})
            continue
        label, group, date = infer_metadata(path, root)
        records.append(
            {
                "path": str(path),
                "file": path.name,
                "label": label,
                "group": group,
                "date": date,
                "matrix": matrix,
                "mean_profile": matrix.mean(axis=1),
                "max_profile": matrix.max(axis=1),
                "valid": True,
                "error": "",
            }
        )
    return pd.DataFrame(records)

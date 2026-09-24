"""`.dat` parser 검증 — 특히 **위상축과 시간축이 뒤바뀌지 않는지**.

2026-08-25 이전 코드는 payload를 `(128, 3600)`으로 reshape했다. 실제 저장 순서는
`(3600 cycle, 128 phase)`이므로 각 행이 위상 16 bin씩 밀리며 섞였고, 위상별
평균/최대 feature가 같은 위상의 값이 아니었다. 아래 방향 테스트가 그 회귀를 막는다.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _skip import Skip
from PRPD_Analyzer.prpd_analyzer.io import (
    ALTERNATE_HEADER_BYTES,
    HEADER_BYTES,
    PAYLOAD_SIZE,
    PHASE_BINS,
    TIME_SAMPLES,
    read_dat,
)

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ 합성 파일 helper


def _phase_ramp_payload() -> bytes:
    """cycle마다 동일한 위상 ramp를 담은 payload.

    값 = `phase % 251` (251은 소수라 128과 공약수가 없어, 축이 뒤집히면 패턴이 깨진다).
    """
    ramp = (np.arange(PHASE_BINS, dtype=np.int64) % 251).astype(np.uint8)
    records = np.tile(ramp, (TIME_SAMPLES, 1))  # (3600 cycle, 128 phase)
    return records.tobytes()


def _write_dat(folder: Path, name: str, payload: bytes, header_bytes: int) -> Path:
    path = folder / name
    path.write_bytes(b"\x00" * header_bytes + payload)
    return path


# ------------------------------------------------------------------ 방향 테스트


def test_reshape_keeps_phase_on_axis0():
    """payload의 연속 128개 = 한 cycle의 위상 스캔이어야 한다."""
    payload = _phase_ramp_payload()
    with tempfile.TemporaryDirectory() as folder:
        matrix = read_dat(_write_dat(Path(folder), "ramp.dat", payload, HEADER_BYTES))

    assert matrix.shape == (PHASE_BINS, TIME_SAMPLES)
    # 모든 cycle이 같은 ramp이므로 시간축으로 상수여야 한다.
    assert matrix.std(axis=1).max() == 0.0
    expected = (np.arange(PHASE_BINS) % 251).astype(np.uint8)
    assert np.array_equal(matrix[:, 0], expected)
    # 위상별 평균/최대 = ramp 그 자체. 축이 뒤집히면 둘 다 깨진다.
    assert np.array_equal(matrix.mean(axis=1), expected.astype(np.float64))
    assert np.array_equal(matrix.max(axis=1), expected)


def test_time_axis_carries_cycle_variation():
    """cycle마다 값이 다르면 그 변화는 시간축(axis=1)에 나타나야 한다."""
    cycles = np.arange(TIME_SAMPLES, dtype=np.int64) % 256
    records = np.repeat(cycles[:, None], PHASE_BINS, axis=1).astype(np.uint8)
    with tempfile.TemporaryDirectory() as folder:
        matrix = read_dat(_write_dat(Path(folder), "cycles.dat", records.tobytes(), HEADER_BYTES))

    assert matrix.std(axis=0).max() == 0.0  # 같은 cycle 안에서는 위상별로 동일
    assert np.array_equal(matrix[0], cycles.astype(np.uint8))


def test_both_header_lengths_give_same_matrix():
    payload = _phase_ramp_payload()
    with tempfile.TemporaryDirectory() as folder:
        base = Path(folder)
        standard = read_dat(_write_dat(base, "h227.dat", payload, HEADER_BYTES))
        alternate = read_dat(_write_dat(base, "h167.dat", payload, ALTERNATE_HEADER_BYTES))
    assert np.array_equal(standard, alternate)


def test_returned_matrix_is_contiguous_and_writable():
    """np.stack / savez_compressed / memmap 저장 경로가 이 두 성질에 의존한다."""
    payload = _phase_ramp_payload()
    with tempfile.TemporaryDirectory() as folder:
        matrix = read_dat(_write_dat(Path(folder), "flags.dat", payload, HEADER_BYTES))
    assert matrix.flags["C_CONTIGUOUS"]
    assert matrix.flags["WRITEABLE"]
    assert matrix.dtype == np.uint8


# ------------------------------------------------------------------ 오류 처리


def test_wrong_payload_size_raises():
    with tempfile.TemporaryDirectory() as folder:
        path = _write_dat(Path(folder), "short.dat", b"\x01" * (PAYLOAD_SIZE - 1), HEADER_BYTES)
        try:
            read_dat(path)
        except ValueError:
            pass
        else:
            raise AssertionError("payload 크기가 틀린 파일이 통과했습니다.")


def test_file_shorter_than_header_raises():
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "tiny.dat"
        path.write_bytes(b"\x00" * 10)
        try:
            read_dat(path)
        except ValueError:
            pass
        else:
            raise AssertionError("헤더보다 짧은 파일이 통과했습니다.")


# ------------------------------------------------------------------ 실데이터 검증


def _autocorrelation(values: np.ndarray, lag: int) -> float:
    left, right = values[:-lag], values[lag:]
    return float((left * right).sum() / (np.linalg.norm(left) * np.linalg.norm(right)))


def test_real_payload_is_periodic_at_128_not_3600():
    """원본 파일에서 위상이 연속 축임을 직접 확인한다.

    Noise 파일은 위상 구조가 아예 없어 두 자기상관이 모두 0 근처다(관측: 0.007 vs 0.010).
    그런 파일의 대소 비교는 의미가 없으므로, **주기 구조가 실제로 있는 파일**만 판정한다.
    `Data/`가 없으면 건너뛴다.
    """
    share = ROOT / "Data"
    if not share.is_dir():
        raise Skip("Data/ 폴더가 없습니다.")
    files = sorted(share.rglob("*.dat"))
    if not files:
        raise Skip("Data/ 아래에 .dat 파일이 없습니다.")

    structured = 0
    for path in files[::97]:
        raw = path.read_bytes()
        header = len(raw) - PAYLOAD_SIZE
        if header not in (HEADER_BYTES, ALTERNATE_HEADER_BYTES):
            continue
        values = np.frombuffer(raw[header:], dtype=np.uint8).astype(np.float64)
        values = values - values.mean()
        at_phase = _autocorrelation(values, PHASE_BINS)
        at_time = _autocorrelation(values, TIME_SAMPLES)
        # 어떤 파일도 3600 주기를 보이면 안 된다 — 그건 축이 반대라는 뜻이다.
        assert at_time < 0.2, f"{path}: lag 3600 자기상관 {at_time:.3f}"
        if max(at_phase, at_time) < 0.2:
            continue  # 구조 없음(대부분 Noise). 판정 대상 아님
        assert at_phase > at_time, f"{path}: lag128 {at_phase:.3f} <= lag3600 {at_time:.3f}"
        structured += 1
        if structured >= 5:
            break
    if structured == 0:
        raise Skip("주기 구조가 뚜렷한 파일을 표본에서 찾지 못했습니다.")


def test_real_file_matrix_shape_and_profiles():
    share = ROOT / "Data"
    if not share.is_dir():
        raise Skip("Data/ 폴더가 없습니다.")
    files = sorted(share.rglob("*.dat"))
    if not files:
        raise Skip("Data/ 아래에 .dat 파일이 없습니다.")

    matrix = read_dat(files[0])
    assert matrix.shape == (PHASE_BINS, TIME_SAMPLES)
    assert matrix.dtype == np.uint8
    assert matrix.mean(axis=1).shape == (PHASE_BINS,)
    assert matrix.max(axis=1).shape == (PHASE_BINS,)

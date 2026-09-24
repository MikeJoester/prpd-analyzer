"""실행 결과 폴더 생성과 재현 정보 기록.

Analyzer의 `ai_data_YYYYMMDD_HHMMSS` 규칙을 그대로 따른다.

    Results/diffusion_runs/denoise_YYYYMMDD_HHMMSS/
    ├── run_config.json          # 실행 설정 전체
    ├── split_config.json        # 날짜 단위 분할 결과
    ├── dataset_description.txt  # 사람이 읽는 실행 설명
    ├── train_log.jsonl
    ├── checkpoints/
    ├── samples/
    ├── metrics.json
    └── manifest.json            # 산출물 경로·크기·checksum
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys

DEFAULT_RUNS_ROOT = Path("Results") / "diffusion_runs"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_run_folder(runs_root: Path = DEFAULT_RUNS_ROOT, prefix: str = "denoise") -> Path:
    """생성 시각으로 이름 붙인 새 폴더. 기존 실행을 덮어쓰지 않는다."""
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now().astimezone()
    folder = runs_root / f"{prefix}_{created_at:%Y%m%d_%H%M%S}"
    folder.mkdir(exist_ok=False)
    (folder / "checkpoints").mkdir()
    (folder / "samples").mkdir()
    return folder


def environment_info() -> dict:
    """재현에 필요한 실행 환경. torch가 없으면 해당 항목만 비운다."""
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for name in ("numpy", "pandas", "torch"):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "unknown")
        except ModuleNotFoundError:
            info[name] = "not installed"
    if info.get("torch") != "not installed":
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    return info


def write_json(path: Path, payload: dict) -> Path:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return Path(path)


def write_description(
    run_folder: Path,
    config,                 # DenoiseConfig
    dataset_root: Path,
    pool_counts: dict,
    split_summary: dict,
    extra_lines: list[str] | None = None,
) -> Path:
    """`dataset_description.txt` — 이 실행이 무엇을 학습했는지 사람이 읽는 설명."""
    created_at = datetime.now().astimezone().isoformat()
    lines = [
        "PRPD denoising diffusion run description",
        "",
        f"생성 날짜 및 시간: {created_at}",
        f"실행 폴더: {run_folder.name}",
        f"config hash: {config.config_hash()}",
        f"random seed: {config.seed}",
        "",
        "[입력 데이터]",
        f"AI dataset 경로: {Path(dataset_root).resolve()}",
        "원본 .dat를 다시 해석하지 않고 Analyzer가 검증한 raw tensor만 사용함",
        "날짜별/유형별 중 하나의 root만 사용하며 두 결과를 합치지 않음",
        f"clean pool 그룹: {', '.join(config.data.clean_groups)}",
        f"noise pool 그룹: {', '.join(config.data.noise_groups)}",
        f"실제 noisy 평가 그룹: {', '.join(config.data.real_noisy_groups)}",
        f"split별 pool 수: {json.dumps(pool_counts, ensure_ascii=False)}",
        "",
        "[학습 데이터 구성]",
        "paired 데이터는 실측 노이즈를 clean PD에 합성해 만든다(같은 시점의 실제 쌍은 없음).",
        f"mixing mode: {config.noise.mode}",
        f"noise gain 범위: {config.noise.gain_min} ~ {config.noise.gain_max}",
        f"crop: 128 x {config.data.crop_width} (시간축 무작위 window)",
        "",
        "[분할]",
        "날짜 단위 분할 — 같은 날짜가 train과 test에 동시에 들어가지 않음",
        f"split 요약: {json.dumps(split_summary, ensure_ascii=False)}",
        "",
        "[모델]",
        f"조건부 U-Net base_channels={config.model.base_channels}, "
        f"multipliers={config.model.channel_multipliers}",
        f"diffusion: {config.diffusion.schedule} schedule, T={config.diffusion.timesteps}, "
        f"loss={config.diffusion.loss_type}",
        "",
        "[환경]",
        json.dumps(environment_info(), ensure_ascii=False, indent=2),
    ]
    if config.notes:
        lines += ["", "[메모]", config.notes]
    if extra_lines:
        lines += ["", *extra_lines]

    path = Path(run_folder) / "dataset_description.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_manifest(run_folder: Path, skip_suffixes: tuple[str, ...] = (".pt",)) -> Path:
    """산출물 목록과 checksum. 큰 checkpoint는 기본적으로 checksum을 생략한다."""
    run_folder = Path(run_folder)
    entries: list[dict] = []
    for path in sorted(run_folder.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        entry = {
            "path": str(path.relative_to(run_folder)).replace("\\", "/"),
            "size": path.stat().st_size,
        }
        entry["sha256"] = None if path.suffix in skip_suffixes else _file_sha256(path)
        entries.append(entry)
    payload = {
        "created_at": datetime.now().astimezone().isoformat(),
        "run_folder": run_folder.name,
        "files": entries,
    }
    return write_json(run_folder / "manifest.json", payload)

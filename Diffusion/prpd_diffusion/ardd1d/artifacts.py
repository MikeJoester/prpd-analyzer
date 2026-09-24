"""ARDD 1D 트랙의 run 산출물 기록.

폴더 생성·manifest·환경 정보는 2D 트랙과 같은 함수를 재사용한다. `dataset_description.txt`만
이 트랙 고유의 항목(표현 방식, CIN 축 매핑, 256 feature 예외 선언)이 있어 따로 쓴다.
"""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from ..runs.artifacts import environment_info

FEATURE_EXCEPTION_NOTE = (
    "이 트랙은 Analyzer의 256차원 mean/max feature를 딥러닝 입력으로 사용한다. "
    "`claude.md` 11절의 '256 feature를 딥러닝 입력으로 쓰지 않는다' 규칙에 대한 "
    "명시적 예외이며, 논문(ARDD-2025)과 같은 1차원 신호에서 기법을 검증하려는 목적이다. "
    "이 트랙의 복원 출력은 raw tensor로 되돌릴 수 없으므로 분류·이상탐지 트랙의 입력으로 "
    "넘기지 않는다."
)


DEFAULT_SYNTH_CHECK = Path("Results/diffusion_runs/ardd1d_synth_check.json")


def algorithm_record(config) -> dict:
    return config.algorithm_record()


def synthesis_check_summary(path: Path = DEFAULT_SYNTH_CHECK) -> dict | None:
    """`synth-check` 산출물이 있으면 판정과 진단을 읽어 온다.

    합성 전제가 검증되지 않은 채 학습한 결과를 나중에 "실측과 같다"고 오해하지 않도록,
    모든 run 설명에 이 판정을 그대로 옮긴다(`DIFFUSION.md` 14절 D1의 규칙).
    """
    path = Path(path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return {
        "source": str(path),
        "verdict": payload.get("verdict", ""),
        "baseline_frechet": payload.get("baseline_frechet"),
        "baseline_mmd": payload.get("baseline_mmd"),
        "diagnosis": payload.get("diagnosis", {}),
        "synthesis_limited": "합성 노이즈 한정" in str(payload.get("verdict", "")),
    }


def write_run_config(run_folder: Path, config, dataset) -> Path:
    """`run_config.json` — 설정 전체 + 선택한 ai_data 실행의 확정 경로와 버전."""
    payload = config.to_dict()
    payload["algorithm"] = config.algorithm_record()
    payload["feature_input_exception"] = True
    payload["feature_input_exception_reason"] = FEATURE_EXCEPTION_NOTE
    payload["ai_data_root_resolved"] = str(Path(dataset.root).resolve())
    payload["raw_data_version"] = str(dataset.run_config.get("raw_data_version", "unknown"))
    payload["config_hash"] = config.config_hash()
    payload["environment"] = environment_info()
    check = synthesis_check_summary()
    payload["synthesis_check"] = check
    payload["synthesis_limited"] = bool(check and check["synthesis_limited"])
    path = Path(run_folder) / "run_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_description(
    run_folder: Path,
    config,
    dataset,
    pool_counts: dict,
    split_summary: dict,
    extra_lines: list[str] | None = None,
) -> Path:
    """`dataset_description.txt` — 이 실행이 무엇을 어떻게 학습했는지 사람이 읽는 설명."""
    run_folder = Path(run_folder)
    components = config.components.to_params()
    lines = [
        "PRPD ARDD 1D denoising run description",
        "",
        f"생성 날짜 및 시간: {datetime.now().astimezone().isoformat()}",
        f"실행 폴더: {run_folder.name}",
        f"config hash: {config.config_hash()}",
        f"random seed: {config.seed}",
        "",
        "[입력 데이터]",
        f"AI dataset 경로: {Path(dataset.root).resolve()}",
        f"raw_data_version: {dataset.run_config.get('raw_data_version', 'unknown')}",
        "원본 .dat를 다시 해석하지 않고 Analyzer가 검증한 raw tensor만 사용함",
        "한 run 안에서 하나의 ai_data 실행만 사용하며 여러 실행을 합치지 않음",
        f"clean pool 그룹: {', '.join(config.data.clean_groups) or '(없음)'}",
        f"noise pool 그룹: {', '.join(config.data.noise_groups) or '(없음 — 파라메트릭 CIN 사용)'}",
        f"실제 noisy 평가 그룹: {', '.join(config.data.real_noisy_groups) or '(없음)'}",
        f"split별 pool 수: {json.dumps(pool_counts, ensure_ascii=False)}",
        "",
        "[표현]",
        f"representation: {config.data.representation}",
        "raw (128, 3600) → 위상 profile (2, 128) = [시간 평균 128, 시간 최대 128]",
        "두 채널을 이어붙이면 Analyzer의 256차원 mean/max feature와 정확히 같다.",
        "비가역 변환이므로 복원 결과를 128x3600으로 되돌릴 수 없다.",
        "mean 채널은 3,600 사이클 평균이라 시간축 노이즈에 원래 둔감하다 —",
        "채널별 개선폭을 반드시 분리해서 읽을 것.",
        "",
        "[256 feature 사용에 대한 예외 선언]",
        FEATURE_EXCEPTION_NOTE,
        "",
        "[학습 데이터 구성]",
        "같은 시점의 clean/noisy 실측 쌍이 없으므로 노이즈를 합성해 paired 데이터를 만든다.",
        "노이즈는 raw 영역에서 섞은 뒤 profile을 추출한다(profile에 직접 더하지 않는다).",
        f"노이즈 종류: {config.noise.kind}",
        f"mixing mode: {config.noise.mode}",
        f"noise gain 범위: {config.noise.gain_min} ~ {config.noise.gain_max}",
        f"노이즈 없는 샘플 비율(dropout): {config.noise.dropout_probability}",
        f"profile 캐시 사용: {config.data.use_profile_cache} "
        f"(파일당 실현 {config.data.realizations_per_file}개)",
    ]

    if config.noise.kind == "cin":
        lines += [
            "",
            "[CIN 성분과 축 매핑]",
            "논문(ARDD-2025)의 Composite Industrial Noise = 백색 + pink(1/f) + 전원주파수 협대역.",
            "우리 raw는 (위상 128, 시간 3600)이고 두 축의 물리적 의미가 다르다:",
            "  128 위상 bin = AC 1주기, 3600 시간 열 = 사이클 인덱스(약 1분치).",
            f"  백색 sigma={config.cin.white_sigma} (양축 i.i.d.)",
            f"  pink sigma={config.cin.pink_sigma}, 축={config.cin.pink_axis}",
            f"  전원주파수 진폭={list(config.cin.harmonic_amplitudes)}, "
            f"차수={list(config.cin.harmonic_orders)}, 축={config.cin.harmonic_axis}",
            f"  target_snr_db={config.cin.target_snr_db}",
            "축 매핑은 설계 판단이며, 대안(전부 시간축)도 config로 선택해 비교할 수 있다.",
        ]

    check = synthesis_check_summary()
    lines += ["", "[합성 전제 검증 (D1)]"]
    if check is None:
        lines += [
            "synth-check 산출물을 찾지 못했다. `cli synth-check`를 먼저 실행할 것.",
            "이 검증 없이 학습하면 '합성 노이즈만 잘 지우는 모델'이 되어도 알 수 없다.",
        ]
    else:
        diagnosis = check.get("diagnosis", {})
        lines += [
            f"근거 파일: {check['source']}",
            f"판정: {check['verdict']}",
            f"기준선(Lab PD ↔ Field PD): frechet={check['baseline_frechet']}, "
            f"mmd={check['baseline_mmd']}",
        ]
        if diagnosis.get("interpretation"):
            lines += [
                f"mean 채널 격차(Field−Lab): {diagnosis.get('mean_channel_gap')}",
                f"max 채널 격차(Field−Lab): {diagnosis.get('max_channel_gap')}",
                diagnosis["interpretation"],
            ]
        if check["synthesis_limited"]:
            lines += [
                "",
                "*** 이 실행의 결과는 '합성 노이즈 한정'이다. ***",
                "합성 noisy 분포가 실제 Field PD에 더 가까워지지 않았으므로, paired 지표의 개선을",
                "'실제 현장 노이즈를 제거한다'는 뜻으로 읽으면 안 된다.",
            ]

    lines += [
        "",
        "[분할]",
        "날짜 단위 분할 — 같은 날짜가 train과 test에 동시에 들어가지 않음",
        f"split 요약: {json.dumps(split_summary, ensure_ascii=False)}",
        "",
        "[모델과 논문 구성요소]",
        f"조건부 1D U-Net base_channels={config.model.base_channels}, "
        f"multipliers={tuple(config.model.channel_multipliers)}, "
        f"padding={config.model.padding_mode}",
        f"diffusion: {config.diffusion.schedule} schedule, T={config.diffusion.timesteps}, "
        f"loss={config.diffusion.loss_type}",
        f"DDIM 추론: {config.sample.steps} steps, eta={config.sample.eta}",
        "켜진 논문 구성요소(DIFFUSION.md 18.4): "
        + (json.dumps(components, ensure_ascii=False) if components else "없음 (baseline)"),
        "",
        "[환경]",
        json.dumps(environment_info(), ensure_ascii=False, indent=2),
        "",
        "[해석 주의]",
        "복원 출력은 노이즈 감소 결과이지 고장 판정이 아니다.",
        "논문의 분류 정확도 수치는 우리 개선 목표가 아니다(평가 지표가 다르다).",
    ]

    if config.notes:
        lines += ["", "[메모]", config.notes]
    if extra_lines:
        lines += ["", *extra_lines]

    path = run_folder / "dataset_description.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

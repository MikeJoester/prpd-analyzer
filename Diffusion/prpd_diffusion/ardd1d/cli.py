"""ARDD 1D 트랙 명령행 진입점.

    python -m Diffusion.prpd_diffusion.ardd1d.cli prepare  --config Diffusion/configs/ardd1d_base.json [--cache]
    python -m Diffusion.prpd_diffusion.ardd1d.cli train    --config Diffusion/configs/ardd1d_base.json
    python -m Diffusion.prpd_diffusion.ardd1d.cli sample   --run Results/diffusion_runs/ardd1d_...
    python -m Diffusion.prpd_diffusion.ardd1d.cli evaluate --run Results/diffusion_runs/ardd1d_...
    python -m Diffusion.prpd_diffusion.ardd1d.cli sweep    --run ... --snr 3,0,-3,-6,-9,-12
    python -m Diffusion.prpd_diffusion.ardd1d.cli compare  --runs ... --output Results/diffusion_runs/...

`prepare`와 `compare`는 torch 없이 동작한다. `train`/`sample`/`sweep`은 torch가 필요하고,
`evaluate`는 저장된 npz만 읽으므로 torch 없이 동작한다.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from ..contract import AiDataset, latest_run_folder
from ..data.pairs import PoolConfig, build_pools, describe_pools
from ..data.profile import profile_from_matrix, profiles_from_matrices
from ..data.splits import (
    check_no_date_leakage,
    coverage_report,
    load_split_config,
    make_date_splits,
    save_split_config,
)
from ..runs.artifacts import create_run_folder, write_json, write_manifest
from .artifacts import synthesis_check_summary, write_description, write_run_config
from .config import Ardd1dConfig
from .pairs import (
    PairFactory,
    build_profile_pairs,
    cache_folder,
    factory_from_config,
    load_profile_pairs,
    save_profile_pairs,
)

DEFAULT_CONFIG = Path("Diffusion/configs/ardd1d_base.json")
SPLITS = ("train", "val", "test")


# ------------------------------------------------------------------ 공통


def _resolve_dataset(config: Ardd1dConfig) -> AiDataset:
    root = Path(config.data.ai_data_root) if config.data.ai_data_root else None
    if root is None:
        root = latest_run_folder(Path("artifacts"))
        print(f"경고: ai_data_root 미지정 → 최신 실행 사용: {root}")
        print("      실행 폴더가 늘어나면 가리키는 대상이 바뀝니다. config에 명시하세요.")
    return AiDataset.load(root)


def _build_context(config: Ardd1dConfig, split_path: Path | None = None):
    dataset = _resolve_dataset(config)
    if split_path is not None and Path(split_path).is_file():
        date_split = load_split_config(Path(split_path))
    else:
        date_split = make_date_splits(dataset.metadata, config.split.ratios, config.seed)

    leakage = check_no_date_leakage(date_split)
    if leakage:
        raise RuntimeError("날짜 누수 발견: " + "; ".join(leakage))

    pools = build_pools(
        dataset.metadata,
        date_split,
        PoolConfig(
            clean_groups=tuple(config.data.clean_groups),
            noise_groups=tuple(config.data.noise_groups),
            real_noisy_groups=tuple(config.data.real_noisy_groups),
            clean_labels=tuple(config.data.clean_labels) if config.data.clean_labels else None,
        ),
    )
    return dataset, date_split, pools


def check_pools_1d(config: Ardd1dConfig, pools: dict) -> list[str]:
    """학습에 필요한 pool 점검.

    2D 트랙의 `data.pairs.check_pools`와 달리 **노이즈 pool이 비어 있어도 문제가 아니다** —
    `noise.kind="cin"`이면 노이즈를 파라메트릭으로 생성하므로 노이즈 그룹이 필요 없다.
    최신 생성 dataset(`ai_data_20260821_160405`)이 정확히 그런 경우다.
    """
    problems: list[str] = []
    for split in ("train", "val"):
        pool = pools.get(split)
        if pool is None:
            problems.append(f"split '{split}' 없음")
            continue
        if pool.clean.empty:
            problems.append(f"split '{split}'의 clean pool이 비어 있음")
        if config.noise.kind == "measured" and pool.noise.empty:
            problems.append(f"split '{split}'의 noise pool이 비어 있음 (kind='measured')")
    return problems


def _noise_bank(config: Ardd1dConfig, dataset, pool):
    """`measured` 모드에서만 NoiseBank를 만든다."""
    if config.noise.kind != "measured":
        return None
    from ..data.noise_model import NoiseAugmentConfig, NoiseBank

    return NoiseBank(
        dataset,
        pool.noise,
        NoiseAugmentConfig(
            mode=config.noise.mode,
            gain_min=1.0,
            gain_max=1.0,
            dropout_probability=0.0,
        ),
    )


def _pair_cache_path(config: Ardd1dConfig, dataset, split: str) -> Path:
    return cache_folder(Path(config.data.profile_cache), dataset, config) / f"{split}.npz"


def _ensure_profile_pairs(
    config: Ardd1dConfig, dataset, pools, split: str, rebuild: bool = False
) -> dict[str, np.ndarray]:
    """split의 profile 쌍을 캐시에서 읽거나 새로 만든다."""
    path = _pair_cache_path(config, dataset, split)
    if path.is_file() and not rebuild:
        return load_profile_pairs(path)

    factory = factory_from_config(config, _noise_bank(config, dataset, pools[split]))
    seed_offset = {"train": 0, "val": 1, "test": 2}[split]
    payload = build_profile_pairs(
        dataset,
        pools[split].clean,
        factory,
        realizations=config.data.realizations_per_file,
        seed=config.seed + seed_offset * 10_000,
    )
    save_profile_pairs(path, payload)
    print(f"  [{split}] profile 쌍 생성 → {path} "
          f"({payload['clean'].shape[0]} 파일 × {payload['noisy'].shape[1]} 실현)")
    return payload


def _make_dataset(config: Ardd1dConfig, dataset, pools, split: str, seed: int, samples_per_file: int):
    from .dataset import CachedProfilePairDataset, LiveProfilePairDataset

    if config.data.use_profile_cache:
        payload = _ensure_profile_pairs(config, dataset, pools, split)
        return CachedProfilePairDataset(payload, seed=seed, samples_per_file=samples_per_file)
    return LiveProfilePairDataset(
        dataset,
        pools[split].clean,
        factory_from_config(config, _noise_bank(config, dataset, pools[split])),
        seed=seed,
        samples_per_file=samples_per_file,
    )


def _build_model(config: Ardd1dConfig, device=None):
    from ..models.unet1d import build_from_config

    model = build_from_config(config)
    return model.to(device) if device is not None else model


def _build_diffusion(config: Ardd1dConfig, device=None):
    from ..diffusion.gaussian import GaussianDiffusion
    from ..diffusion.schedule import DiffusionSchedule

    schedule = DiffusionSchedule.create(config.diffusion.schedule, config.diffusion.timesteps)
    return GaussianDiffusion(schedule, device=device, loss_type=config.diffusion.loss_type)


# ----------------------------------------------------------------- prepare


def command_prepare(args: argparse.Namespace) -> None:
    config = Ardd1dConfig.load(args.config)
    dataset, date_split, pools = _build_context(config)

    print(f"\n[dataset] {dataset.root}")
    print(dataset.summary().to_string(index=False))
    print(f"\n날짜 수: {len(dataset.dates)}  샘플 수: {len(dataset.metadata)}")
    print(f"raw_data_version: {dataset.run_config.get('raw_data_version', 'unknown')}")

    print("\n[split] 날짜 단위 분할")
    for name in SPLITS:
        print(f"  {name:5s}: 날짜 {len(date_split.dates(name)):3d}개")

    print("\n[coverage] split × group × label")
    coverage = coverage_report(dataset.metadata, date_split)
    print(coverage.to_string(index=False))

    print("\n[pools]")
    print(describe_pools(pools).to_string(index=False))
    problems = check_pools_1d(config, pools)
    for problem in problems:
        print(f"  경고: {problem}")
    if config.noise.kind == "cin":
        print("  노이즈 pool은 비어 있어도 됩니다 (noise.kind='cin' — 파라메트릭 생성).")

    print("\n[평가 가능 조합] test split에 실제로 존재하는 (group, label)")
    test_table = date_split.apply(dataset.metadata)
    test_table = test_table[test_table["split"] == "test"]
    if test_table.empty:
        print("  없음")
    else:
        counts = test_table.groupby(["group", "label"]).size().reset_index(name="files")
        print(counts.to_string(index=False))
        print("  위 조합 밖의 (group, label)은 이 dataset에서 평가할 수 없습니다.")

    if args.cache:
        needed = tuple(
            set(config.data.clean_groups)
            | set(config.data.noise_groups)
            | set(config.data.real_noisy_groups)
        )
        target = dataset.materialize_memmap(Path(config.data.memmap_cache), needed)
        print(f"\n[cache] memmap 캐시(실행별 분리): {target}")
        if not problems:
            print("[cache] profile 쌍 생성")
            for split in SPLITS:
                if pools[split].clean.empty:
                    print(f"  [{split}] clean pool이 비어 있어 건너뜁니다.")
                    continue
                _ensure_profile_pairs(config, dataset, pools, split, rebuild=args.rebuild_cache)

    if args.verify_manifest:
        issues = dataset.verify_manifest()
        print("\n[manifest] " + ("정상" if not issues else "; ".join(issues)))

    if problems:
        raise SystemExit("pool 구성 오류가 있어 학습을 시작할 수 없습니다: " + "; ".join(problems))


# -------------------------------------------------------------- synth-check


def command_synth_check(args: argparse.Namespace) -> None:
    """합성 전제 검증 (`DIFFUSION.md` 14절 D1에 해당). torch 불필요.

    질문: `mix(Lab PD, CIN)`이 실제 `Field PD`와 얼마나 닮았는가?
    기준선은 **아무것도 섞지 않은 `Lab PD` ↔ `Field PD` 거리**다. 합성이 이보다 가깝지 않으면
    "합성 노이즈만 잘 지우는 모델"이 되기 쉽고, 이후 결과는 모두 합성 한정으로 표기해야 한다.

    이 단계를 건너뛰면 학습 결과의 해석이 무의미해진다.
    """
    import pandas as pd

    from ..evaluation.report import write_html_report
    from .metrics import feature_matrix, frechet_distance, mmd_rbf, standardize

    config = Ardd1dConfig.load(args.config)
    dataset, date_split, pools = _build_context(config)
    if config.noise.kind != "cin":
        raise SystemExit("synth-check는 noise.kind='cin'에서만 의미가 있습니다.")

    clean_table = pools["train"].clean.head(args.files)
    real_table = pools["train"].real_noisy.head(args.files)
    if clean_table.empty or real_table.empty:
        raise SystemExit("train clean 또는 real_noisy pool이 비어 있어 검증할 수 없습니다.")

    clean_matrices = [dataset.matrix(str(value)) for value in clean_table["sample_id"]]
    clean_profiles = np.stack([profile_from_matrix(matrix) for matrix in clean_matrices])
    real_profiles = profiles_from_matrices(dataset.matrices(real_table))
    reference = feature_matrix(clean_profiles)
    real_features = feature_matrix(real_profiles)

    def distances(candidate: np.ndarray) -> tuple[float, float]:
        scaled_reference, scaled_real, scaled_candidate = standardize(
            reference, real_features, candidate
        )
        return (
            frechet_distance(scaled_real, scaled_candidate),
            mmd_rbf(scaled_real, scaled_candidate),
        )

    base_frechet, base_mmd = distances(reference)
    rows = [
        {
            "mode": "(없음)",
            "scale": 0.0,
            "note": "Lab PD ↔ Field PD — 넘어야 할 기준선",
            "frechet": base_frechet,
            "mmd": base_mmd,
        }
    ]

    base_cin = config.cin_config()
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    scales = [float(value) for value in args.scale.split(",") if value.strip()]
    for mode in modes:
        for scale in scales:
            cin = replace(
                base_cin,
                white_sigma=base_cin.white_sigma * scale,
                pink_sigma=base_cin.pink_sigma * scale,
                harmonic_amplitudes=tuple(
                    amplitude * scale for amplitude in base_cin.harmonic_amplitudes
                ),
            )
            factory = PairFactory(
                kind="cin",
                mode=mode,
                gain_min=1.0,
                gain_max=1.0,
                dropout_probability=0.0,
                cin=cin,
            )
            synthetic = np.stack(
                [
                    factory.make_pair(matrix, np.random.default_rng((config.seed, index)))[0]
                    for index, matrix in enumerate(clean_matrices)
                ]
            )
            frechet, mmd = distances(feature_matrix(synthetic))
            rows.append(
                {
                    "mode": mode,
                    "scale": scale,
                    "note": "",
                    "frechet": frechet,
                    "mmd": mmd,
                }
            )
            print(f"  {mode:10s} scale {scale:4.2f} → frechet {frechet:9.3f}  mmd {mmd:8.5f}")

    frame = pd.DataFrame(rows)
    candidates = frame[frame["mode"] != "(없음)"]
    # 노이즈가 너무 약해 profile이 사실상 바뀌지 않으면 거리도 기준선과 같아진다.
    # 소수점 아래 흔들림을 "개선"으로 읽지 않도록 상대 여유를 둔다(18.6의 정신).
    margin = 1.0 - args.tolerance
    closer = candidates[
        (candidates["frechet"] < base_frechet * margin)
        & (candidates["mmd"] < base_mmd * margin)
    ]
    best = candidates.sort_values("frechet").head(1)

    verdict = (
        "합성 분포가 기준선보다 실제 Field PD에 가깝다. 해당 설정을 config에 반영할 것."
        if not closer.empty
        else (
            f"어떤 설정도 기준선(Lab PD ↔ Field PD)보다 {args.tolerance:.0%} 이상 가깝지 않다. "
            "이후 학습 결과는 모두 '합성 노이즈 한정'으로 표기해야 한다."
        )
    )

    # 왜 그런지까지 남긴다 — 판정만 있으면 다음 사람이 같은 조사를 다시 하게 된다.
    diagnosis = {
        "lab_mean_channel": float(clean_profiles[:, 0].mean()),
        "field_mean_channel": float(real_profiles[:, 0].mean()),
        "lab_max_channel": float(clean_profiles[:, 1].mean()),
        "field_max_channel": float(real_profiles[:, 1].mean()),
    }
    diagnosis["mean_channel_gap"] = diagnosis["field_mean_channel"] - diagnosis["lab_mean_channel"]
    diagnosis["max_channel_gap"] = diagnosis["field_max_channel"] - diagnosis["lab_max_channel"]
    if diagnosis["mean_channel_gap"] < 0 < diagnosis["max_channel_gap"]:
        diagnosis["interpretation"] = (
            "Field PD는 Lab PD보다 mean profile이 낮고 max profile이 높다. 즉 시간축에서 "
            "더 드물지만 더 큰 펄스를 낸다. 음수가 없는 노이즈를 섞으면 두 채널이 함께 "
            "올라가므로 이 차이를 재현할 수 없다 — Lab→Field 격차는 '노이즈 바닥' 차이가 아니다."
        )
    else:
        diagnosis["interpretation"] = (
            "두 채널의 차이 방향이 같으므로 노이즈 합성으로 격차를 좁힐 여지가 있다."
        )
    print(
        f"\n[진단] mean 채널 Lab {diagnosis['lab_mean_channel']:.2f} → Field "
        f"{diagnosis['field_mean_channel']:.2f} ({diagnosis['mean_channel_gap']:+.2f})"
    )
    print(
        f"       max  채널 Lab {diagnosis['lab_max_channel']:.2f} → Field "
        f"{diagnosis['field_max_channel']:.2f} ({diagnosis['max_channel_gap']:+.2f})"
    )
    print(f"       {diagnosis['interpretation']}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    write_json(
        output.with_suffix(".json"),
        {
            "ai_data_root": str(Path(dataset.root).resolve()),
            "clean_files": int(len(clean_table)),
            "real_files": int(len(real_table)),
            "baseline_frechet": base_frechet,
            "baseline_mmd": base_mmd,
            "tolerance": args.tolerance,
            "verdict": verdict,
            "diagnosis": diagnosis,
            "best_by_frechet": best.to_dict("records"),
            "rows": rows,
        },
    )
    write_html_report(
        output.with_suffix(".html"),
        "ARDD 1D 합성 전제 검증 (D1)",
        {
            "baseline_frechet": base_frechet,
            "baseline_mmd": base_mmd,
            "verdict": verdict,
            "diagnosis": diagnosis,
        },
        [],
        notes=[
            "기준선은 아무것도 섞지 않은 Lab PD ↔ Field PD 거리다. 합성이 이보다 가까워야 한다.",
            f"표본 수({len(clean_table)}, {len(real_table)})가 feature 차원(256)보다 작으면 "
            "Fréchet distance가 불안정하다. MMD와 함께 읽을 것.",
            "노이즈가 지나치게 약하면 profile이 거의 안 바뀌어 거리도 기준선과 같아진다. "
            f"상대 여유 {args.tolerance:.0%}를 넘겨야 '개선'으로 인정한다.",
            "<pre>" + frame.to_string(index=False) + "</pre>",
        ],
    )

    print(f"\n기준선 (Lab PD ↔ Field PD): frechet {base_frechet:.3f}  mmd {base_mmd:.5f}")
    print(f"판정: {verdict}")
    print(f"결과: {output.with_suffix('.csv')}")


# ------------------------------------------------------------------- train


def command_train(args: argparse.Namespace) -> None:
    from ..training.trainer import Trainer

    config = Ardd1dConfig.load(args.config)
    if args.train_seed is not None:
        # 모델 초기화·배치 순서만 바꾼다. 날짜 분할과 합성 노이즈는 `config.seed`가 정하므로
        # 후보 간 비교 조건은 그대로 유지된다(DIFFUSION.md 18.1).
        config.train_seed = int(args.train_seed)
        config.run_prefix = f"{config.run_prefix}s{config.train_seed}"
    dataset, date_split, pools = _build_context(config)

    problems = check_pools_1d(config, pools)
    if problems:
        raise RuntimeError("pool 구성 오류: " + "; ".join(problems))

    if args.cache:
        dataset.materialize_memmap(
            Path(config.data.memmap_cache),
            tuple(set(config.data.clean_groups) | set(config.data.noise_groups)),
        )

    train_dataset = _make_dataset(
        config, dataset, pools, "train", config.seed, config.data.samples_per_file
    )
    val_dataset = _make_dataset(config, dataset, pools, "val", config.seed + 1, 1)

    run_folder = create_run_folder(prefix=config.run_prefix)
    config.data.ai_data_root = str(dataset.root)
    write_run_config(run_folder, config, dataset)
    save_split_config(run_folder / "split_config.json", date_split, dataset.metadata)

    model = _build_model(config)
    parameters = model.parameter_count()
    print(f"모델 파라미터 수: {parameters:,}")
    components = config.components.to_params()
    print(f"켜진 구성요소: {json.dumps(components, ensure_ascii=False) or '없음 (baseline)'}")

    diffusion = _build_diffusion(config)
    write_description(
        run_folder,
        config,
        dataset,
        {name: pool.counts() for name, pool in pools.items()},
        {name: len(date_split.dates(name)) for name in SPLITS},
        extra_lines=[f"모델 파라미터 수: {parameters:,}"],
    )

    # Trainer는 `config.seed`(모델 초기화)와 `config.train.*`만 참조한다. 학습 seed만 바꾸려고
    # seed 필드만 교체한 사본을 넘긴다 — dataset은 이미 만들어졌으므로 데이터 쪽 seed는
    # 영향을 받지 않는다. (Trainer가 checkpoint에 `asdict(config)`를 넣으므로 dataclass여야 한다.)
    trainer_config = replace(config, seed=config.effective_train_seed)
    trainer = Trainer(model, diffusion, train_dataset, val_dataset, run_folder, trainer_config)
    best = trainer.train()
    write_manifest(run_folder)
    print(f"\n학습 완료. best checkpoint: {best}")
    print(f"결과 폴더: {run_folder}")


# ------------------------------------------------------------------ sample


def _restore(config, diffusion, model, noisy_profiles, device, seed):
    from .dataset import denoise_profiles

    return denoise_profiles(
        diffusion,
        model,
        noisy_profiles,
        steps=config.sample.steps,
        eta=config.sample.eta,
        batch_size=config.sample.batch_size,
        device=device,
        seed=seed,
    )


def _load_trained(run_folder: Path, device_name: str | None):
    import torch

    from ..training.trainer import load_checkpoint

    config = Ardd1dConfig.load(run_folder / "run_config.json")
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _build_model(config, device)
    checkpoint = load_checkpoint(run_folder / "checkpoints" / "best.pt", model)
    model.eval()
    print(f"checkpoint epoch={checkpoint.get('epoch')} val_loss={checkpoint.get('val_loss')}")
    return config, model, _build_diffusion(config, device), device


def command_sample(args: argparse.Namespace) -> None:
    import torch

    from ..training.trainer import load_checkpoint

    run_folder = Path(args.run)
    config = Ardd1dConfig.load(run_folder / "run_config.json")
    dataset, date_split, pools = _build_context(config, run_folder / "split_config.json")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = _build_model(config, device)
    checkpoint = load_checkpoint(run_folder / "checkpoints" / args.checkpoint, model)
    model.eval()
    print(f"checkpoint epoch={checkpoint.get('epoch')} val_loss={checkpoint.get('val_loss')}")
    diffusion = _build_diffusion(config, device)

    limit = config.sample.max_files

    # 1) 합성 paired test — 정답 clean이 있으므로 오차를 직접 계산한다.
    if pools["test"].clean.empty:
        print("경고: test clean pool이 비어 있어 paired 샘플을 만들지 않습니다.")
    else:
        payload = _ensure_profile_pairs(config, dataset, pools, "test")
        count = min(limit, payload["clean"].shape[0])
        clean = payload["clean"][:count]
        noisy = payload["noisy"][:count, 0]  # 평가는 항상 0번 실현으로 고정해 비교 가능하게 둔다
        restored = _restore(config, diffusion, model, noisy, device, config.seed)
        np.savez_compressed(
            run_folder / "samples" / "paired_test.npz",
            clean=clean,
            noisy=noisy,
            restored=restored,
            sample_id=payload["sample_ids"][:count].astype(str),
        )
        print(f"  paired {count}개 저장")

    # 2) 실제 Field PD — 정답이 없으므로 분포 지표로만 평가한다.
    real_table = pools["test"].real_noisy.head(limit)
    if real_table.empty:
        print("경고: test real_noisy pool이 비어 있어 분포 평가용 샘플이 없습니다.")
    else:
        real_noisy = profiles_from_matrices(dataset.matrices(real_table))
        real_restored = _restore(config, diffusion, model, real_noisy, device, config.seed)
        np.savez_compressed(
            run_folder / "samples" / "real_noisy.npz",
            noisy=real_noisy,
            restored=real_restored,
            sample_id=np.array([str(value) for value in real_table["sample_id"]]),
        )
        print(f"  real {len(real_table)}개 저장")

    write_json(
        run_folder / "samples" / "sample_config.json",
        {
            "checkpoint": args.checkpoint,
            "ddim_steps": config.sample.steps,
            "eta": config.sample.eta,
            "seed": config.seed,
            "device": str(device),
            "realization_index": 0,
        },
    )
    write_manifest(run_folder)
    print(f"\n샘플 저장 완료: {run_folder / 'samples'}")


# ---------------------------------------------------------------- evaluate


def command_evaluate(args: argparse.Namespace) -> None:
    from ..evaluation.report import write_html_report
    from .metrics import distribution_shift, feature_matrix, improved_channels, paired_profile_metrics
    from .report import channel_improvement_figure, profile_pair_figure

    run_folder = Path(args.run)
    config = Ardd1dConfig.load(run_folder / "run_config.json")
    dataset, date_split, pools = _build_context(config, run_folder / "split_config.json")

    metrics: dict = {
        "run": run_folder.name,
        "config_hash": config.config_hash(),
        "algorithm": config.algorithm_record(),
        "ai_data_root": str(Path(dataset.root).resolve()),
        "raw_data_version": str(dataset.run_config.get("raw_data_version", "unknown")),
        "feature_input_exception": True,
    }
    check = synthesis_check_summary()
    metrics["synthesis_limited"] = bool(check and check["synthesis_limited"])
    if check:
        metrics["synthesis_check_verdict"] = check["verdict"]
    figures = []

    paired_path = run_folder / "samples" / "paired_test.npz"
    if paired_path.is_file():
        with np.load(paired_path, allow_pickle=False) as archive:
            clean, noisy, restored = archive["clean"], archive["noisy"], archive["restored"]
        paired = paired_profile_metrics(restored, clean, noisy)
        paired["files"] = int(len(clean))
        paired["improved_channels"] = improved_channels(paired)
        metrics["paired"] = paired
        figures.append(profile_pair_figure(noisy[0], restored[0], clean[0]))
        figures.append(channel_improvement_figure(paired))

    real_path = run_folder / "samples" / "real_noisy.npz"
    if real_path.is_file():
        with np.load(real_path, allow_pickle=False) as archive:
            real_noisy, real_restored = archive["noisy"], archive["restored"]
        reference_table = pools["train"].clean.head(args.reference_files)
        reference = feature_matrix(profiles_from_matrices(dataset.matrices(reference_table)))
        shift = distribution_shift(
            reference, feature_matrix(real_noisy), feature_matrix(real_restored)
        )
        shift["reference_files"] = int(len(reference_table))
        shift["evaluated_files"] = int(len(real_noisy))
        metrics["distribution"] = shift
        figures.append(
            profile_pair_figure(real_noisy[0], real_restored[0], title="실제 Field PD 복원 예시")
        )

    limitation = (
        ["*** 이 실행의 결과는 '합성 노이즈 한정'이다 — 합성 noisy 분포가 실제 Field PD에 "
         "더 가까워지지 않았다. paired 지표의 개선을 '실제 현장 노이즈 제거'로 읽지 말 것. "
         f"근거: {check['verdict'] if check else ''}"]
        if metrics["synthesis_limited"]
        else []
    )
    write_json(run_folder / "metrics.json", metrics)
    write_html_report(
        run_folder / "evaluation_report.html",
        f"ARDD 1D denoising 평가 — {run_folder.name}",
        metrics,
        figures,
        notes=limitation + [
            "표현은 위상 profile(2×128)이며 비가역이다. 128×3600 raw로 되돌릴 수 없다.",
            "mean 채널은 3,600 사이클 평균이라 시간축 노이즈에 원래 둔감하다. "
            "채널별 개선폭을 따로 읽을 것.",
            "paired 지표는 합성 노이즈로 만든 test 쌍 기준이다.",
            "distribution 지표는 정답이 없는 실제 Field PD를 clean 분포(Lab PD)와 비교한 것이다.",
            "복원 결과는 노이즈 감소 정도이지 물리적 고장 판정이 아니다.",
            "논문(ARDD-2025)의 분류 정확도 수치는 평가 지표가 달라 개선 목표로 쓰지 않는다.",
        ],
    )
    write_manifest(run_folder)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\n리포트: {run_folder / 'evaluation_report.html'}")


# ------------------------------------------------------------------- sweep


def command_sweep(args: argparse.Namespace) -> None:
    """`ARDD-2025`의 로버스트니스 프로토콜. 지표는 우리 것(profile 공간)을 쓴다.

    * SNR 스윕: +3 → −12 dB, 3 dB 간격 (`--snr`)
    * 연속 결측 주입: 시간축 5~15% (`--burst-missing`)

    `identity` baseline(무처리)이 항상 같은 표에 들어가므로 "노이즈가 심해질 때 무너지는
    지점"을 모델과 baseline 양쪽에서 읽을 수 있다.
    """
    from ..data.cin_noise import apply_burst_missing
    from ..evaluation.report import write_html_report
    from .metrics import paired_profile_metrics
    from .report import missing_sweep_figure, snr_sweep_figure

    run_folder = Path(args.run)
    config, model, diffusion, device = _load_trained(run_folder, args.device)
    dataset, date_split, pools = _build_context(config, run_folder / "split_config.json")
    if pools["test"].clean.empty:
        raise RuntimeError("test clean pool이 비어 있어 스윕을 실행할 수 없습니다.")

    table = pools["test"].clean.head(config.sample.max_files)
    matrices = [dataset.matrix(str(sample_id)) for sample_id in table["sample_id"]]
    clean = np.stack([profile_from_matrix(matrix) for matrix in matrices])
    # 스윕은 노이즈 강도를 축으로 삼으므로 무작위 gain과 노이즈 dropout을 끈다.
    # 켜 두면 목표 SNR이 흐려져 곡선의 x축이 의미를 잃는다.
    base_factory = replace(
        factory_from_config(config, _noise_bank(config, dataset, pools["test"])),
        gain_min=1.0,
        gain_max=1.0,
        dropout_probability=0.0,
    )

    def evaluate(noisy: np.ndarray) -> dict:
        restored = _restore(config, diffusion, model, noisy, device, config.seed)
        return paired_profile_metrics(restored, clean, noisy)

    results: dict = {"run": run_folder.name, "files": int(len(table))}
    figures = []

    if args.snr:
        if config.noise.kind != "cin":
            raise RuntimeError("SNR 스윕은 파라메트릭 노이즈(noise.kind='cin')에서만 정의됩니다.")
        rows = []
        for snr_db in [float(value) for value in args.snr.split(",")]:
            factory = replace(base_factory, cin=replace(base_factory.cin, target_snr_db=snr_db))
            noisy = np.stack(
                [
                    factory.make_pair(matrix, np.random.default_rng((config.seed, index)))[0]
                    for index, matrix in enumerate(matrices)
                ]
            )
            row = {"snr_db": snr_db, **evaluate(noisy)}
            rows.append(row)
            print(f"  SNR {snr_db:+5.1f} dB → max MAE {row['max_profile_mae']:.2f} "
                  f"(baseline {row['baseline_max_profile_mae']:.2f})")
        results["snr_sweep"] = rows
        figures.append(snr_sweep_figure(rows))

    if args.burst_missing:
        rows = []
        for fraction in [float(value) for value in args.burst_missing.split(",")]:
            noisy = np.stack(
                [
                    base_factory.make_pair(
                        apply_burst_missing(
                            matrix, np.random.default_rng((config.seed, index)), fraction
                        ),
                        np.random.default_rng((config.seed, index)),
                    )[0]
                    for index, matrix in enumerate(matrices)
                ]
            )
            row = {"fraction": fraction, **evaluate(noisy)}
            rows.append(row)
            print(f"  결측 {fraction:.0%} → max MAE {row['max_profile_mae']:.2f} "
                  f"(baseline {row['baseline_max_profile_mae']:.2f})")
        results["burst_missing_sweep"] = rows
        figures.append(missing_sweep_figure(rows))

    if not figures:
        raise SystemExit("--snr 또는 --burst-missing 중 하나 이상을 지정하세요.")

    write_json(run_folder / "robustness.json", results)
    write_html_report(
        run_folder / "robustness_report.html",
        f"ARDD 1D 로버스트니스 — {run_folder.name}",
        results,
        figures,
        notes=[
            "프로토콜은 ARDD-2025에서 가져왔고 지표는 우리 profile 공간 지표다.",
            "결측 실험의 clean 목표는 결측을 넣지 않은 원본이다 — 복원이 결측 구간을 "
            "메우지 못하면 baseline보다 나빠질 수 있고, 그 지점이 곧 한계다.",
        ],
    )
    write_manifest(run_folder)
    print(f"\n리포트: {run_folder / 'robustness_report.html'}")


# ----------------------------------------------------------------- compare


def command_compare(args: argparse.Namespace) -> None:
    """여러 run의 지표를 `DIFFUSION.md` 18.1 형식의 비교 표로 모은다. torch 불필요."""
    import pandas as pd

    from ..evaluation.report import write_html_report

    rows = []
    references = []
    for run_text in args.runs:
        run_folder = Path(run_text)
        metrics_path = run_folder / "metrics.json"
        if not metrics_path.is_file():
            rows.append({"run": run_folder.name, "status": "metrics.json 없음 (미실행/실패)"})
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        paired = metrics.get("paired", {})
        distribution = metrics.get("distribution", {})
        algorithm = metrics.get("algorithm", {})
        rows.append(
            {
                "run": run_folder.name,
                "status": "ok",
                "schedule": algorithm.get("schedule"),
                "components": ",".join(sorted(algorithm.get("components", {}))) or "none",
                "noise_model": algorithm.get("noise_model"),
                "mean_mae": paired.get("mean_profile_mae"),
                "max_mae": paired.get("max_profile_mae"),
                "mean_mae_improvement": paired.get("mean_profile_mae_improvement"),
                "max_mae_improvement": paired.get("max_profile_mae_improvement"),
                "frechet_before": distribution.get("frechet_before"),
                "frechet_after": distribution.get("frechet_after"),
                "mmd_before": distribution.get("mmd_before"),
                "mmd_after": distribution.get("mmd_after"),
            }
        )
        references.append(
            {
                "run": run_folder.name,
                "ai_data_root": metrics.get("ai_data_root"),
                "raw_data_version": metrics.get("raw_data_version"),
                "config_hash": metrics.get("config_hash"),
                "split_config": str(run_folder / "split_config.json"),
            }
        )

    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "comparison.csv", index=False, encoding="utf-8-sig")
    try:
        frame.to_parquet(output / "comparison.parquet", index=False)
    except Exception as error:  # noqa: BLE001 - parquet 엔진이 없어도 CSV는 남긴다
        print(f"경고: parquet 저장 실패({error}). CSV만 남깁니다.")
    write_json(output / "runs.json", {"runs": references})

    datasets = {entry["ai_data_root"] for entry in references if entry["ai_data_root"]}
    notes = [
        "같은 dataset·같은 분할·같은 합성 노이즈·같은 지표에서만 비교가 성립한다(18.1).",
        "개선폭이 seed 간 변동 범위 안이면 '차이 없음'으로 읽을 것.",
        "mean/max 채널을 합치지 않는다. 한쪽만 개선되는 것이 정상일 수 있다.",
    ]
    if len(datasets) > 1:
        notes.insert(0, f"경고: run들이 서로 다른 ai_data_root를 사용했다 → 비교 불가: {datasets}")

    write_html_report(
        output / "comparison_report.html",
        f"ARDD 1D 후보 비교 — {output.name}",
        {"runs": len(rows), "ai_data_roots": sorted(datasets)},
        [],
        notes=notes + ["<pre>" + frame.to_string(index=False) + "</pre>"],
    )
    print(frame.to_string(index=False))
    print(f"\n비교 결과: {output}")


# --------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prpd-ardd1d", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="데이터·split·pool 점검 (torch 불필요)")
    prepare.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prepare.add_argument("--cache", action="store_true", help="memmap + profile 쌍 캐시 생성")
    prepare.add_argument("--rebuild-cache", action="store_true", help="profile 쌍 캐시 재생성")
    prepare.add_argument("--verify-manifest", action="store_true", help="입력 checksum 검증")
    prepare.set_defaults(func=command_prepare)

    synth = subparsers.add_parser(
        "synth-check", help="합성 노이즈가 실제 Field PD와 닮았는지 검증 (D1, torch 불필요)"
    )
    synth.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    synth.add_argument("--files", type=int, default=256, help="각 집합에서 쓸 파일 수 상한")
    synth.add_argument("--modes", default="maximum,additive,quadrature")
    synth.add_argument("--scale", default="0.5,1.0,2.0", help="CIN 전체 진폭 배율 후보")
    synth.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="기준선 대비 이 비율 이상 가까워야 '개선'으로 인정 (기본 5%%)",
    )
    synth.add_argument(
        "--output", type=Path, default=Path("Results/diffusion_runs/ardd1d_synth_check")
    )
    synth.set_defaults(func=command_synth_check)

    train = subparsers.add_parser("train", help="조건부 1D diffusion denoiser 학습")
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    train.add_argument("--cache", action="store_true", help="학습 전 memmap 캐시 생성")
    train.add_argument(
        "--train-seed",
        type=int,
        default=None,
        help="모델 초기화·배치 순서 seed만 교체 (분할·합성 노이즈는 config.seed 유지). "
        "18.1의 seed 반복 실행용",
    )
    train.set_defaults(func=command_train)

    sample = subparsers.add_parser("sample", help="test split 복원 결과 생성")
    sample.add_argument("--run", type=Path, required=True)
    sample.add_argument("--checkpoint", default="best.pt")
    sample.add_argument("--device", default=None)
    sample.set_defaults(func=command_sample)

    evaluate = subparsers.add_parser("evaluate", help="저장된 복원 결과 평가 (torch 불필요)")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--reference-files", type=int, default=64)
    evaluate.set_defaults(func=command_evaluate)

    sweep = subparsers.add_parser("sweep", help="SNR·결측 로버스트니스 프로토콜")
    sweep.add_argument("--run", type=Path, required=True)
    sweep.add_argument("--snr", default="", help="예: 3,0,-3,-6,-9,-12")
    sweep.add_argument("--burst-missing", default="", help="예: 0.05,0.10,0.15")
    sweep.add_argument("--device", default=None)
    sweep.set_defaults(func=command_sweep)

    compare = subparsers.add_parser("compare", help="여러 run의 지표를 한 표로 (torch 불필요)")
    compare.add_argument("--runs", nargs="+", required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.set_defaults(func=command_compare)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

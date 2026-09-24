"""명령행 진입점.

    python -m Diffusion.prpd_diffusion.cli prepare  --config Diffusion/configs/denoise_base.json [--cache]
    python -m Diffusion.prpd_diffusion.cli train    --config Diffusion/configs/denoise_base.json
    python -m Diffusion.prpd_diffusion.cli sample   --run Results/diffusion_runs/denoise_YYYYMMDD_HHMMSS
    python -m Diffusion.prpd_diffusion.cli evaluate --run Results/diffusion_runs/denoise_YYYYMMDD_HHMMSS

`prepare`와 `evaluate`는 torch 없이 동작한다(데이터 점검·지표 계산 전용).
`train`과 `sample`만 torch를 필요로 한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .contract import AiDataset, latest_run_folder
from .data.noise_model import NoiseAugmentConfig, NoiseBank
from .data.pairs import PoolConfig, build_pools, check_pools, describe_pools
from .data.splits import (
    check_no_date_leakage,
    coverage_report,
    load_split_config,
    make_date_splits,
    save_split_config,
)
from .runs.artifacts import create_run_folder, write_description, write_json, write_manifest
from .runs.config import DenoiseConfig


# ------------------------------------------------------------------ 공통


def _resolve_dataset(config: DenoiseConfig) -> AiDataset:
    root = Path(config.data.ai_data_root) if config.data.ai_data_root else None
    if root is None:
        root = latest_run_folder(Path("artifacts"))
        print(f"ai_data_root 미지정 → 최신 실행 사용: {root}")
    return AiDataset.load(root)


def _build_context(config: DenoiseConfig, split_path: Path | None = None):
    """dataset, date split, pools를 한 번에 준비한다."""
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


def _noise_config(config: DenoiseConfig) -> NoiseAugmentConfig:
    return NoiseAugmentConfig(
        mode=config.noise.mode,
        gain_min=config.noise.gain_min,
        gain_max=config.noise.gain_max,
        phase_roll=config.noise.phase_roll,
        time_roll=config.noise.time_roll,
        dropout_probability=config.noise.dropout_probability,
    )


# ----------------------------------------------------------------- prepare


def command_prepare(args: argparse.Namespace) -> None:
    config = DenoiseConfig.load(args.config)
    dataset, date_split, pools = _build_context(config)

    print(f"\n[dataset] {dataset.root}")
    print(dataset.summary().to_string(index=False))
    print(f"\n날짜 수: {len(dataset.dates)}  샘플 수: {len(dataset.metadata)}")

    print("\n[split] 날짜 단위 분할")
    for name in ("train", "val", "test"):
        dates = date_split.dates(name)
        print(f"  {name:5s}: 날짜 {len(dates):3d}개")
    print("\n[coverage] split × group × label")
    print(coverage_report(dataset.metadata, date_split).to_string(index=False))

    print("\n[pools]")
    print(describe_pools(pools).to_string(index=False))
    problems = check_pools(pools)
    for problem in problems:
        print(f"  경고: {problem}")

    if args.cache:
        needed = tuple(
            set(config.data.clean_groups)
            | set(config.data.noise_groups)
            | set(config.data.real_noisy_groups)
        )
        target = dataset.materialize_memmap(Path(config.data.memmap_cache), needed)
        print(f"\n[cache] memmap 캐시 생성: {target}")

    if args.verify_manifest:
        issues = dataset.verify_manifest()
        print("\n[manifest] " + ("정상" if not issues else "; ".join(issues)))


# ------------------------------------------------------------------- train


def command_train(args: argparse.Namespace) -> None:
    from .data.torch_dataset import PairedCropDataset
    from .diffusion.gaussian import GaussianDiffusion
    from .diffusion.schedule import DiffusionSchedule
    from .models.unet import ConditionalUNet
    from .training.trainer import Trainer

    config = DenoiseConfig.load(args.config)
    dataset, date_split, pools = _build_context(config)

    problems = check_pools(pools)
    if problems:
        raise RuntimeError("pool 구성 오류: " + "; ".join(problems))

    if args.cache:
        dataset.materialize_memmap(
            Path(config.data.memmap_cache),
            tuple(set(config.data.clean_groups) | set(config.data.noise_groups)),
        )

    run_folder = create_run_folder(prefix=config.run_prefix)
    config.data.ai_data_root = str(dataset.root)
    config.save(run_folder / "run_config.json")
    save_split_config(run_folder / "split_config.json", date_split, dataset.metadata)

    noise_config = _noise_config(config)
    train_dataset = PairedCropDataset(
        dataset,
        pools["train"].clean,
        NoiseBank(dataset, pools["train"].noise, noise_config),
        crop_width=config.data.crop_width,
        seed=config.seed,
        samples_per_file=config.data.samples_per_file,
    )
    val_dataset = PairedCropDataset(
        dataset,
        pools["val"].clean,
        NoiseBank(dataset, pools["val"].noise, noise_config),
        crop_width=config.data.crop_width,
        seed=config.seed + 1,
        samples_per_file=1,
    )

    if getattr(config, "type", "base") == "ldm":
        from .models.ldm_wrapper import LDMUNetWrapper
        model = LDMUNetWrapper(model_channels=config.model.unet_channels)
    else:
        model = ConditionalUNet(
            base_channels=config.model.base_channels,
            channel_multipliers=tuple(config.model.channel_multipliers),
            blocks_per_stage=config.model.blocks_per_stage,
            attention_stages=tuple(config.model.attention_stages),
            dropout=config.model.dropout,
        )
    print(f"모델 파라미터 수: {model.parameter_count():,}")

    schedule = DiffusionSchedule.create(config.diffusion.schedule, config.diffusion.timesteps)
    diffusion = GaussianDiffusion(schedule, loss_type=config.diffusion.loss_type)

    write_description(
        run_folder,
        config,
        dataset.root,
        {name: pool.counts() for name, pool in pools.items()},
        {name: len(date_split.dates(name)) for name in ("train", "val", "test")},
        extra_lines=[f"모델 파라미터 수: {model.parameter_count():,}"],
    )

    trainer = Trainer(model, diffusion, train_dataset, val_dataset, run_folder, config)
    best = trainer.train()
    write_manifest(run_folder)
    print(f"\n학습 완료. best checkpoint: {best}")
    print(f"결과 폴더: {run_folder}")


# ------------------------------------------------------------------ sample


def command_sample(args: argparse.Namespace) -> None:
    import torch

    from .diffusion.gaussian import GaussianDiffusion
    from .diffusion.sampler import denoise_full_file
    from .diffusion.schedule import DiffusionSchedule
    from .models.unet import ConditionalUNet
    from .training.trainer import load_checkpoint

    run_folder = Path(args.run)
    config = DenoiseConfig.load(run_folder / "run_config.json")
    dataset, date_split, pools = _build_context(config, run_folder / "split_config.json")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if getattr(config, "type", "base") == "ldm":
        from .models.ldm_wrapper import LDMUNetWrapper
        model = LDMUNetWrapper(model_channels=config.model.unet_channels).to(device)
    else:
        model = ConditionalUNet(
            base_channels=config.model.base_channels,
            channel_multipliers=tuple(config.model.channel_multipliers),
            blocks_per_stage=config.model.blocks_per_stage,
            attention_stages=tuple(config.model.attention_stages),
            dropout=config.model.dropout,
        ).to(device)
    checkpoint = load_checkpoint(run_folder / "checkpoints" / args.checkpoint, model)
    model.eval()
    print(f"checkpoint epoch={checkpoint.get('epoch')} val_loss={checkpoint.get('val_loss')}")

    schedule = DiffusionSchedule.create(config.diffusion.schedule, config.diffusion.timesteps)
    diffusion = GaussianDiffusion(schedule, device=device, loss_type=config.diffusion.loss_type)

    limit = config.sample.max_files
    rng = np.random.default_rng(config.seed)
    noise_bank = NoiseBank(dataset, pools["test"].noise, _noise_config(config))

    # 1) 합성 paired test — 정답 clean이 있으므로 오차를 직접 계산할 수 있다.
    clean_table = pools["test"].clean.head(limit)
    paired = {"clean": [], "noisy": [], "restored": [], "sample_id": []}
    for sample_id in clean_table["sample_id"]:
        clean_matrix = dataset.matrix(str(sample_id))
        noisy_matrix, clean_matrix = noise_bank.make_pair(clean_matrix, rng, lambda m: m)
        restored = denoise_full_file(
            diffusion, model, noisy_matrix,
            crop_width=config.data.crop_width,
            batch_size=config.sample.batch_size,
            steps=config.sample.steps,
            eta=config.sample.eta,
            device=device,
            seed=config.seed,
        )
        paired["clean"].append(clean_matrix)
        paired["noisy"].append(noisy_matrix)
        paired["restored"].append(restored)
        paired["sample_id"].append(str(sample_id))
        print(f"  paired {len(paired['sample_id'])}/{len(clean_table)}")

    np.savez_compressed(
        run_folder / "samples" / "paired_test.npz",
        clean=np.stack(paired["clean"]),
        noisy=np.stack(paired["noisy"]),
        restored=np.stack(paired["restored"]),
        sample_id=np.array(paired["sample_id"]),
    )

    # 2) 실제 Field PD — 정답이 없으므로 분포 지표로만 평가한다.
    real_table = pools["test"].real_noisy.head(limit)
    real = {"noisy": [], "restored": [], "sample_id": []}
    for sample_id in real_table["sample_id"]:
        noisy_matrix = dataset.matrix(str(sample_id))
        restored = denoise_full_file(
            diffusion, model, noisy_matrix,
            crop_width=config.data.crop_width,
            batch_size=config.sample.batch_size,
            steps=config.sample.steps,
            eta=config.sample.eta,
            device=device,
            seed=config.seed,
        )
        real["noisy"].append(noisy_matrix)
        real["restored"].append(restored)
        real["sample_id"].append(str(sample_id))
        print(f"  real {len(real['sample_id'])}/{len(real_table)}")

    if real["sample_id"]:
        np.savez_compressed(
            run_folder / "samples" / "real_noisy.npz",
            noisy=np.stack(real["noisy"]),
            restored=np.stack(real["restored"]),
            sample_id=np.array(real["sample_id"]),
        )

    write_json(
        run_folder / "samples" / "sample_config.json",
        {
            "checkpoint": args.checkpoint,
            "ddim_steps": config.sample.steps,
            "eta": config.sample.eta,
            "seed": config.seed,
            "paired_files": len(paired["sample_id"]),
            "real_files": len(real["sample_id"]),
            "device": str(device),
        },
    )
    write_manifest(run_folder)
    print(f"\n샘플 저장 완료: {run_folder / 'samples'}")


# ---------------------------------------------------------------- evaluate


def command_evaluate(args: argparse.Namespace) -> None:
    from .evaluation.metrics import distribution_shift, feature_matrix, paired_metrics
    from .evaluation.report import (
        prpd_heatmap_figure,
        profile_comparison_figure,
        write_html_report,
    )

    run_folder = Path(args.run)
    config = DenoiseConfig.load(run_folder / "run_config.json")
    dataset, date_split, pools = _build_context(config, run_folder / "split_config.json")

    metrics: dict = {"run": run_folder.name, "config_hash": config.config_hash()}
    figures = []

    paired_path = run_folder / "samples" / "paired_test.npz"
    if paired_path.is_file():
        with np.load(paired_path) as archive:
            clean, noisy, restored = archive["clean"], archive["noisy"], archive["restored"]
        rows = [
            paired_metrics(restored[index], clean[index], noisy[index])
            for index in range(len(clean))
        ]
        metrics["paired"] = {
            key: float(np.mean([row[key] for row in rows])) for key in rows[0]
        }
        metrics["paired"]["files"] = len(rows)
        figures.append(profile_comparison_figure(noisy[0], restored[0], clean[0]))
        figures.append(prpd_heatmap_figure(noisy[0], "합성 noisy (입력)"))
        figures.append(prpd_heatmap_figure(restored[0], "복원 결과"))
        figures.append(prpd_heatmap_figure(clean[0], "clean 정답"))

    real_path = run_folder / "samples" / "real_noisy.npz"
    if real_path.is_file():
        with np.load(real_path) as archive:
            real_noisy, real_restored = archive["noisy"], archive["restored"]
        reference_table = pools["train"].clean.head(args.reference_files)
        reference = feature_matrix(dataset.matrices(reference_table))
        metrics["distribution"] = distribution_shift(
            reference, feature_matrix(real_noisy), feature_matrix(real_restored)
        )
        metrics["distribution"]["reference_files"] = int(len(reference_table))
        metrics["distribution"]["evaluated_files"] = int(len(real_noisy))
        figures.append(profile_comparison_figure(real_noisy[0], real_restored[0]))

    write_json(run_folder / "metrics.json", metrics)
    write_html_report(
        run_folder / "evaluation_report.html",
        f"PRPD denoising 평가 — {run_folder.name}",
        metrics,
        figures,
        notes=[
            "paired 지표는 실측 노이즈를 합성해 만든 test 쌍 기준이다.",
            "distribution 지표는 정답이 없는 실제 Field PD를 clean 분포(Lab PD)와 비교한 것이다.",
            "복원 결과는 물리적 고장 판정이 아니라 노이즈 감소 정도를 나타낸다.",
        ],
    )
    write_manifest(run_folder)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\n리포트: {run_folder / 'evaluation_report.html'}")


# --------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prpd-diffusion", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="데이터·split·pool 점검 (torch 불필요)")
    prepare.add_argument("--config", type=Path, default=Path("Diffusion/configs/denoise_base.json"))
    prepare.add_argument("--cache", action="store_true", help="npz → npy memmap 캐시 생성")
    prepare.add_argument("--verify-manifest", action="store_true", help="입력 checksum 검증")
    prepare.set_defaults(func=command_prepare)

    train = subparsers.add_parser("train", help="조건부 diffusion denoiser 학습")
    train.add_argument("--config", type=Path, default=Path("Diffusion/configs/denoise_base.json"))
    train.add_argument("--cache", action="store_true", help="학습 전 memmap 캐시 생성")
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

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

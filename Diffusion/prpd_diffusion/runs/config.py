"""실행 설정. JSON으로 읽고 쓴다(기존 `run_config.json` 관례와 동일, 새 의존성 없음).

설정은 run 폴더에 그대로 저장되며, `config_hash`로 동일 설정 여부를 판별한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
from pathlib import Path


@dataclass
class DataConfig:
    ai_data_root: str = ""                       # artifacts/ai_data_YYYYMMDD_HHMMSS
    memmap_cache: str = "artifacts/repr_cache"   # npz → npy 캐시 위치
    crop_width: int = 256
    samples_per_file: int = 4                    # epoch당 파일에서 뽑는 crop 수
    clean_groups: tuple[str, ...] = ("Lab PD",)
    noise_groups: tuple[str, ...] = ("Lab Noise", "Field Noise")
    real_noisy_groups: tuple[str, ...] = ("Field PD",)
    clean_labels: tuple[str, ...] | None = None


@dataclass
class SplitConfig:
    ratios: dict = field(default_factory=lambda: {"train": 0.7, "val": 0.15, "test": 0.15})
    scope: str = "global"                        # 한 날짜는 하나의 split에만 속한다


@dataclass
class NoiseConfig:
    mode: str = "maximum"                        # maximum | additive | quadrature
    gain_min: float = 0.6
    gain_max: float = 1.4
    phase_roll: bool = True
    time_roll: bool = True
    dropout_probability: float = 0.05            # 일부 배치는 노이즈 없이 학습(항등 보존)


@dataclass
class ModelConfig:
    base_channels: int = 64
    channel_multipliers: tuple[int, ...] = (1, 2, 4)
    blocks_per_stage: int = 2
    attention_stages: tuple[int, ...] = (2,)
    dropout: float = 0.0
    embed_dim: int = 4
    ch_mult: tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    unet_channels: int = 128


@dataclass
class DiffusionConfig:
    schedule: str = "cosine"                     # cosine | linear
    timesteps: int = 1000
    loss_type: str = "l2"


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 16
    learning_rate: float = 1e-4
    ema_decay: float = 0.999
    grad_clip: float = 1.0
    num_workers: int = 0                         # Windows에서는 0이 안전하다
    max_steps: int = 0                           # 0이면 제한 없음


@dataclass
class SampleConfig:
    steps: int = 50                              # DDIM 스텝
    eta: float = 0.0
    batch_size: int = 8
    max_files: int = 32                          # 평가용 복원 파일 수 상한


_SECTIONS = {
    "data": DataConfig,
    "split": SplitConfig,
    "noise": NoiseConfig,
    "model": ModelConfig,
    "diffusion": DiffusionConfig,
    "train": TrainConfig,
    "sample": SampleConfig,
}


@dataclass
class DenoiseConfig:
    type: str = "base"
    seed: int = 42
    run_prefix: str = "denoise"
    notes: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "DenoiseConfig":
        kwargs: dict = {}
        for entry in fields(cls):
            if entry.name not in payload:
                continue
            value = payload[entry.name]
            section = _SECTIONS.get(entry.name)
            if section is None:
                kwargs[entry.name] = value
                continue
            names = {item.name for item in fields(section)}
            unknown = set(value) - names
            if unknown:
                raise ValueError(f"'{entry.name}' 설정에 알 수 없는 키: {sorted(unknown)}")
            kwargs[entry.name] = section(
                **{
                    key: tuple(item) if isinstance(item, list) else item
                    for key, item in value.items()
                }
            )
        return cls(**kwargs)

    @classmethod
    def load(cls, path: Path) -> "DenoiseConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: Path) -> Path:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return Path(path)

    def config_hash(self) -> str:
        """설정 내용의 짧은 해시. run_id 및 캐시 키에 사용한다."""
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

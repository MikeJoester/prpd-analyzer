"""ARDD 1D 트랙의 실행 설정.

2D baseline의 `runs.config.DenoiseConfig`와 별도 dataclass 집합을 쓴다. 같은 것을 공유하면
한쪽의 키 추가가 다른 쪽의 "알 수 없는 키" 검증을 흔들기 때문이다.
`split` / `diffusion` / `train` / `sample`은 의미가 완전히 같으므로 **그대로 재사용**한다.

논문 기법은 별도 알고리즘이 아니라 `components` 섹션의 on/off 옵션이다(`DIFFUSION.md` 18.5).
그래야 ablation이 config 조합만으로 만들어지고 어느 요소가 켜졌는지 `run_config.json`에 남는다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
from pathlib import Path

from ..runs.config import DiffusionConfig, SampleConfig, SplitConfig, TrainConfig

NOISE_KINDS = ("cin", "measured")
ENTROPY_SOURCES = ("condition", "x_t")
PADDING_MODES = ("circular", "zeros")
# `components.morph_attn.NORMALIZE_MODES`와 같아야 한다(테스트로 고정).
MORPH_NORMALIZE_MODES = ("none", "instance", "otsu")


@dataclass
class Data1dConfig:
    ai_data_root: str = ""                       # artifacts/ai_data_YYYYMMDD_HHMMSS
    memmap_cache: str = "artifacts/repr_cache"   # npz → npy 캐시 (실행별 하위 폴더로 분리됨)
    profile_cache: str = "artifacts/profile_cache"
    representation: str = "profile_v1"           # profile_v1 | phase_row_v1(미구현)
    use_profile_cache: bool = True               # False면 매 스텝 raw에서 새로 합성(느림, 다양성 최대)
    realizations_per_file: int = 8               # 캐시에 저장할 파일당 노이즈 실현 수
    samples_per_file: int = 1                    # epoch당 파일에서 뽑는 샘플 수
    clean_groups: tuple[str, ...] = ("Lab PD",)
    noise_groups: tuple[str, ...] = ()           # noise.kind="measured"일 때만 사용
    real_noisy_groups: tuple[str, ...] = ("Field PD",)
    clean_labels: tuple[str, ...] | None = None


@dataclass
class Noise1dConfig:
    """합성 노이즈의 종류와 mixing. CIN 성분 자체는 `cin` 섹션에 있다."""

    kind: str = "cin"                            # cin | measured
    mode: str = "maximum"                        # maximum | additive | quadrature
    gain_min: float = 0.6
    gain_max: float = 1.4
    dropout_probability: float = 0.05            # 이 확률로 노이즈 없이 학습(항등 보존)


@dataclass
class CinConfig:
    """`ARDD-2025` 로버스트니스 실험의 Composite Industrial Noise 성분."""

    white_sigma: float = 6.0
    pink_sigma: float = 4.0
    harmonic_amplitudes: tuple[float, ...] = (10.0, 4.0, 2.0)
    harmonic_orders: tuple[int, ...] = (1, 3, 5)
    harmonic_axis: str = "phase"                 # 128 bin = AC 1주기 → 전원주파수는 위상축
    pink_axis: str = "time"                      # 3600 열 = 사이클 인덱스 → 1/f 드리프트
    randomize_harmonic_phase: bool = True
    target_snr_db: float | None = None           # 지정 시 clean 대비 SNR로 진폭 재조정


@dataclass
class Model1dConfig:
    base_channels: int = 64
    channel_multipliers: tuple[int, ...] = (1, 2, 4)
    blocks_per_stage: int = 2
    attention_stages: tuple[int, ...] = (2,)
    dropout: float = 0.0
    padding_mode: str = "circular"               # 위상축은 순환 구조다


@dataclass
class ComponentsConfig:
    """논문 구성요소 on/off (`DIFFUSION.md` 18.4의 A2·A3).

    전부 False면 forward는 2D baseline과 수학적으로 동일한 1D U-Net이다.
    """

    ardd_resid: bool = False                     # A2 — 엔트로피 기반 적응형 잔차, Eq.(5)~(7)
    ardd_alpha: float = 0.5
    ardd_entropy_window: int = 9
    ardd_entropy_bins: int = 16
    ardd_entropy_source: str = "condition"       # condition | x_t
    ardd_invert_entropy: bool = False            # 논문 본문 해석(엔트로피↑ → 잔차↓)을 시험할 때
    morph_attn: bool = False                     # A3 — 형태학적 gradient attention, Eq.(9)~(15)
    morph_scales: tuple[int, ...] = (1, 3, 7)    # 위상축 길이 128 기준(논문의 3600과 다름)
    morph_normalize: str = "none"                # none(논문 그대로) | instance | otsu

    def to_params(self) -> dict:
        """`run_config.json`의 `algorithm.components`에 남길 켜진 요소만 추린다."""
        params: dict = {}
        if self.ardd_resid:
            params["ardd_resid"] = {
                "alpha": self.ardd_alpha,
                "entropy_window": self.ardd_entropy_window,
                "entropy_bins": self.ardd_entropy_bins,
                "entropy_source": self.ardd_entropy_source,
                "invert_entropy": self.ardd_invert_entropy,
            }
        if self.morph_attn:
            params["morph_attn"] = {
                "scales": list(self.morph_scales),
                "normalize": self.morph_normalize,
            }
        return params


_SECTIONS = {
    "data": Data1dConfig,
    "split": SplitConfig,
    "noise": Noise1dConfig,
    "cin": CinConfig,
    "model": Model1dConfig,
    "components": ComponentsConfig,
    "diffusion": DiffusionConfig,
    "train": TrainConfig,
    "sample": SampleConfig,
}


@dataclass
class Ardd1dConfig:
    seed: int = 42
    # 모델 초기화·배치 순서만 바꾸는 seed. None이면 `seed`와 같다.
    # `seed`를 바꾸면 날짜 분할과 합성 노이즈 실현까지 바뀌어 후보 간 비교가 깨진다
    # (`DIFFUSION.md` 18.1의 "같은 분할", "같은 합성 노이즈" 규칙).
    # 18.1이 요구하는 "seed 3개 이상 반복"은 이 값만 바꿔서 수행한다.
    train_seed: int | None = None
    run_prefix: str = "ardd1d"
    notes: str = ""
    data: Data1dConfig = field(default_factory=Data1dConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    noise: Noise1dConfig = field(default_factory=Noise1dConfig)
    cin: CinConfig = field(default_factory=CinConfig)
    model: Model1dConfig = field(default_factory=Model1dConfig)
    components: ComponentsConfig = field(default_factory=ComponentsConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)

    # ------------------------------------------------------------ 검증

    def validate(self) -> None:
        from ..data.profile import resolve_kind

        resolve_kind(self.data.representation)
        if self.noise.kind not in NOISE_KINDS:
            raise ValueError(f"알 수 없는 noise.kind: {self.noise.kind} (가능: {NOISE_KINDS})")
        if self.noise.kind == "measured" and not self.data.noise_groups:
            raise ValueError(
                "noise.kind='measured'인데 data.noise_groups가 비어 있습니다. "
                "노이즈 그룹이 있는 ai_data 실행을 지정하거나 kind를 'cin'으로 두세요."
            )
        if not 0 < self.noise.gain_min <= self.noise.gain_max:
            raise ValueError("noise gain 범위가 잘못되었습니다.")
        if not 0.0 <= self.noise.dropout_probability <= 1.0:
            raise ValueError("noise.dropout_probability는 0~1이어야 합니다.")
        if self.data.realizations_per_file < 1:
            raise ValueError("realizations_per_file은 1 이상이어야 합니다.")
        if self.components.ardd_entropy_source not in ENTROPY_SOURCES:
            raise ValueError(
                f"알 수 없는 ardd_entropy_source: {self.components.ardd_entropy_source}"
            )
        # torch 없이도 설정을 검증할 수 있어야 하므로 components 모듈을 import 하지 않는다.
        # 값 목록이 `components.morph_attn.NORMALIZE_MODES`와 같은지는 테스트로 고정한다.
        if self.components.morph_normalize not in MORPH_NORMALIZE_MODES:
            raise ValueError(
                f"알 수 없는 morph_normalize: {self.components.morph_normalize} "
                f"(가능: {MORPH_NORMALIZE_MODES})"
            )
        if self.components.ardd_entropy_window % 2 == 0:
            raise ValueError("ardd_entropy_window는 홀수여야 합니다.")
        if self.model.padding_mode not in PADDING_MODES:
            raise ValueError(f"알 수 없는 padding_mode: {self.model.padding_mode}")
        self.cin_config().validate()

    @property
    def effective_train_seed(self) -> int:
        return self.seed if self.train_seed is None else int(self.train_seed)

    def cin_config(self):
        from ..data.cin_noise import CinNoiseConfig

        return CinNoiseConfig(
            white_sigma=self.cin.white_sigma,
            pink_sigma=self.cin.pink_sigma,
            harmonic_amplitudes=tuple(self.cin.harmonic_amplitudes),
            harmonic_orders=tuple(int(order) for order in self.cin.harmonic_orders),
            harmonic_axis=self.cin.harmonic_axis,
            pink_axis=self.cin.pink_axis,
            randomize_harmonic_phase=self.cin.randomize_harmonic_phase,
            target_snr_db=self.cin.target_snr_db,
        )

    def algorithm_record(self) -> dict:
        """`run_config.json`에 남길 알고리즘 식별 정보 (`DIFFUSION.md` 18.1)."""
        return {
            "kind": "ardd1d",
            "representation": self.data.representation,
            "noise_model": self.noise.kind,
            "schedule": self.diffusion.schedule,
            "components": self.components.to_params(),
            "seed": self.seed,
            "train_seed": self.effective_train_seed,
        }

    # ------------------------------------------------------------ 직렬화

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "Ardd1dConfig":
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
    def load(cls, path: Path) -> "Ardd1dConfig":
        config = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        config.validate()
        return config

    def save(self, path: Path) -> Path:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return Path(path)

    def config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

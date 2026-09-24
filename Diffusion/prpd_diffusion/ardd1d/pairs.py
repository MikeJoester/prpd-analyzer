"""ARDD 1D 트랙의 paired 데이터 생성. numpy만 사용(torch 불필요).

    clean x0 = Lab PD raw → profile
    noise n  = CIN 파라메트릭 생성  (또는 실측 pool)
    noisy y  = profile( mix(x0_raw, n_raw) )

**노이즈는 raw 영역에서 섞고 그 다음에 profile을 뽑는다.** 반대로 profile에 직접 노이즈를
더하면 "위상축을 따라 흐르는 1/f 잡음" 같은 물리적으로 없는 것을 만들게 된다. 시간축 노이즈가
mean/max profile에 어떻게 반영되는지가 이 트랙의 실제 관심사이기도 하다.

프로파일 캐시: clean profile은 결정적이므로 한 번만 계산한다. noisy는 노이즈 실현마다 다르므로
파일당 `realizations`개를 미리 만들어 둔다. `(N, M, 2, 128) float32`라 1,062 파일 × 8 실현이
8.7 MB밖에 안 되고, 학습 루프에서 raw memmap을 전혀 읽지 않아도 된다.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from ..data.cin_noise import CinNoiseConfig, sample_cin_matrix
from ..data.noise_model import augment_noise, mix, NoiseAugmentConfig
from ..data.profile import PHASE_BINS, PROFILE_CHANNELS, profile_from_matrix

CACHE_VERSION = "profile_pairs_v1"


@dataclass(frozen=True)
class PairFactory:
    """`(noisy_profile, clean_profile)` 한 쌍을 만든다. 값 범위 `[0, 255]`."""

    kind: str                      # cin | measured
    mode: str                      # maximum | additive | quadrature
    gain_min: float
    gain_max: float
    dropout_probability: float
    cin: CinNoiseConfig | None = None
    noise_bank: object | None = None  # data.noise_model.NoiseBank (measured 모드)

    def __post_init__(self) -> None:
        if self.kind == "cin" and self.cin is None:
            raise ValueError("kind='cin'인데 CinNoiseConfig가 없습니다.")
        if self.kind == "measured" and self.noise_bank is None:
            raise ValueError("kind='measured'인데 NoiseBank가 없습니다.")

    def noise_matrix(self, rng: np.random.Generator, clean: np.ndarray) -> np.ndarray:
        if self.kind == "cin":
            noise = sample_cin_matrix(rng, self.cin, clean.shape, clean=clean)
        else:
            noise = augment_noise(
                self.noise_bank.sample_matrix(rng),
                NoiseAugmentConfig(
                    mode=self.mode, gain_min=1.0, gain_max=1.0
                ),  # gain은 아래에서 공통 적용
                rng,
            )
        gain = float(rng.uniform(self.gain_min, self.gain_max))
        if gain != 1.0:
            noise = np.clip(noise.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        return noise

    def make_pair(
        self, clean_matrix: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        clean_profile = profile_from_matrix(clean_matrix)
        if rng.random() < self.dropout_probability:
            # 일부 샘플은 노이즈 없이 학습해 항등 사상을 보존한다.
            return clean_profile.copy(), clean_profile
        noisy_matrix = mix(clean_matrix, self.noise_matrix(rng, clean_matrix), self.mode)
        return profile_from_matrix(noisy_matrix), clean_profile


def factory_from_config(config, noise_bank=None) -> PairFactory:
    """`Ardd1dConfig` → `PairFactory`."""
    return PairFactory(
        kind=config.noise.kind,
        mode=config.noise.mode,
        gain_min=config.noise.gain_min,
        gain_max=config.noise.gain_max,
        dropout_probability=config.noise.dropout_probability,
        cin=config.cin_config() if config.noise.kind == "cin" else None,
        noise_bank=noise_bank,
    )


# ------------------------------------------------------------------ 캐시


def noise_signature(config) -> str:
    """노이즈 합성 결과를 바꾸는 설정만 모은 해시. 캐시 키에 쓴다.

    학습 하이퍼파라미터(epochs, lr 등)는 pair 내용에 영향을 주지 않으므로 제외한다.
    전체 `config_hash`를 쓰면 lr만 바꿔도 캐시를 다시 만들게 된다.
    """
    payload = {
        "version": CACHE_VERSION,
        "representation": config.data.representation,
        "realizations": config.data.realizations_per_file,
        "seed": config.seed,
        "noise": {
            "kind": config.noise.kind,
            "mode": config.noise.mode,
            "gain_min": config.noise.gain_min,
            "gain_max": config.noise.gain_max,
            "dropout_probability": config.noise.dropout_probability,
        },
        "cin": config.cin_config().to_dict() if config.noise.kind == "cin" else None,
        "clean_groups": list(config.data.clean_groups),
        "noise_groups": list(config.data.noise_groups),
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def cache_folder(cache_root: Path, dataset, config) -> Path:
    """`{cache_root}/{ai_data 실행 이름}/{노이즈 설정 해시}`.

    실행 이름을 넣지 않으면 dataset을 바꿔도 이전 실행의 profile을 그대로 재사용하게 된다
    (`DIFFUSION.md` 2.2절 캐시 규칙).
    """
    return Path(cache_root) / dataset.root.name / noise_signature(config)


def build_profile_pairs(
    dataset,
    table,
    factory: PairFactory,
    realizations: int,
    seed: int,
    progress=None,
) -> dict[str, np.ndarray]:
    """clean profile과 노이즈 실현 M개의 noisy profile을 만든다.

    Returns:
        `{"sample_ids": (N,), "labels": (N,), "clean": (N, 2, 128), "noisy": (N, M, 2, 128)}`
    """
    sample_ids = [str(value) for value in table["sample_id"]]
    labels = [str(value) for value in table["label"]]
    count = len(sample_ids)
    clean = np.zeros((count, PROFILE_CHANNELS, PHASE_BINS), dtype=np.float32)
    noisy = np.zeros((count, realizations, PROFILE_CHANNELS, PHASE_BINS), dtype=np.float32)

    iterator = range(count)
    if progress is not None:
        iterator = progress(iterator)
    for index in iterator:
        matrix = dataset.matrix(sample_ids[index])
        clean[index] = profile_from_matrix(matrix)
        for realization in range(realizations):
            rng = np.random.default_rng((seed, index, realization))
            noisy[index, realization] = factory.make_pair(matrix, rng)[0]

    return {
        "sample_ids": np.array(sample_ids, dtype=object),
        "labels": np.array(labels, dtype=object),
        "clean": clean,
        "noisy": noisy,
    }


def save_profile_pairs(path: Path, payload: dict[str, np.ndarray]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        sample_ids=payload["sample_ids"].astype(str),
        labels=payload["labels"].astype(str),
        clean=payload["clean"],
        noisy=payload["noisy"],
    )
    return path


def load_profile_pairs(path: Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: archive[key] for key in ("sample_ids", "labels", "clean", "noisy")}

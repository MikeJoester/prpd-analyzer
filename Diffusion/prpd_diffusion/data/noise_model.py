"""측정된 노이즈를 PD 데이터에 합성해 paired 학습 데이터를 만든다.

노이즈 제거 모델은 (노이즈 섞인 입력, 깨끗한 목표) 쌍이 필요하지만 현장 데이터에는
같은 시점의 clean/noisy 쌍이 존재하지 않는다. 대신 다음을 사용한다.

    clean  x0 = Lab PD raw matrix              (상대적으로 깨끗한 PD 패턴)
    noise  n  = Lab Noise / Field Noise matrix (실측 노이즈, 합성 노이즈가 아님)
    noisy  y  = mix(x0, n)                     (아래 mixing operator)

전제와 한계 — 새 세션에서 반드시 재검토할 것:

* Lab PD도 완전한 clean이 아니라 자체 노이즈 바닥을 포함한다. 따라서 학습 목표는
  "모든 노이즈 제거"가 아니라 "Field 수준 노이즈를 Lab 수준까지 낮추기"이다.
* PRPS 한 bin 값은 해당 위상 window에서 검출된 피크 진폭이므로, PD 펄스와 노이즈가
  같은 bin에 들어오면 측정값은 대체로 둘 중 큰 값에 가깝다 → 기본 mixing은 `maximum`.
  검출기 특성이 다르면 `additive`(clip) 또는 `quadrature`(RMS)로 교체해 비교한다.
* 이 가정의 타당성은 합성 noisy와 실제 Field PD의 256 feature 분포를 비교해 점검한다
  (`evaluation.metrics.frechet_distance`). 분포가 크게 다르면 mixing 설정을 재조정한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIX_MODES = ("maximum", "additive", "quadrature")


@dataclass(frozen=True)
class NoiseAugmentConfig:
    """노이즈 pool에서 뽑은 샘플에 적용할 무작위 변형."""

    mode: str = "maximum"
    gain_min: float = 0.6
    gain_max: float = 1.4
    phase_roll: bool = True          # 위상축 순환 이동 (노이즈의 위상 고정 편향 완화)
    time_roll: bool = True           # 시간축 순환 이동
    dropout_probability: float = 0.0  # 이 확률로 노이즈를 섞지 않음(clean 그대로 학습)

    def validate(self) -> None:
        if self.mode not in MIX_MODES:
            raise ValueError(f"알 수 없는 mixing mode: {self.mode} (가능: {MIX_MODES})")
        if not 0 < self.gain_min <= self.gain_max:
            raise ValueError("gain 범위가 잘못되었습니다.")
        if not 0.0 <= self.dropout_probability <= 1.0:
            raise ValueError("dropout_probability는 0~1이어야 합니다.")


def mix(clean: np.ndarray, noise: np.ndarray, mode: str = "maximum") -> np.ndarray:
    """clean과 noise를 합성해 uint8 noisy 행렬을 만든다.

    계산은 uint16으로 올린 뒤 255에서 clip한다(uint8 wrap-around 방지).
    """
    if clean.shape != noise.shape:
        raise ValueError(f"shape 불일치: clean {clean.shape} vs noise {noise.shape}")
    left = clean.astype(np.uint16)
    right = noise.astype(np.uint16)
    if mode == "maximum":
        mixed = np.maximum(left, right)
    elif mode == "additive":
        mixed = left + right
    elif mode == "quadrature":
        mixed = np.sqrt(left.astype(np.float32) ** 2 + right.astype(np.float32) ** 2)
    else:
        raise ValueError(f"알 수 없는 mixing mode: {mode}")
    return np.clip(mixed, 0, 255).astype(np.uint8)


def augment_noise(noise: np.ndarray, config: NoiseAugmentConfig, rng: np.random.Generator) -> np.ndarray:
    """gain 조정과 순환 이동을 적용한 노이즈 행렬을 만든다."""
    result = noise
    if config.phase_roll:
        result = np.roll(result, int(rng.integers(0, result.shape[0])), axis=0)
    if config.time_roll:
        result = np.roll(result, int(rng.integers(0, result.shape[1])), axis=1)
    gain = float(rng.uniform(config.gain_min, config.gain_max))
    if gain != 1.0:
        result = np.clip(result.astype(np.float32) * gain, 0, 255)
    return result.astype(np.uint8)


class NoiseBank:
    """split별 노이즈 pool. 학습 중 crop 단위로 노이즈를 공급한다.

    노이즈 pool도 날짜 단위 split을 따른다. train에서 본 노이즈 날짜가 val/test에
    다시 등장하면 성능이 과대평가된다.
    """

    def __init__(
        self,
        dataset,  # contract.AiDataset (순환 import를 피하려 타입 주석 생략)
        table,    # pandas.DataFrame — 이 pool에 쓸 노이즈 metadata 부분집합
        config: NoiseAugmentConfig | None = None,
    ) -> None:
        if len(table) == 0:
            raise ValueError("노이즈 pool이 비어 있습니다. split 또는 그룹 선택을 확인하세요.")
        self._dataset = dataset
        self._sample_ids = tuple(str(value) for value in table["sample_id"])
        self._groups = tuple(sorted(set(str(value) for value in table["group"])))
        self._dates = tuple(sorted(set(str(value) for value in table["date"])))
        self.config = config or NoiseAugmentConfig()
        self.config.validate()

    def __len__(self) -> int:
        return len(self._sample_ids)

    @property
    def groups(self) -> tuple[str, ...]:
        return self._groups

    @property
    def dates(self) -> tuple[str, ...]:
        return self._dates

    def sample_matrix(self, rng: np.random.Generator) -> np.ndarray:
        """pool에서 노이즈 파일 하나를 균등 추출한다."""
        index = int(rng.integers(0, len(self._sample_ids)))
        return self._dataset.matrix(self._sample_ids[index])

    def make_pair(
        self,
        clean_crop: np.ndarray,
        rng: np.random.Generator,
        crop_fn,
    ) -> tuple[np.ndarray, np.ndarray]:
        """clean crop에 대응하는 `(noisy, clean)` uint8 쌍을 만든다.

        Args:
            clean_crop: `(128, width)` clean 행렬 crop.
            crop_fn: 전체 노이즈 행렬에서 같은 width의 crop을 뽑는 함수.
        """
        if rng.random() < self.config.dropout_probability:
            return clean_crop.copy(), clean_crop
        noise_crop = crop_fn(self.sample_matrix(rng))
        noise_crop = augment_noise(noise_crop, self.config, rng)
        return mix(clean_crop, noise_crop, self.config.mode), clean_crop

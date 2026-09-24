"""ARDD 1D 트랙의 paired 데이터 생성 테스트. torch 불필요.

여기서 확인하는 것은 "노이즈가 raw 영역에서 섞이고 그 다음에 profile이 뽑히는가"와
"같은 seed에서 완전히 재현되는가"다. 둘 중 하나라도 깨지면 ablation 비교가 성립하지 않는다.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Diffusion.prpd_diffusion.ardd1d.config import Ardd1dConfig
from Diffusion.prpd_diffusion.ardd1d.metrics import channel_errors, improved_channels, paired_profile_metrics
from Diffusion.prpd_diffusion.ardd1d.pairs import (
    PairFactory,
    build_profile_pairs,
    cache_folder,
    factory_from_config,
    load_profile_pairs,
    noise_signature,
    save_profile_pairs,
)
from Diffusion.prpd_diffusion.data.cin_noise import CinNoiseConfig
from Diffusion.prpd_diffusion.data.profile import profile_from_matrix

ROOT = Path(__file__).resolve().parent.parent


class _FakeDataset:
    """작은 대체 dataset. 실제 artifacts 없이 pair 로직만 검증한다."""

    def __init__(self, count: int = 4, time_samples: int = 300) -> None:
        rng = np.random.default_rng(0)
        self._matrices = {
            f"s{index}": rng.integers(0, 120, size=(128, time_samples), dtype=np.uint8)
            for index in range(count)
        }
        self.root = Path("artifacts/ai_data_TEST")

    def matrix(self, sample_id: str) -> np.ndarray:
        return self._matrices[sample_id]

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sample_id": list(self._matrices),
                "label": ["Void"] * len(self._matrices),
            }
        )


def _factory(**overrides) -> PairFactory:
    defaults = dict(
        kind="cin",
        mode="maximum",
        gain_min=1.0,
        gain_max=1.0,
        dropout_probability=0.0,
        cin=CinNoiseConfig(),
    )
    defaults.update(overrides)
    return PairFactory(**defaults)


def test_noisy_profile_is_at_least_clean_under_maximum_mixing():
    """`maximum` 합성이므로 noisy의 max profile은 clean보다 작을 수 없다."""
    dataset = _FakeDataset()
    matrix = dataset.matrix("s0")
    noisy, clean = _factory().make_pair(matrix, np.random.default_rng(0))
    assert np.all(noisy[1] >= clean[1] - 1e-4)   # max 채널
    assert np.all(noisy[0] >= clean[0] - 1e-4)   # mean 채널도 bin별 max이므로 단조
    assert np.array_equal(clean, profile_from_matrix(matrix))


def test_pair_is_reproducible_for_same_seed():
    matrix = _FakeDataset().matrix("s1")
    first = _factory().make_pair(matrix, np.random.default_rng(3))[0]
    second = _factory().make_pair(matrix, np.random.default_rng(3))[0]
    assert np.array_equal(first, second)
    other = _factory().make_pair(matrix, np.random.default_rng(4))[0]
    assert not np.array_equal(first, other)


def test_dropout_returns_clean_untouched():
    matrix = _FakeDataset().matrix("s2")
    noisy, clean = _factory(dropout_probability=1.0).make_pair(matrix, np.random.default_rng(0))
    assert np.array_equal(noisy, clean)


def test_gain_range_changes_noise_strength():
    matrix = _FakeDataset().matrix("s0")
    weak = _factory(gain_min=0.1, gain_max=0.1).make_pair(matrix, np.random.default_rng(5))[0]
    strong = _factory(gain_min=3.0, gain_max=3.0).make_pair(matrix, np.random.default_rng(5))[0]
    assert strong[1].mean() >= weak[1].mean()


def test_build_profile_pairs_shapes_and_determinism():
    dataset = _FakeDataset(count=3)
    payload = build_profile_pairs(dataset, dataset.table(), _factory(), realizations=4, seed=7)
    assert payload["clean"].shape == (3, 2, 128)
    assert payload["noisy"].shape == (3, 4, 2, 128)
    assert list(payload["sample_ids"]) == ["s0", "s1", "s2"]
    again = build_profile_pairs(dataset, dataset.table(), _factory(), realizations=4, seed=7)
    assert np.array_equal(payload["noisy"], again["noisy"])


def test_realizations_differ_from_each_other():
    dataset = _FakeDataset(count=1)
    payload = build_profile_pairs(dataset, dataset.table(), _factory(), realizations=3, seed=1)
    assert not np.array_equal(payload["noisy"][0, 0], payload["noisy"][0, 1])


def test_profile_pairs_survive_save_and_load():
    dataset = _FakeDataset(count=2)
    payload = build_profile_pairs(dataset, dataset.table(), _factory(), realizations=2, seed=2)
    with tempfile.TemporaryDirectory() as folder:
        path = save_profile_pairs(Path(folder) / "train.npz", payload)
        restored = load_profile_pairs(path)
    assert np.array_equal(payload["clean"], restored["clean"])
    assert np.array_equal(payload["noisy"], restored["noisy"])
    assert list(restored["sample_ids"]) == list(payload["sample_ids"])


# ------------------------------------------------------------------ 캐시 키


def test_cache_key_depends_on_noise_settings_but_not_on_training_hyperparameters():
    config = Ardd1dConfig.load(ROOT / "configs" / "ardd1d_base.json")
    baseline = noise_signature(config)

    config.train.learning_rate = 0.01  # pair 내용과 무관
    assert noise_signature(config) == baseline

    config.cin.white_sigma = 99.0      # pair 내용이 바뀐다
    assert noise_signature(config) != baseline


def test_cache_folder_separates_ai_data_runs():
    config = Ardd1dConfig.load(ROOT / "configs" / "ardd1d_base.json")
    first = cache_folder(Path("artifacts/profile_cache"), _FakeDataset(), config)

    class _Other(_FakeDataset):
        def __init__(self):
            super().__init__()
            self.root = Path("artifacts/ai_data_OTHER")

    second = cache_folder(Path("artifacts/profile_cache"), _Other(), config)
    assert first != second
    assert first.parent.name == "ai_data_TEST"
    assert second.parent.name == "ai_data_OTHER"


def test_factory_from_config_uses_cin_when_selected():
    config = Ardd1dConfig.load(ROOT / "configs" / "ardd1d_base.json")
    factory = factory_from_config(config)
    assert factory.kind == "cin"
    assert factory.cin is not None
    assert factory.cin.harmonic_axis == "phase"


def test_measured_kind_requires_a_noise_bank():
    try:
        PairFactory(
            kind="measured", mode="maximum", gain_min=1.0, gain_max=1.0, dropout_probability=0.0
        )
    except ValueError:
        return
    raise AssertionError("measured 모드에서 NoiseBank 없이 만들면 실패해야 합니다.")


# ------------------------------------------------------------------ 지표


def test_channel_errors_are_zero_for_identical_profiles():
    profile = np.random.default_rng(0).uniform(0, 255, size=(3, 2, 128))
    metrics = channel_errors(profile, profile)
    assert metrics["mean_profile_mae"] == 0.0
    assert metrics["max_profile_psnr"] == float("inf")


def test_paired_metrics_report_channels_separately():
    rng = np.random.default_rng(1)
    clean = rng.uniform(0, 200, size=(4, 2, 128))
    noisy = clean + rng.uniform(5, 15, size=clean.shape)
    restored = clean + rng.uniform(0, 3, size=clean.shape)
    metrics = paired_profile_metrics(restored, clean, noisy)
    for channel in ("mean", "max"):
        assert metrics[f"{channel}_profile_mae"] < metrics[f"baseline_{channel}_profile_mae"]
        assert metrics[f"{channel}_profile_mae_improvement"] > 0
    assert set(improved_channels(metrics)) == {"mean", "max"}


def test_improved_channels_reports_only_the_channel_that_improved():
    clean = np.zeros((2, 2, 128))
    noisy = np.stack([np.full((2, 128), 10.0)] * 2)
    restored = noisy.copy()
    restored[:, 1, :] = 1.0  # max 채널만 개선
    metrics = paired_profile_metrics(restored, clean, noisy)
    assert improved_channels(metrics) == ["max"]


def test_metrics_reject_shape_mismatch():
    try:
        channel_errors(np.zeros((2, 2, 128)), np.zeros((3, 2, 128)))
    except ValueError:
        return
    raise AssertionError("shape 불일치를 거부해야 합니다.")

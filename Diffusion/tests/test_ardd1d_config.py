"""ARDD 1D 트랙 설정 테스트. torch 불필요."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Diffusion.prpd_diffusion.ardd1d.config import Ardd1dConfig

ROOT = Path(__file__).resolve().parent.parent
SHIPPED = ("ardd1d_base.json", "ardd1d_ardd.json", "ardd1d_smoke.json")


def test_shipped_configs_load_and_validate():
    for name in SHIPPED:
        config = Ardd1dConfig.load(ROOT / "configs" / name)
        assert config.data.ai_data_root
        assert config.data.representation == "profile_v1"
        config.validate()


def test_base_is_component_free_and_ardd_turns_everything_on():
    base = Ardd1dConfig.load(ROOT / "configs" / "ardd1d_base.json")
    ardd = Ardd1dConfig.load(ROOT / "configs" / "ardd1d_ardd.json")
    assert base.components.to_params() == {}
    assert set(ardd.components.to_params()) == {"ardd_resid", "morph_attn"}
    assert base.diffusion.schedule == "cosine"
    assert ardd.diffusion.schedule == "linear"  # A1


def test_ablation_pair_differs_only_in_algorithm_axes():
    """E1과 E5는 데이터·분할·노이즈가 같아야 비교가 성립한다(DIFFUSION.md 18.1)."""
    base = Ardd1dConfig.load(ROOT / "configs" / "ardd1d_base.json").to_dict()
    ardd = Ardd1dConfig.load(ROOT / "configs" / "ardd1d_ardd.json").to_dict()
    for section in ("data", "split", "noise", "cin", "train", "sample"):
        if section == "data":
            # run 이름만 다르고 데이터 관련 키는 모두 같아야 한다.
            assert base[section] == ardd[section]
        else:
            assert base[section] == ardd[section], section
    assert base["seed"] == ardd["seed"]


def test_roundtrip_preserves_content_and_hash(tmp_path=None):
    import tempfile

    config = Ardd1dConfig.load(ROOT / "configs" / "ardd1d_ardd.json")
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "roundtrip.json"
        config.save(path)
        restored = Ardd1dConfig.load(path)
    assert restored.to_dict() == config.to_dict()
    assert restored.config_hash() == config.config_hash()


def test_lists_are_restored_as_tuples():
    config = Ardd1dConfig.from_dict({"model": {"channel_multipliers": [1, 2, 4]}})
    assert config.model.channel_multipliers == (1, 2, 4)
    assert isinstance(config.components.morph_scales, tuple)


def test_unknown_key_is_rejected():
    for payload in (
        {"data": {"crop_width": 256}},          # 2D 트랙 키 — 이 트랙에는 없다
        {"components": {"morph_use_otsu": True}},
        {"cin": {"brown_sigma": 1.0}},
    ):
        try:
            Ardd1dConfig.from_dict(payload)
        except ValueError:
            continue
        raise AssertionError(f"알 수 없는 키를 거부해야 합니다: {payload}")


def test_hash_changes_when_a_component_is_toggled():
    base = Ardd1dConfig.load(ROOT / "configs" / "ardd1d_base.json")
    toggled = Ardd1dConfig.load(ROOT / "configs" / "ardd1d_base.json")
    toggled.components.morph_attn = True
    assert base.config_hash() != toggled.config_hash()


def test_measured_noise_requires_noise_groups():
    config = Ardd1dConfig.from_dict({"noise": {"kind": "measured"}, "data": {"noise_groups": []}})
    try:
        config.validate()
    except ValueError as error:
        assert "noise_groups" in str(error)
        return
    raise AssertionError("measured 모드에서 빈 noise_groups를 거부해야 합니다.")


def test_validate_rejects_bad_values():
    for payload in (
        {"noise": {"kind": "gaussian"}},
        {"noise": {"gain_min": 1.5, "gain_max": 0.5}},
        {"model": {"padding_mode": "reflect"}},
        {"components": {"ardd_entropy_source": "middle"}},
        {"components": {"ardd_entropy_window": 8}},
        {"components": {"morph_normalize": "softmax"}},
        {"data": {"realizations_per_file": 0}},
        {"data": {"representation": "phase_row_v1"}},  # 아직 미구현
    ):
        try:
            Ardd1dConfig.from_dict(payload).validate()
        except (ValueError, NotImplementedError):
            continue
        raise AssertionError(f"잘못된 값을 거부해야 합니다: {payload}")


def test_algorithm_record_names_the_track_and_components():
    config = Ardd1dConfig.load(ROOT / "configs" / "ardd1d_ardd.json")
    record = config.algorithm_record()
    assert record["kind"] == "ardd1d"
    assert record["noise_model"] == "cin"
    assert record["schedule"] == "linear"
    assert record["components"]["morph_attn"]["scales"] == [1, 3, 7]

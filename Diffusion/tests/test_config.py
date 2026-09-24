from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from Diffusion.prpd_diffusion.runs.config import DenoiseConfig

ROOT = Path(__file__).resolve().parent.parent


def test_shipped_configs_load():
    for name in ("denoise_base.json", "denoise_smoke.json"):
        path = ROOT / "configs" / name
        assert path.is_file(), f"설정 파일 없음: {path}"
        config = DenoiseConfig.load(path)
        assert config.data.crop_width > 0
        assert config.diffusion.timesteps >= 2


def test_roundtrip_preserves_values():
    config = DenoiseConfig.load(ROOT / "configs" / "denoise_base.json")
    with tempfile.TemporaryDirectory() as folder:
        path = config.save(Path(folder) / "run_config.json")
        restored = DenoiseConfig.load(path)
    assert restored.to_dict() == config.to_dict()
    assert restored.config_hash() == config.config_hash()


def test_unknown_key_is_rejected():
    payload = json.loads((ROOT / "configs" / "denoise_base.json").read_text(encoding="utf-8"))
    payload["train"]["learning_rat"] = 0.1        # 오타를 조용히 무시하면 안 된다
    try:
        DenoiseConfig.from_dict(payload)
    except ValueError:
        return
    raise AssertionError("알 수 없는 설정 키는 ValueError여야 합니다.")


def test_config_hash_changes_with_content():
    config = DenoiseConfig.load(ROOT / "configs" / "denoise_base.json")
    before = config.config_hash()
    config.train.learning_rate *= 2
    assert config.config_hash() != before


def test_lists_are_restored_as_tuples():
    config = DenoiseConfig.load(ROOT / "configs" / "denoise_base.json")
    assert isinstance(config.model.channel_multipliers, tuple)
    assert isinstance(config.data.clean_groups, tuple)

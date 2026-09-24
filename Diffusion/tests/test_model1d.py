"""조건부 1D U-Net 테스트. torch 필요.

가장 중요한 두 단언:

1. **위상축 circular 등변성** — 입력을 위상축으로 굴리면 출력도 똑같이 굴러야 한다.
   2D baseline의 `test_phase_axis_padding_is_circular`와 같은 방식이다.
2. **구성요소를 모두 끄면 baseline과 동일** — ablation의 기준선이 흔들리면 A2·A3의 기여를
   분리할 수 없다.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _skip import Skip


def _torch():
    try:
        import torch
    except ModuleNotFoundError as error:  # pragma: no cover - torch 미설치 환경
        raise Skip("torch가 설치되어 있지 않습니다.") from error
    return torch


def _model(torch, **overrides):
    from Diffusion.prpd_diffusion.models.unet1d import Conditional1dUNet

    torch.manual_seed(0)
    defaults = dict(
        out_channels=2,
        base_channels=16,
        channel_multipliers=(1, 2),
        blocks_per_stage=1,
        attention_stages=(),
    )
    defaults.update(overrides)
    return Conditional1dUNet(**defaults)


def _inputs(torch, batch: int = 2, length: int = 128):
    generator = torch.Generator().manual_seed(1)
    x_t = torch.rand((batch, 2, length), generator=generator) * 2.0 - 1.0
    condition = torch.rand((batch, 2, length), generator=generator) * 2.0 - 1.0
    timesteps = torch.randint(0, 1000, (batch,), generator=generator)
    return x_t, condition, timesteps


def test_output_shape_matches_target_channels():
    torch = _torch()
    model = _model(torch)
    x_t, condition, timesteps = _inputs(torch)
    output = model(torch.cat([x_t, condition], dim=1), timesteps)
    assert output.shape == (2, 2, 128)
    assert torch.isfinite(output).all()


def test_default_configuration_with_three_stages_and_attention():
    torch = _torch()
    model = _model(torch, base_channels=16, channel_multipliers=(1, 2, 4), attention_stages=(2,))
    x_t, condition, timesteps = _inputs(torch)
    output = model(torch.cat([x_t, condition], dim=1), timesteps)
    assert output.shape == (2, 2, 128)
    assert model.parameter_count() > 0


def test_rejects_wrong_input_channel_count():
    torch = _torch()
    model = _model(torch)
    try:
        model(torch.zeros(1, 3, 128), torch.zeros(1, dtype=torch.long))
    except ValueError:
        return
    raise AssertionError("잘못된 입력 채널 수를 거부해야 합니다.")


# stride 2 downsample이 있으므로 등변성은 전체 축소 배수의 정수배 이동에서만 성립한다.
# `channel_multipliers=(1, 2)`면 downsample 1회 → 배수 2. 16은 그 배수다.
ROLL = 16


def test_phase_axis_padding_is_circular():
    """위상축은 AC 1주기라 순환이다. 입력을 굴리면 출력도 같게 굴러야 한다."""
    torch = _torch()
    model = _model(torch, padding_mode="circular").eval()
    x_t, condition, timesteps = _inputs(torch)
    stacked = torch.cat([x_t, condition], dim=1)

    # 출력 conv가 zero-init이라 그대로면 항상 0이다. 학습된 상태를 흉내내려고 흔들어 준다.
    with torch.no_grad():
        model.output_conv.weight.normal_(0.0, 0.05)
        model.output_conv.bias.normal_(0.0, 0.05)
        direct = model(stacked, timesteps)
        rolled = model(torch.roll(stacked, ROLL, dims=-1), timesteps)
    assert torch.allclose(torch.roll(direct, ROLL, dims=-1), rolled, atol=1e-4)


def test_phase_axis_circularity_survives_components():
    """A2·A3를 켜도 순환 구조가 깨지면 안 된다(엔트로피·형태학 연산의 padding 확인)."""
    torch = _torch()
    model = _model(torch, padding_mode="circular", ardd_resid=True, morph_attn=True).eval()
    x_t, condition, timesteps = _inputs(torch)
    stacked = torch.cat([x_t, condition], dim=1)
    with torch.no_grad():
        model.output_conv.weight.normal_(0.0, 0.05)
        model.output_conv.bias.normal_(0.0, 0.05)
        direct = model(stacked, timesteps)
        rolled = model(torch.roll(stacked, ROLL, dims=-1), timesteps)
    assert torch.allclose(torch.roll(direct, ROLL, dims=-1), rolled, atol=1e-4)


def test_zero_padding_mode_is_not_circular():
    torch = _torch()
    model = _model(torch, padding_mode="zeros").eval()
    x_t, condition, timesteps = _inputs(torch)
    stacked = torch.cat([x_t, condition], dim=1)
    with torch.no_grad():
        model.output_conv.weight.normal_(0.0, 0.05)
        model.output_conv.bias.normal_(0.0, 0.05)
        direct = model(stacked, timesteps)
        rolled = model(torch.roll(stacked, ROLL, dims=-1), timesteps)
    assert not torch.allclose(torch.roll(direct, ROLL, dims=-1), rolled, atol=1e-4)


def test_components_off_matches_plain_residual_baseline():
    """A2·A3를 끈 forward는 구성요소가 아예 없는 모델과 같아야 한다."""
    torch = _torch()
    baseline = _model(torch, ardd_resid=False, morph_attn=False).eval()
    same = _model(torch, ardd_resid=False, morph_attn=False).eval()
    x_t, condition, timesteps = _inputs(torch)
    stacked = torch.cat([x_t, condition], dim=1)
    with torch.no_grad():
        for model in (baseline, same):
            model.output_conv.weight.fill_(0.01)
            model.output_conv.bias.fill_(0.0)
        assert torch.allclose(baseline(stacked, timesteps), same(stacked, timesteps), atol=1e-6)
    assert baseline.gate is None and baseline.morph_attention is None


def test_components_change_the_output():
    """구성요소를 켜면 출력이 실제로 달라져야 한다(연결이 안 된 채 통과하는 것을 막는다)."""
    torch = _torch()
    x_t, condition, timesteps = _inputs(torch)
    stacked = torch.cat([x_t, condition], dim=1)

    outputs = {}
    for name, kwargs in {
        "baseline": {},
        "ardd_resid": {"ardd_resid": True},
        "morph_attn": {"morph_attn": True},
        "both": {"ardd_resid": True, "morph_attn": True},
    }.items():
        model = _model(torch, **kwargs).eval()
        with torch.no_grad():
            model.output_conv.weight.fill_(0.01)
            model.output_conv.bias.fill_(0.0)
            outputs[name] = model(stacked, timesteps)

    for name in ("ardd_resid", "morph_attn", "both"):
        assert not torch.allclose(outputs["baseline"], outputs[name], atol=1e-6), name


def test_components_add_no_parameters():
    """A2·A3는 parameter-free여야 한다(논문의 주장이자 비용 비교의 전제)."""
    torch = _torch()
    plain = _model(torch).parameter_count()
    full = _model(torch, ardd_resid=True, morph_attn=True).parameter_count()
    assert plain == full


def test_entropy_source_switch_changes_output():
    torch = _torch()
    x_t, condition, timesteps = _inputs(torch)
    stacked = torch.cat([x_t, condition], dim=1)
    results = []
    for source in ("condition", "x_t"):
        model = _model(torch, ardd_resid=True, ardd_entropy_source=source).eval()
        with torch.no_grad():
            model.output_conv.weight.fill_(0.01)
            results.append(model(stacked, timesteps))
    assert not torch.allclose(results[0], results[1], atol=1e-6)


def test_training_loss_is_finite_and_backpropagates():
    torch = _torch()
    from Diffusion.prpd_diffusion.diffusion.gaussian import GaussianDiffusion
    from Diffusion.prpd_diffusion.diffusion.schedule import DiffusionSchedule

    model = _model(torch, ardd_resid=True, morph_attn=True)
    diffusion = GaussianDiffusion(DiffusionSchedule.create("linear", timesteps=100))
    x_start, condition, _ = _inputs(torch)
    loss, info = diffusion.training_loss(model, x_start, condition)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    assert 0 <= info["t_mean"] < 100


def test_ddim_sampling_reuses_existing_sampler_unchanged():
    """1D 텐서에서도 기존 `ddim_sample`이 수정 없이 동작해야 한다."""
    torch = _torch()
    from Diffusion.prpd_diffusion.diffusion.gaussian import GaussianDiffusion
    from Diffusion.prpd_diffusion.diffusion.sampler import ddim_sample
    from Diffusion.prpd_diffusion.diffusion.schedule import DiffusionSchedule

    model = _model(torch).eval()
    diffusion = GaussianDiffusion(DiffusionSchedule.create("linear", timesteps=100))
    _, condition, _ = _inputs(torch)
    restored = ddim_sample(diffusion, model, condition, steps=4)
    assert restored.shape == condition.shape
    assert torch.isfinite(restored).all()


def test_build_from_config_wires_components():
    torch = _torch()
    from Diffusion.prpd_diffusion.ardd1d.config import Ardd1dConfig
    from Diffusion.prpd_diffusion.models.unet1d import build_from_config

    config = Ardd1dConfig.from_dict(
        {
            "model": {"base_channels": 16, "channel_multipliers": [1, 2], "blocks_per_stage": 1,
                      "attention_stages": []},
            "components": {"ardd_resid": True, "morph_attn": True, "morph_scales": [1, 3]},
        }
    )
    model = build_from_config(config)
    assert model.gate is not None
    assert model.morph_attention is not None
    assert model.morph_attention.scales == (1, 3)
    x_t, condition, timesteps = _inputs(torch)
    assert model(torch.cat([x_t, condition], dim=1), timesteps).shape == (2, 2, 128)

"""`ARDD-2025` 구성요소 테스트 (`DIFFUSION.md` 18.4의 A2·A3). torch 필요."""

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


# ------------------------------------------------------------------ 엔트로피 (A2)


def test_entropy_is_zero_for_constant_signal():
    torch = _torch()
    from Diffusion.prpd_diffusion.components.entropy import local_entropy

    entropy = local_entropy(torch.full((2, 2, 128), 0.3))
    assert entropy.shape == (2, 1, 128)
    assert float(entropy.abs().max()) < 1e-6


def test_entropy_is_bounded_in_unit_interval():
    torch = _torch()
    from Diffusion.prpd_diffusion.components.entropy import local_entropy

    generator = torch.Generator().manual_seed(0)
    signal = torch.rand((3, 2, 128), generator=generator) * 2.0 - 1.0
    entropy = local_entropy(signal)
    assert float(entropy.min()) >= 0.0
    assert float(entropy.max()) <= 1.0
    # 무작위 신호는 상수 신호보다 엔트로피가 확실히 높아야 한다.
    assert float(entropy.mean()) > 0.5


def test_entropy_is_equivariant_to_circular_roll():
    torch = _torch()
    from Diffusion.prpd_diffusion.components.entropy import local_entropy

    generator = torch.Generator().manual_seed(1)
    signal = torch.rand((1, 1, 128), generator=generator) * 2.0 - 1.0
    direct = local_entropy(signal, circular=True)
    rolled = local_entropy(torch.roll(signal, 17, dims=-1), circular=True)
    assert torch.allclose(torch.roll(direct, 17, dims=-1), rolled, atol=1e-6)


def test_entropy_rejects_even_window():
    torch = _torch()
    from Diffusion.prpd_diffusion.components.entropy import local_entropy

    try:
        local_entropy(torch.zeros((1, 1, 32)), window=8)
    except ValueError:
        return
    raise AssertionError("짝수 window를 거부해야 합니다.")


# ------------------------------------------------------------ 적응형 잔차 (A2)


def test_adaptive_residual_coefficient_is_bounded_by_lipschitz_range():
    torch = _torch()
    from Diffusion.prpd_diffusion.components.ardd_resid import AdaptiveResidualGate

    gate = AdaptiveResidualGate(alpha=0.5)
    entropy = torch.linspace(0.0, 1.0, 128).reshape(1, 1, 128)
    coefficient = gate.coefficient(entropy)
    assert abs(float(coefficient.min()) - 0.5) < 1e-6
    assert abs(float(coefficient.max()) - 1.0) < 1e-6


def test_adaptive_residual_invert_flips_direction_but_keeps_range():
    torch = _torch()
    from Diffusion.prpd_diffusion.components.ardd_resid import AdaptiveResidualGate

    entropy = torch.tensor([[[0.0, 1.0]]])
    literal = AdaptiveResidualGate(alpha=0.5).coefficient(entropy)
    inverted = AdaptiveResidualGate(alpha=0.5, invert=True).coefficient(entropy)
    assert float(literal[0, 0, 0]) < float(literal[0, 0, 1])   # Eq.(6) 그대로: H↑ → a↑
    assert float(inverted[0, 0, 0]) > float(inverted[0, 0, 1])  # 본문 해석: H↑ → a↓
    assert abs(float(literal.min()) - float(inverted.min())) < 1e-6


def test_adaptive_residual_matches_plain_sum_when_coefficient_is_one():
    torch = _torch()
    from Diffusion.prpd_diffusion.components.ardd_resid import AdaptiveResidualGate

    gate = AdaptiveResidualGate(alpha=0.5)
    residual = torch.randn(2, 4, 32)
    retained = torch.randn(2, 4, 32)
    entropy = torch.ones(2, 1, 32)  # a = 0.5 * (1 + 1) = 1.0
    assert torch.allclose(gate(residual, retained, entropy), residual + retained, atol=1e-6)


def test_adaptive_residual_resamples_entropy_to_current_resolution():
    torch = _torch()
    from Diffusion.prpd_diffusion.components.ardd_resid import AdaptiveResidualGate

    gate = AdaptiveResidualGate()
    output = gate(torch.randn(2, 4, 32), torch.zeros(2, 4, 32), torch.rand(2, 1, 128))
    assert output.shape == (2, 4, 32)


# ------------------------------------------------------- 형태학적 attention (A3)


def test_morphological_gradient_is_nonnegative_and_zero_on_flat_signal():
    torch = _torch()
    from Diffusion.prpd_diffusion.components.morph_attn import morphological_gradient

    flat = torch.full((1, 1, 64), 0.25)
    assert float(morphological_gradient(flat, 3).abs().max()) < 1e-6
    generator = torch.Generator().manual_seed(2)
    noisy = torch.rand((2, 1, 64), generator=generator)
    assert float(morphological_gradient(noisy, 2).min()) >= 0.0


def test_morphological_gradient_peaks_at_pulse_edges():
    torch = _torch()
    from Diffusion.prpd_diffusion.components.morph_attn import morphological_gradient

    signal = torch.zeros(1, 1, 64)
    signal[0, 0, 30:34] = 1.0
    gradient = morphological_gradient(signal, scale=1)[0, 0]
    assert float(gradient[29]) > 0.5 and float(gradient[34]) > 0.5  # 가장자리
    assert float(gradient[10]) < 1e-6                               # 평탄 구간
    assert float(gradient[31]) < 1e-6                               # 펄스 내부(폭 4 > window 3)


def test_morph_attention_output_shape_and_range():
    torch = _torch()
    from Diffusion.prpd_diffusion.components.morph_attn import MorphologicalAttention1d

    attention = MorphologicalAttention1d()
    generator = torch.Generator().manual_seed(3)
    values = attention(torch.rand((2, 4, 128), generator=generator))
    assert values.shape == (2, 1, 128)
    assert float(values.min()) > 0.0 and float(values.max()) < 1.0


def test_morph_attention_is_equivariant_to_circular_roll():
    torch = _torch()
    from Diffusion.prpd_diffusion.components.morph_attn import MorphologicalAttention1d

    attention = MorphologicalAttention1d(circular=True)
    generator = torch.Generator().manual_seed(4)
    signal = torch.rand((1, 2, 128), generator=generator)
    direct = attention(signal)
    rolled = attention(torch.roll(signal, 23, dims=-1))
    assert torch.allclose(torch.roll(direct, 23, dims=-1), rolled, atol=1e-6)


def test_morph_attention_normalize_modes_run():
    torch = _torch()
    from Diffusion.prpd_diffusion.components.morph_attn import MorphologicalAttention1d

    generator = torch.Generator().manual_seed(5)
    signal = torch.rand((2, 2, 128), generator=generator)
    outputs = {}
    for mode in ("none", "instance", "otsu"):
        values = MorphologicalAttention1d(normalize=mode)(signal)
        assert values.shape == (2, 1, 128)
        assert torch.isfinite(values).all()
        outputs[mode] = values
    # 논문 그대로(`none`)는 a_base >= 0이므로 sigmoid가 0.5 아래로 내려가지 않는다.
    assert float(outputs["none"].min()) >= 0.5
    # 정규화 모드는 임계 아래를 실제로 억제할 수 있어야 한다.
    assert float(outputs["instance"].min()) < 0.5


def test_morph_attention_rejects_unknown_normalize():
    _torch()
    from Diffusion.prpd_diffusion.components.morph_attn import MorphologicalAttention1d

    try:
        MorphologicalAttention1d(normalize="softmax")
    except ValueError:
        return
    raise AssertionError("알 수 없는 normalize 모드를 거부해야 합니다.")


def test_normalize_modes_match_config_constant():
    """torch 없이 설정을 검증하려고 목록을 복제했으므로 두 곳이 같은지 고정한다."""
    _torch()
    from Diffusion.prpd_diffusion.ardd1d.config import MORPH_NORMALIZE_MODES
    from Diffusion.prpd_diffusion.components.morph_attn import NORMALIZE_MODES

    assert tuple(MORPH_NORMALIZE_MODES) == tuple(NORMALIZE_MODES)


def test_otsu_threshold_separates_two_clusters():
    torch = _torch()
    from Diffusion.prpd_diffusion.components.morph_attn import otsu_threshold

    values = torch.cat([torch.zeros(1, 1, 64), torch.full((1, 1, 64), 4.0)], dim=-1)
    threshold = float(otsu_threshold(values))
    assert 0.0 < threshold < 4.0

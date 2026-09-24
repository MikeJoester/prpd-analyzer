"""조건부 1D U-Net (epsilon 예측) — ARDD 1D 트랙의 백본. `DIFFUSION.md` 18.4의 A5.

입력은 `[x_t, y]`를 채널로 concat한 `(B, 2C, L)`, 출력은 `(B, C, L)`.
기본 표현 `profile_v1`에서는 `C=2`(mean/max 채널), `L=128`(위상 bin)이므로
입력 `(B, 4, 128)` → 출력 `(B, 2, 128)`이다.

위상축(128)은 AC 1주기라 순환 구조이므로 `padding_mode="circular"`가 기본이다.
`flat256`처럼 mean·max를 한 줄로 이어붙인 표현을 쓸 때만 `zeros`로 바꾼다(128/129 경계가
인위적이기 때문).

논문 구성요소(`DIFFUSION.md` 18.4)는 별도 모델이 아니라 이 백본의 on/off 옵션이다.

* `A2 ardd_resid` — ResidualBlock의 잔차 합을 엔트로피 게이트로 대체
* `A3 morph_attn` — skip connection에 형태학적 attention 계수를 곱함

**둘 다 끄면 forward는 2D baseline(`models.unet.ConditionalUNet`)의 1D 대응과
수학적으로 동일하다**(`tests/test_model1d.py`에서 단언).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..components.ardd_resid import AdaptiveResidualGate
from ..components.entropy import local_entropy
from ..components.morph_attn import MorphologicalAttention1d
from .common import group_norm, timestep_embedding

__all__ = ["Conditional1dUNet", "ResidualBlock1d", "SelfAttention1d"]


def _conv(in_channels: int, out_channels: int, padding_mode: str, stride: int = 1) -> nn.Conv1d:
    return nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        padding_mode=padding_mode,
    )


class ResidualBlock1d(nn.Module):
    """GroupNorm + SiLU + Conv1d 2단, timestep bias 주입, 잔차 합.

    `gate`가 주어지면 잔차 합이 `skip(x) + a(H)·hidden`으로 바뀐다(`ARDD-2025` Eq.5).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        dropout: float = 0.0,
        padding_mode: str = "circular",
        gate: AdaptiveResidualGate | None = None,
    ) -> None:
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.conv1 = _conv(in_channels, out_channels, padding_mode)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = _conv(out_channels, out_channels, padding_mode)
        self.skip = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )
        self.gate = gate

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
        entropy: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_projection(F.silu(time_embedding))[:, :, None]
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        retained = self.skip(x)
        if self.gate is not None and entropy is not None:
            return self.gate(hidden, retained, entropy)
        return hidden + retained


class SelfAttention1d(nn.Module):
    """저해상도 단계의 전역 self-attention. 위상 전체에 걸친 패턴을 잇는다."""

    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.norm = group_norm(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.projection = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, length = x.shape
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(batch, 3, self.heads, channels // self.heads, length)
        query, key, value = qkv.unbind(dim=1)
        attention = F.scaled_dot_product_attention(
            query.transpose(-2, -1), key.transpose(-2, -1), value.transpose(-2, -1)
        )
        attention = attention.transpose(-2, -1).reshape(batch, channels, length)
        return x + self.projection(attention)


class Conditional1dUNet(nn.Module):
    """`model(cat[x_t, y], t) -> eps_hat`.

    `models.unet.ConditionalUNet`과 같은 구조를 1D로 옮긴 것이라 `GaussianDiffusion`,
    `ddim_sample` 등 기존 diffusion 코드가 **수정 없이** 그대로 동작한다.
    """

    def __init__(
        self,
        out_channels: int = 2,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 2, 4),
        blocks_per_stage: int = 2,
        attention_stages: tuple[int, ...] = (2,),
        dropout: float = 0.0,
        padding_mode: str = "circular",
        # --- 논문 구성요소 (DIFFUSION.md 18.4) -------------------------------
        ardd_resid: bool = False,
        ardd_alpha: float = 0.5,
        ardd_entropy_window: int = 9,
        ardd_entropy_bins: int = 16,
        ardd_entropy_source: str = "condition",
        ardd_invert_entropy: bool = False,
        morph_attn: bool = False,
        morph_scales: tuple[int, ...] = (1, 3, 7),
        morph_normalize: str = "none",
    ) -> None:
        super().__init__()
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.in_channels = out_channels * 2  # [x_t, condition]
        self.circular = padding_mode == "circular"

        # ---- 논문 구성요소 --------------------------------------------------
        self.ardd_resid = bool(ardd_resid)
        self.ardd_entropy_window = int(ardd_entropy_window)
        self.ardd_entropy_bins = int(ardd_entropy_bins)
        self.ardd_entropy_source = ardd_entropy_source
        gate = (
            AdaptiveResidualGate(alpha=ardd_alpha, invert=ardd_invert_entropy)
            if self.ardd_resid
            else None
        )
        self.gate = gate
        self.morph_attention = (
            MorphologicalAttention1d(
                scales=morph_scales, circular=self.circular, normalize=morph_normalize
            )
            if morph_attn
            else None
        )

        self.input_conv = _conv(self.in_channels, base_channels, padding_mode)

        self.down_blocks = nn.ModuleList()
        self.down_attentions = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_channels = [base_channels]
        channels = base_channels
        for stage, multiplier in enumerate(channel_multipliers):
            stage_channels = base_channels * multiplier
            for _ in range(blocks_per_stage):
                self.down_blocks.append(
                    ResidualBlock1d(channels, stage_channels, time_dim, dropout, padding_mode, gate)
                )
                self.down_attentions.append(
                    SelfAttention1d(stage_channels) if stage in attention_stages else nn.Identity()
                )
                channels = stage_channels
                skip_channels.append(channels)
            is_last = stage == len(channel_multipliers) - 1
            self.downsamples.append(
                nn.Identity() if is_last else _conv(channels, channels, padding_mode, stride=2)
            )
            if not is_last:
                skip_channels.append(channels)

        self.middle_block1 = ResidualBlock1d(
            channels, channels, time_dim, dropout, padding_mode, gate
        )
        self.middle_attention = SelfAttention1d(channels)
        self.middle_block2 = ResidualBlock1d(
            channels, channels, time_dim, dropout, padding_mode, gate
        )

        self.up_blocks = nn.ModuleList()
        self.up_attentions = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for stage, multiplier in reversed(list(enumerate(channel_multipliers))):
            stage_channels = base_channels * multiplier
            for _ in range(blocks_per_stage + 1):
                self.up_blocks.append(
                    ResidualBlock1d(
                        channels + skip_channels.pop(),
                        stage_channels,
                        time_dim,
                        dropout,
                        padding_mode,
                        gate,
                    )
                )
                self.up_attentions.append(
                    SelfAttention1d(stage_channels) if stage in attention_stages else nn.Identity()
                )
                channels = stage_channels
            self.upsamples.append(
                nn.Identity() if stage == 0 else _conv(channels, channels, padding_mode)
            )

        self.output_norm = group_norm(channels)
        self.output_conv = _conv(channels, out_channels, padding_mode)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

        self.blocks_per_stage = blocks_per_stage
        self.stage_count = len(channel_multipliers)

    # ------------------------------------------------------------------ 보조

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _entropy_map(self, x: torch.Tensor) -> torch.Tensor | None:
        """`ardd_resid`가 켜졌을 때만 엔트로피 맵을 만든다.

        기본 `entropy_source="condition"`은 노이즈 입력 `y`에서 계산한다. `x_t`는 diffusion
        step마다 가우시안 노이즈가 섞여 있어 "국소 구조"를 나타내지 못하기 때문이다.
        """
        if not self.ardd_resid:
            return None
        source = (
            x[:, self.out_channels :]
            if self.ardd_entropy_source == "condition"
            else x[:, : self.out_channels]
        )
        return local_entropy(
            source,
            window=self.ardd_entropy_window,
            bins=self.ardd_entropy_bins,
            circular=self.circular,
        )

    def _skip(self, tensor: torch.Tensor) -> torch.Tensor:
        """skip connection에 형태학적 attention을 곱한다(`ARDD-2025` Eq.15)."""
        if self.morph_attention is None:
            return tensor
        return tensor * self.morph_attention(tensor)

    # ------------------------------------------------------------------ forward

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"입력 shape이 (B, C, L)이 아닙니다: {tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"입력 채널 수가 {self.in_channels}가 아닙니다: {x.shape[1]}")

        time_embedding = self.time_mlp(timestep_embedding(timesteps, self.base_channels))
        entropy = self._entropy_map(x)

        hidden = self.input_conv(x)
        skips = [hidden]
        block_index = 0
        for stage in range(self.stage_count):
            for _ in range(self.blocks_per_stage):
                hidden = self.down_blocks[block_index](hidden, time_embedding, entropy)
                hidden = self.down_attentions[block_index](hidden)
                skips.append(hidden)
                block_index += 1
            downsample = self.downsamples[stage]
            if not isinstance(downsample, nn.Identity):
                hidden = downsample(hidden)
                skips.append(hidden)

        hidden = self.middle_block1(hidden, time_embedding, entropy)
        hidden = self.middle_attention(hidden)
        hidden = self.middle_block2(hidden, time_embedding, entropy)

        block_index = 0
        for stage_position in range(self.stage_count):
            for _ in range(self.blocks_per_stage + 1):
                hidden = torch.cat([hidden, self._skip(skips.pop())], dim=1)
                hidden = self.up_blocks[block_index](hidden, time_embedding, entropy)
                hidden = self.up_attentions[block_index](hidden)
                block_index += 1
            upsample = self.upsamples[stage_position]
            if not isinstance(upsample, nn.Identity):
                hidden = upsample(F.interpolate(hidden, scale_factor=2, mode="nearest"))

        return self.output_conv(F.silu(self.output_norm(hidden)))


def build_from_config(config) -> Conditional1dUNet:
    """`ardd1d.config.Ardd1dConfig` → 모델. components를 그대로 옮긴다."""
    from ..data.profile import PROFILE_CHANNELS

    model = config.model
    components = config.components
    return Conditional1dUNet(
        out_channels=PROFILE_CHANNELS,
        base_channels=model.base_channels,
        channel_multipliers=tuple(model.channel_multipliers),
        blocks_per_stage=model.blocks_per_stage,
        attention_stages=tuple(model.attention_stages),
        dropout=model.dropout,
        padding_mode=model.padding_mode,
        ardd_resid=components.ardd_resid,
        ardd_alpha=components.ardd_alpha,
        ardd_entropy_window=components.ardd_entropy_window,
        ardd_entropy_bins=components.ardd_entropy_bins,
        ardd_entropy_source=components.ardd_entropy_source,
        ardd_invert_entropy=components.ardd_invert_entropy,
        morph_attn=components.morph_attn,
        morph_scales=tuple(components.morph_scales),
        morph_normalize=components.morph_normalize,
    )

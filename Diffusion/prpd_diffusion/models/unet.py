"""조건부 2D U-Net (epsilon 예측).

입력은 `[x_t, y]`를 채널로 concat한 `(B, 2, 128, W)`, 출력은 `(B, 1, 128, W)`.
기본 해상도는 `128 x 256` crop이며 downsample 3단계에서 `16 x 32`가 된다.

위상축(128)은 순환 구조이므로 위상 방향 padding을 `circular`로 둔다.
시간축은 순환이 아니므로 `zeros`. → conv마다 두 축을 분리해 padding 한다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import group_norm, timestep_embedding

__all__ = ["timestep_embedding", "PhaseCircularConv2d", "ResidualBlock", "SelfAttention", "ConditionalUNet"]


class PhaseCircularConv2d(nn.Conv2d):
    """위상축은 circular, 시간축은 zero padding으로 3x3 convolution을 수행한다."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0)
        self._pad = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        pad = self._pad
        if pad:
            x = F.pad(x, (0, 0, pad, pad), mode="circular")   # 위상축(H)
            x = F.pad(x, (pad, pad, 0, 0), mode="constant")   # 시간축(W)
        return self._conv_forward(x, self.weight, self.bias)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.conv1 = PhaseCircularConv2d(in_channels, out_channels)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.norm2 = group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = PhaseCircularConv2d(out_channels, out_channels)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x)))
        hidden = hidden + self.time_projection(F.silu(time_embedding))[:, :, None, None]
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return hidden + self.skip(x)


class SelfAttention(nn.Module):
    """저해상도 단계의 전역 self-attention. 위상 전체에 걸친 패턴을 잇는다."""

    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.norm = group_norm(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.projection = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(batch, 3, self.heads, channels // self.heads, height * width)
        query, key, value = qkv.unbind(dim=1)
        attention = F.scaled_dot_product_attention(
            query.transpose(-2, -1), key.transpose(-2, -1), value.transpose(-2, -1)
        )
        attention = attention.transpose(-2, -1).reshape(batch, channels, height, width)
        return x + self.projection(attention)


class ConditionalUNet(nn.Module):
    """`model(cat[x_t, y], t) -> eps_hat`"""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 2, 4),
        blocks_per_stage: int = 2,
        attention_stages: tuple[int, ...] = (2,),
        dropout: float = 0.0,
    ):
        super().__init__()
        time_dim = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.base_channels = base_channels

        self.input_conv = PhaseCircularConv2d(in_channels, base_channels)

        self.down_blocks = nn.ModuleList()
        self.down_attentions = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        skip_channels = [base_channels]
        channels = base_channels
        for stage, multiplier in enumerate(channel_multipliers):
            stage_channels = base_channels * multiplier
            for _ in range(blocks_per_stage):
                self.down_blocks.append(ResidualBlock(channels, stage_channels, time_dim, dropout))
                self.down_attentions.append(
                    SelfAttention(stage_channels) if stage in attention_stages else nn.Identity()
                )
                channels = stage_channels
                skip_channels.append(channels)
            is_last = stage == len(channel_multipliers) - 1
            self.downsamples.append(
                nn.Identity() if is_last else PhaseCircularConv2d(channels, channels, stride=2)
            )
            if not is_last:
                skip_channels.append(channels)

        self.middle_block1 = ResidualBlock(channels, channels, time_dim, dropout)
        self.middle_attention = SelfAttention(channels)
        self.middle_block2 = ResidualBlock(channels, channels, time_dim, dropout)

        self.up_blocks = nn.ModuleList()
        self.up_attentions = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for stage, multiplier in reversed(list(enumerate(channel_multipliers))):
            stage_channels = base_channels * multiplier
            for _ in range(blocks_per_stage + 1):
                self.up_blocks.append(
                    ResidualBlock(channels + skip_channels.pop(), stage_channels, time_dim, dropout)
                )
                self.up_attentions.append(
                    SelfAttention(stage_channels) if stage in attention_stages else nn.Identity()
                )
                channels = stage_channels
            self.upsamples.append(
                nn.Identity() if stage == 0 else PhaseCircularConv2d(channels, channels)
            )

        self.output_norm = group_norm(channels)
        self.output_conv = PhaseCircularConv2d(channels, out_channels)
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

        self.blocks_per_stage = blocks_per_stage
        self.stage_count = len(channel_multipliers)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_mlp(timestep_embedding(timesteps, self.base_channels))

        hidden = self.input_conv(x)
        skips = [hidden]
        block_index = 0
        for stage in range(self.stage_count):
            for _ in range(self.blocks_per_stage):
                hidden = self.down_blocks[block_index](hidden, time_embedding)
                hidden = self.down_attentions[block_index](hidden)
                skips.append(hidden)
                block_index += 1
            downsample = self.downsamples[stage]
            if not isinstance(downsample, nn.Identity):
                hidden = downsample(hidden)
                skips.append(hidden)

        hidden = self.middle_block1(hidden, time_embedding)
        hidden = self.middle_attention(hidden)
        hidden = self.middle_block2(hidden, time_embedding)

        block_index = 0
        for stage_position in range(self.stage_count):
            for _ in range(self.blocks_per_stage + 1):
                hidden = torch.cat([hidden, skips.pop()], dim=1)
                hidden = self.up_blocks[block_index](hidden, time_embedding)
                hidden = self.up_attentions[block_index](hidden)
                block_index += 1
            upsample = self.upsamples[stage_position]
            if not isinstance(upsample, nn.Identity):
                hidden = upsample(F.interpolate(hidden, scale_factor=2, mode="nearest"))

        return self.output_conv(F.silu(self.output_norm(hidden)))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

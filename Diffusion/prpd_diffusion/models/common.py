"""2D/1D U-Net이 공유하는 작은 유틸. torch 필요.

`group_count`가 존재하는 이유: U-Net decoder는 skip connection을 concat하므로 채널 수가
`stage_channels + skip_channels`가 되어 32의 배수가 아닌 값(예: 16+32=48)이 자주 나온다.
`nn.GroupNorm(min(32, C), C)`는 이때 "num_channels must be divisible by num_groups"로 실패한다.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_GROUPS = 32


def group_count(channels: int, maximum: int = MAX_GROUPS) -> int:
    """`channels`를 나누어떨어뜨리는 가장 큰 그룹 수(최대 `maximum`)."""
    if channels < 1:
        raise ValueError(f"채널 수는 1 이상이어야 합니다: {channels}")
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def group_norm(channels: int, maximum: int = MAX_GROUPS) -> nn.GroupNorm:
    """채널 수에 관계없이 항상 유효한 GroupNorm을 만든다."""
    return nn.GroupNorm(group_count(channels, maximum), channels)


def timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """DDPM 표준 sinusoidal timestep embedding."""
    half = dim // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding

import torch
import torch.nn as nn
from .asym_autoencoder import AsymmetricalAutoencoderKL
from ldm.modules.diffusionmodules.openaimodel import UNetModel

class LDMAutoencoderWrapper(nn.Module):
    def __init__(self, embed_dim=4, ch_mult=(1, 2, 4), num_res_blocks=2):
        super().__init__()
        self.autoencoder = AsymmetricalAutoencoderKL(in_channels=1, latent_channels=embed_dim, base_channels=16)

    def forward(self, x):
        return self.autoencoder(x)

    def encode(self, x):
        return self.autoencoder.encode(x).sample()

    def decode(self, z):
        return self.autoencoder.decode(z)

from .unet1d import Conditional1dUNet

class LDMUNetWrapper(nn.Module):
    def __init__(self, out_channels=4, base_channels=128):
        super().__init__()
        self.unet = Conditional1dUNet(
            out_channels=out_channels,
            base_channels=base_channels,
            channel_multipliers=(1, 2, 4),
            blocks_per_stage=2,
            attention_stages=(1, 2),
            dropout=0.1,
            padding_mode="zeros"
        )

    def forward(self, x, t):
        # x is (B, 8, 128, 1). We need (B, 8, 128)
        x_1d = x.squeeze(-1)
        out_1d = self.unet(x_1d, t)
        # out is (B, 4, 128). We need (B, 4, 128, 1)
        return out_1d.unsqueeze(-1)

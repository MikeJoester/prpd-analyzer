import torch
import torch.nn as nn
from torch.distributions import Normal

class AsymDownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=(3, 3), padding=(1, 1), padding_mode='circular')
        self.down = nn.Conv2d(out_c, out_c, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1))
        self.norm = nn.GroupNorm(8, out_c)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.norm(self.conv(x)))
        x = self.down(x)
        return x

class AsymUpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1))
        self.conv = nn.Conv2d(out_c, out_c, kernel_size=(3, 3), padding=(1, 1), padding_mode='circular')
        self.norm = nn.GroupNorm(8, out_c)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.up(x)
        x = self.act(self.norm(self.conv(x)))
        return x

class AsymmetricalAutoencoderKL(nn.Module):
    def __init__(self, in_channels=1, latent_channels=4, base_channels=32):
        super().__init__()
        
        # Downsample time axis by 2^8 = 256.
        channels = [base_channels, base_channels, base_channels*2, base_channels*2, 
                    base_channels*4, base_channels*4, base_channels*4, base_channels*4]
        
        # Encoder
        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, padding=1, padding_mode='circular')
        self.down_blocks = nn.ModuleList()
        in_c = channels[0]
        for c in channels:
            self.down_blocks.append(AsymDownBlock(in_c, c))
            in_c = c
        self.conv_out_enc = nn.Conv2d(in_c, latent_channels * 2, 3, padding=1, padding_mode='circular')
        
        # Decoder
        self.conv_in_dec = nn.Conv2d(latent_channels, channels[-1], 3, padding=1, padding_mode='circular')
        self.up_blocks = nn.ModuleList()
        in_c = channels[-1]
        for c in reversed(channels):
            self.up_blocks.append(AsymUpBlock(in_c, c))
            in_c = c
        self.conv_out_dec = nn.Conv2d(channels[0], in_channels, 3, padding=1, padding_mode='circular')

    def encode(self, x):
        h = self.conv_in(x)
        for block in self.down_blocks:
            h = block(h)
        moments = self.conv_out_enc(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        return Normal(mean, std)

    def decode(self, z):
        h = self.conv_in_dec(z)
        for block in self.up_blocks:
            h = block(h)
        out = self.conv_out_dec(h)
        return out

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.rsample()
        else:
            z = posterior.mean
        recon = self.decode(z)
        return recon, posterior

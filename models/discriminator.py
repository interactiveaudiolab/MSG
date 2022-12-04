# This code is taken from 

# Jang, Won, et al. 
# "UnivNet: A neural vocoder with multi-resolution spectrogram discriminators for high-fidelity waveform generation." 
# arXiv preprint arXiv:2106.07889 (2021).

# https://github.com/mindslab-ai/univnet 


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConv2d(*args, **kwargs):
    return weight_norm(nn.Conv2d(*args, **kwargs))


class DiscriminatorP(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList(
            [
                WNConv2d(1, 32, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0)),
                WNConv2d(1024, 1024, (5, 1), 1, padding=(2, 0)),
            ]
        )
        self.conv_post = WNConv2d(1024, 1, kernel_size=(3, 1), padding=(1, 0))

    def pad_to_period(self, x):
        t = x.shape[-1]
        x = F.pad(x, (0, self.period - t % self.period), mode="reflect")
        return x

    def forward(self, x):
        fmap = []

        x = self.pad_to_period(x)
        x = rearrange(x, "b c (l p) -> b c l p", p=self.period)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return fmap


class DiscriminatorS(nn.Module):
    def __init__(self, window_length):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                WNConv2d(1, 32, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(32, 32, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(32, 32, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(32, 32, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(32, 32, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        self.conv_post = WNConv2d(32, 1, kernel_size=(3, 3), padding=(1, 1))
        self.window_length = window_length

    def forward(self, x):
        x = torch.stft(
            x.reshape(-1, x.shape[-1]),
            n_fft=self.window_length,
            hop_length=self.window_length//4,
            return_complex=True,
            center=True
        ).unsqueeze(1)
        x = torch.abs(x)
        fmap = []

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return fmap


class Discriminator(nn.Module):
    def __init__(
        self, periods: list = [2, 3, 5, 7, 11], fft_sizes: list = [2048, 1024, 512]
    ):
        super().__init__()

        discs = [DiscriminatorS(fft_size) for fft_size in fft_sizes]
        discs += [DiscriminatorP(i) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y):
        # Peak normalize the volume of input audio
        y = y / (y.abs().max(dim=-1, keepdim=True)[0].clamp(1e-9))

        fmaps = []
        for i, d in enumerate(self.discriminators):
            fmap = d(y)
            fmaps.append(fmap)

        return fmaps


if __name__ == "__main__":
    disc = Discriminator()
    x = torch.randn(2, 1, 16000)
    results = disc(x)
    for result in results:
        for i, r in enumerate(result):
            print(r.mean(), r.min(), r.max())
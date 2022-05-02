import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import AvgPool1d
from torch.nn.utils import spectral_norm
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def WNConv2d(*args, **kwargs):
    return weight_norm(nn.Conv2d(*args, **kwargs))

def NormedConv1d(*args, **kwargs):
    norm = kwargs.pop("norm", "weight_norm")
    if norm == "weight_norm":
        return weight_norm(nn.Conv1d(*args, **kwargs))
    elif norm == "spectral_norm":
        return spectral_norm(nn.Conv1d(*args, **kwargs))


def NormedConv2d(*args, **kwargs):
    norm = kwargs.pop("norm", "weight_norm")
    if norm == "weight_norm":
        return weight_norm(nn.Conv2d(*args, **kwargs))
    elif norm == "spectral_norm":
        return spectral_norm(nn.Conv2d(*args, **kwargs))


class DiscriminatorP(nn.Module):
    def __init__(self, period, norm="weight_norm"):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                NormedConv2d(1, 32, (5, 1), (3, 1), padding=(2, 0), norm=norm),
                NormedConv2d(32, 128, (5, 1), (3, 1), padding=(2, 0), norm=norm),
                NormedConv2d(128, 512, (5, 1), (3, 1), padding=(2, 0), norm=norm),
                NormedConv2d(512, 1024, (5, 1), (3, 1), padding=(2, 0), norm=norm),
                NormedConv2d(1024, 1024, (5, 1), 1, padding=(2, 0), norm=norm),
            ]
        )

        self.conv_post = NormedConv2d(1024, 1, kernel_size=(3, 1), padding=(1, 0))
        self.period = period

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap

class DiscriminatorSpec(nn.Module):
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
        self.sample_rate = 44100

    def forward(self, x):
        x = torch.stft(
            x.reshape(-1, x.shape[-1]),
            n_fft=self.window_length,
            hop_length=self.window_length//4,
            return_complex=True,
            center=True
        )
        x = torch.abs(x)
        fmap = []

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)

        return fmap

class DiscriminatorS(nn.Module):
    def __init__(self, norm="weight_norm"):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                NormedConv1d(1, 16, 15, 1, padding=7, norm=norm),
                NormedConv1d(16, 64, 41, 4, groups=4, padding=20, norm=norm),
                NormedConv1d(64, 256, 41, 4, groups=16, padding=20, norm=norm),
                NormedConv1d(256, 1024, 41, 4, groups=64, padding=20, norm=norm),
                NormedConv1d(1024, 1024, 41, 4, groups=256, padding=20, norm=norm),
                NormedConv1d(1024, 1024, 5, 1, padding=2, norm=norm),
            ]
        )
        self.conv_post = NormedConv1d(1024, 1, 3, 1, padding=1)

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)

        return fmap

class Discriminator(nn.Module):
    def __init__(self,periods = [2, 3, 5, 7, 11],fft_sizes: list = [2048, 1024, 512],norm: str = "weight_norm"):
        super().__init__()
        
        discs = [DiscriminatorSpec(fft_size) for fft_size in fft_sizes]
        discs += [DiscriminatorP(i, norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y):
        # Peak normalize the volume of input audio
        #y = y / (y.abs().max(dim=-1, keepdim=True)[0] + 1e-9)

        fmaps = []
        for i, d in enumerate(self.discriminators):
            fmap = d(y)
            fmaps.append(fmap)

        return fmaps



if __name__ == "__main__":
    disc = Discriminator()
    x = torch.randn(1, 1, 44100)
    results = disc(x)
    for result in results:
        for i, r in enumerate(result):
            print(r.mean(), r.min(), r.max())


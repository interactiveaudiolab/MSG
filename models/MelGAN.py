from torch import nn
import torch.nn.functional as F
import torch
from librosa.filters import mel as librosa_mel_fn
from torch.nn.utils import weight_norm
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


def WNConv2d(*args, **kwargs):
    return weight_norm(nn.Conv2d(*args, **kwargs))


def WNConvTranspose2d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose2d(*args, **kwargs))


class Audio2Mel(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        sampling_rate=44100,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=None,
    ):
        super().__init__()
        ##############################################
        # FFT Parameters                              #
        ##############################################
        window = torch.hann_window(win_length).float()
        mel_basis = librosa_mel_fn(
            sampling_rate, n_fft, n_mel_channels, mel_fmin, mel_fmax
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels

    def forward(self, audio):
        p = (self.n_fft - self.hop_length) // 2
        audio = F.pad(audio, (p, p), "reflect").squeeze(1)
        fft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
        )
        real_part, imag_part = fft.unbind(-1)
        magnitude = torch.sqrt(real_part ** 2 + imag_part ** 2)
        mel_output = torch.matmul(self.mel_basis, magnitude)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=1e-5))
        return log_mel_spec


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class ResnetBlock2(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad2d(dilation),
            WNConv2d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv2d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)

class GeneratorMel(nn.Module):
    def __init__(self, input_size, ngf, n_residual_layers,skip_cxn=False):
        super().__init__()
        ratios = [8, 8, 2, 2]
        self.skip_cxn = skip_cxn
        self.hop_length = np.prod(ratios)
        mult = int(2 ** len(ratios))

        model = [
            nn.ReflectionPad1d(3),
            WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0),
        ]

        # Upsample to raw audio scale
        for _, r in enumerate(ratios):
            model += [
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2,
                ),
            ]

            for j in range(n_residual_layers):
                model += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]

            mult //= 2

        model += [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(ngf, 1, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)
        self.apply(weights_init)

    def forward(self, x,aud):
        imputed = center_trim(self.model(x),44100)
        if not self.skip_cxn:
            return imputed
        else:
            return imputed + aud

def center_trim(tensor, reference):
    """
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    delta = tensor.size(-1) - reference
    if delta < 0:
        raise ValueError("tensor must be larger than reference. " f"Delta is {delta}.")
    if delta:
        tensor = tensor[..., delta // 2:-(delta - delta // 2)]
    return tensor

class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf, n_layers, downsampling_factor):
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
            )

        nf = min(nf * 2, 1024)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
        )

        model["layer_%d" % (n_layers + 2)] = WNConv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        )

        self.model = model

    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


class DiscriminatorMel(nn.Module):
    def __init__(self, num_D, ndf, n_layers, downsampling_factor):
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                ndf, n_layers, downsampling_factor
            )

        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
        return results
     
class SISDRLoss(nn.Module):
    """
    Computes the Scale-Invariant Source-to-Distortion Ratio between a batch
    of estimated and reference audio signals. Used in end-to-end networks.
    This is essentially a batch PyTorch version of the function
    ``nussl.evaluation.bss_eval.scale_bss_eval`` and can be used to compute
    SI-SDR or SNR.
    Args:
        scaling (bool, optional): Whether to use scale-invariant (True) or
          signal-to-noise ratio (False). Defaults to True.
        return_scaling (bool, optional): Whether to only return the scaling
          factor that the estimate gets scaled by relative to the reference.
          This is just for monitoring this value during training, don't actually
          train with it! Defaults to False.
        reduction (str, optional): How to reduce across the batch (either 'mean', 
          'sum', or none). Defaults to 'mean'.
        zero_mean (bool, optional): Zero mean the references and estimates before
          computing the loss. Defaults to True.
        clip_min (float, optional): The minimum possible loss value. Helps network
          to not focus on making already good examples better. Defaults to None.
    """
    DEFAULT_KEYS = {'audio': 'estimates', 'source_audio': 'references'}
    def __init__(self, scaling=True, return_scaling=False, reduction='mean',
                 zero_mean=True, clip_min=None):
        self.scaling = scaling
        self.reduction = reduction
        self.zero_mean = zero_mean
        self.return_scaling = return_scaling
        self.clip_min = clip_min
        super().__init__()
    def forward(self, estimates, references):
        eps = 1e-8
        # num_batch, num_samples, num_sources
        _shape = references.shape
        references = references.reshape(-1, _shape[-2], _shape[-1]) + eps   # <---- HERE
        estimates = estimates.reshape(-1, _shape[-2], _shape[-1]) + eps   # <---- AND HERE
        # samples now on axis 1
        if self.zero_mean:
            mean_reference = references.mean(dim=1, keepdim=True)
            mean_estimate = estimates.mean(dim=1, keepdim=True)
        else:
            mean_reference = 0
            mean_estimate = 0
        _references = references - mean_reference
        _estimates = estimates - mean_estimate
        references_projection = (_references ** 2).sum(dim=-2) + eps
        references_on_estimates = (_estimates * _references).sum(dim=-2) + eps
        scale = (
            (references_on_estimates / references_projection).unsqueeze(1)
            if self.scaling else 1)
        e_true = scale * _references
        e_res = _estimates - e_true
        signal = (e_true ** 2).sum(dim=1)
        noise = (e_res ** 2).sum(dim=1)
        sdr = -10 * torch.log10(signal / noise + eps)
        if self.clip_min is not None:
            sdr = torch.clamp(sdr, min=self.clip_min)
        if self.reduction == 'mean':
            sdr = sdr.mean()
        elif self.reduction == 'sum':
            sdr = sdr.sum()
        if self.return_scaling:
            return scale
        return sdr

class GeneratorMelMix(nn.Module):
    def __init__(self, input_size, ngf, n_residual_layers,skip_cxn=False):
        super().__init__()
        ratios = [8, 8, 2, 2]
        self.skip_cxn = skip_cxn
        self.hop_length = np.prod(ratios)
        mult = int(2 ** len(ratios))
        
        self.combine_layer= nn.Conv2d(2,1,1)

        model_mix = [
            nn.ReflectionPad1d(3),
            WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0),
        ]

        # Upsample to raw audio scale
        for _, r in enumerate(ratios):
            model_mix += [
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2,
                ),
            ]

            for j in range(n_residual_layers):
                model_mix += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]

            mult //= 2
        
        model_source = [
            nn.ReflectionPad1d(3),
            WNConv1d(input_size, mult * ngf, kernel_size=7, padding=0),
        ]

        # Upsample to raw audio scale
        for _, r in enumerate(ratios):
            model_source += [
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2,
                ),
            ]

            for j in range(n_residual_layers):
                model_source += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]

            mult //= 2

        model_comb = [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(ngf, 1, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        self.model_mix = nn.Sequential(*model_mix)
        self.apply(weights_init)
        self.model_source = nn.Sequential(*model_source)
        self.apply(weights_init)
        self.model_comb = nn.Sequential(*model_comb)
        self.apply(weights_init)

    def forward(self, source, mix, aud):
        
        mix_x = self.model_mix(mix)
        source_x = self.model_source(source)
        x = self.combine_layer(torch.cat((mix_x.unsqueeze(1),source_x.unsqueeze(1)), dim=1))
        imputed = center_trim(self.model_comb(x.squeeze(1)), 44100)

        if not self.skip_cxn:
            return imputed
        else:
            return imputed + aud

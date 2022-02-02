import torch 
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import List

def disc_outputs(config, x_pred_t, x_t_1, device, netD_spec):
    D_fake_det_spec = netD_spec(
        x_pred_t.to(device).detach())
    D_real_spec = netD_spec(x_t_1.to(device))
    return D_fake_det_spec, D_real_spec

def Gen_loss(D_fake, D_fake_spec):
    loss_G = 0
    for scale in D_fake:
        loss_G += -scale[-1].mean()

    loss_G += -3*D_fake_spec[-1].mean()
    return loss_G

def waveform_discriminator_loss(D_fake, D_real):
    loss_D = 0
    for scale in D_fake:
        loss_D += F.relu(1 + scale[-1]).mean()
    for scale in D_real:
        loss_D += F.relu(1 - scale[-1]).mean()
    return loss_D


def spectral_discriminator_loss(fake, real):
    loss_D_spec = 0
    loss_D_spec += F.relu(1 + fake[-1]).mean()

    loss_D_spec += F.relu(1 - real[-1]).mean()
    return loss_D_spec

def feature_loss(config, D_fake, D_real, D_fake_spec, D_real_spec):
    loss_feat = 0
    feat_weights = 4.0 / (config.n_layers_D + 1)
    D_weights = 1.0 / config.num_D
    wt = D_weights * feat_weights
    for i in range(config.num_D):
        for j in range(len(D_fake[i]) - 1):
            loss_feat += wt * F.l1_loss(D_fake[i][j],
                                        D_real[i][j].detach())

    wt = 4.0 / (config.n_layers_D_spec + 1)
    loss_feat_spec = 0
    for k in range(len(D_fake_spec) - 1):
        loss_feat_spec += wt * F.l1_loss(D_fake_spec[k],
                                         D_real_spec[k].detach())
    return loss_feat, loss_feat_spec

def mel_spec_loss(target,estimated):
    eps = 1e-5

    target_spec = torch.stft(input=target, nfft=1024)
    real_part, imag_part = target_spec.unbind(-1)
    target_mag_spec = torch.log10(torch.sqrt(real_part**2 + imag_part**2 + eps))
    
    estimated_spec = torch.stft(input=estimated, nfft=1024)
    real_part, imag_part = estimated_spec.unbind(-1)
    estimated_mag_spec = torch.log10(torch.sqrt(real_part**2 + imag_part**2 +eps))

    return F.l1_loss(target_mag_spec,estimated_mag_spec)

class AutoBalance(nn.Module):
    def __init__(
        self, ratios: List[float] = [1], frequency: int = 1, max_iters: int = None
    ):
        """
        Auto-balances losses with each other by solving a system of
        equations.
        """
        super().__init__()

        self.frequency = frequency
        self.iters = 0
        self.max_iters = max_iters
        self.weights = [1 for _ in range(len(ratios))]

        # Set up the problem
        ratios = torch.from_numpy(np.array(ratios))

        n_losses = ratios.shape[0]

        off_diagonal = torch.eye(n_losses) - 1
        diagonal = (n_losses - 1) * torch.eye(n_losses)

        A = off_diagonal + diagonal
        B = torch.zeros(1 + n_losses)
        B[-1] = 1

        W = 1 / ratios

        self.register_buffer("A", A.float())
        self.register_buffer("B", B.float())
        self.register_buffer("W", W.float())
        self.ratios = ratios

    def __call__(self, *loss_vals):
        exceeded_iters = False
        if self.max_iters is not None:
            exceeded_iters = self.iters >= self.max_iters

        with torch.no_grad():
            if self.iters % self.frequency == 0 and not exceeded_iters:
                num_losses = self.ratios.shape[-1]
                L = self.W.new(loss_vals[:num_losses])
                if L[L > 0].shape == L.shape:
                    _A = self.A * L * self.W
                    _A = torch.vstack([_A, torch.ones_like(self.W)])

                    # Solve with least squares for weights so each
                    # loss function matches what is given in ratios.
                    X = torch.linalg.lstsq(_A.float(), self.B.float(), rcond=None)[0]

                    self.weights = X.tolist()

        self.iters += 1
        return [w * l for w, l in zip(self.weights, loss_vals)]

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

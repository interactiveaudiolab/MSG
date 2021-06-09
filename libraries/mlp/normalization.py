import torch
import numpy as np

from . import complx

STD_MAG = 2.4214
MEAN_MAG = 0.4385

STD_PH = 1.8063
MEAN_PH = -0.0003


def logit(p):
    return torch.log(p/(1-p))


def norm_mag(mag):
    mag_log = torch.log(mag)
    mag_norm = (mag_log - MEAN_MAG)/STD_MAG
    
    return mag_norm


def denorm_mag(mag_norm):
    mag_log = mag_norm * STD_MAG + MEAN_MAG
    mag = torch.exp(mag_log)
    
    return mag


def norm_phase(ph):
    ph = ph/(2 * np.pi + 1e-3) + 0.5
    ph = logit(ph)
    ph_norm = (ph - MEAN_PH)/STD_PH

    return ph_norm


def denorm_phase(ph_norm):
    ph = ph_norm * STD_PH + MEAN_PH
    ph = torch.sigmoid(ph)
    ph = (ph - 0.5) * (2 * np.pi + 1e-3)
    
    return ph


def denorm_polar(out):
    mag, phase = denorm_mag(out[0]), denorm_phase(out[1])
    real, im = complx.to_complex(mag, phase)
    
    return torch.stack([real, im], dim=0)

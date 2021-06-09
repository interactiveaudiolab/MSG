import torch
import numpy as np


def to_polar(complx):
    
    magnitude = (complx[0] ** 2 + complx[1] ** 2)**0.5
    # magnitude = torch.norm(complx, dim=0)
    phase = torch.atan(complx[1] / complx[0])
    
    phase[complx[0] < 0] = phase[complx[0] < 0] + np.pi
    phase = phase - np.pi/2

    return magnitude, phase
    
    
def to_complex(magnitude, phase):
    real = magnitude * torch.cos(phase)
    im = magnitude * torch.sin(phase)
    
    return real, im
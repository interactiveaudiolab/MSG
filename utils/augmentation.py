import torch 
import numpy as np

def shift_phase(waveform,factor):
    stft_data = torch.stft(waveform, n_fft=1024,return_complex=True)
    magnitude = torch.abs(stft_data)
    phase = torch.angle(stft_data)
    new_phase = phase + factor
    stft_data = magnitude * torch.exp(1j*new_phase)
    new_waveform = torch.istft(stft_data, n_fft=1024)
    return new_waveform
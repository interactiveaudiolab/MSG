import scipy.io.wavfile as wav
import scipy.signal as sig

import matplotlib.pyplot as plt
import numpy as np

import IPython.display as ipd

from tqdm import tqdm

def read_monaural_wav(path):
    fs, audio_time = wav.read(path)
    
    
    if len(audio_time.shape) == 2:
        return fs, audio_time[:,0]
    else:
        return fs, audio_time
    
    
def stft(audio, fs, nperseg=256):
    
    
    return sig.stft(audio, fs, nperseg=nperseg)
    
    
def istft(freqs, fs, nperseg=256):
    return sig.istft(freqs, fs, nperseg=nperseg)
    
    
def play_audio(data, rate=None):
    return ipd.Audio(data, rate=rate)


def show_spectra(freq_data, times=None, freqs=None, vmax=300):
    if times is None:
        times = range(freq_data.shape[1])
    
    if freqs is None:
        freqs = range(freq_data.shape[0])
    
    return plt.pcolormesh(times, freqs, np.abs(freq_data), vmin=0, vmax=vmax)


def cutout_slient(freqs, cutoff=25, min_width=128):
    mags = np.abs(freqs).mean(axis=0)
    start = 0
    end = -1
    
    while mags[start] < cutoff:
        start += 1
        
        if start > len(mags) - min_width:
            return None
        
    while mags[end] < cutoff:
        end -= 1
        
        if len(mags) + end < start + min_width:
            return None
            
    return freqs[:,start:end]
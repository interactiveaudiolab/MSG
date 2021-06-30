import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import pickle
from multiprocessing import Pool
import resource
from tqdm import tqdm
import nussl
import librosa
from scipy.spatial.distance import cosine

import sys


# Path to Pix2Pix directory
local_path = '/drive/MelGan-Imputation/'

# Path to libraries directory
sys.path.append('/drive/MelGan-Imputation/libraries')

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from models.MelGAN import *
from models.unet import *
from mlp import audio
from mlp import normalization
from mlp import utils as mlp
from mlp.WaveDataset import MusicDataset


n_mel_channels = 80
ngf = 32
n_residual_layers = 3

num_D = 3
ndf = 16
n_layers_D = 4
downsamp_factor = 4
lambda_feat = 10
save_interval = 20
log_interval = 100

G1 = GeneratorMel(n_mel_channels, ngf, n_residual_layers)
G1.load_state_dict(torch.load("2999netG.pt"))


clean_path ='/content/drive/MyDrive/Pix2Pix/original_test_sources'
dirty_path ='/content/drive/MyDrive/Pix2Pix/demucs_test_separated_flat'

test_dirty = []
test_clean = []

for s in os.listdir(clean_path):
  if 'drums' in s:
      test_dirty.append(dirty_path + '/' + s)
      test_clean.append(clean_path +'/' + s)


ds_test = MusicDataset(test_dirty,test_clean,44100,44100)

G1.to('cuda')
G1.eval()


start = 7


fft = Audio2Mel(n_mel_channels=n_mel_channels).cuda()



si_sdr_noisy = []
si_sdr_generated = []

sd_sdr_noisy = []
sd_sdr_generated = []

si_sar_noisy = []
si_sar_generated = []

si_sir_noisy = []
si_sir_generated = []

snr_noisy = []
snr_generated = []

noisy_cosine = []
generated_cosine = []

cos = nn.CosineSimilarity()

with torch.no_grad():
  for start in range(50):
    clean1 = np.array([])
    noisy1 = np.array([])
    aud1 = np.array([])
    for i in range(7*start, 7*start+7):
      n,c = ds_test[i]
      clean1 = np.concatenate((clean1,c))
      noisy1 = np.concatenate((noisy1,n))


      s_t = fft(torch.from_numpy(n).float().unsqueeze(0).unsqueeze(0).cuda()).detach()
      x_pred_t = G1(s_t.cuda(),torch.from_numpy(n).cuda())

      a = x_pred_t.squeeze().squeeze().detach().cpu().numpy()
      aud1 = np.concatenate((aud1,a))
    
    c = nussl.AudioSignal(audio_data_array=clean1)
    n = nussl.AudioSignal(audio_data_array=noisy1)
    g = nussl.AudioSignal(audio_data_array=aud1)
    bss_eval = nussl.evaluation.BSSEvalScale(
    true_sources_list=[c],
    estimated_sources_list=[n]
    )
    
    noisy = librosa.feature.melspectrogram(y=noisy1, sr=44100, n_mels=128,
                                    fmax=8000)
    clean = librosa.feature.melspectrogram(y=clean1, sr=44100, n_mels=128,
                                    fmax=8000)
    generated = librosa.feature.melspectrogram(y=aud1, sr=44100, n_mels=128,
                                    fmax=8000)
  
    noisy_cosine.append(cosine(clean.flatten(),noisy.flatten()))
    generated_cosine.append(cosine(clean.flatten(),generated.flatten()))



    noisy_eval = bss_eval.evaluate()

    bss_eval = nussl.evaluation.BSSEvalScale(
        true_sources_list=[c],
        estimated_sources_list=[g]
    )
    gen_eval = bss_eval.evaluate()

    si_sdr_noisy.append(noisy_eval['source_0']['SI-SDR'])
    si_sdr_generated.append(gen_eval['source_0']['SI-SDR'])
    sd_sdr_noisy.append(noisy_eval['source_0']['SD-SDR'])
    sd_sdr_generated.append(gen_eval['source_0']['SD-SDR'])
    si_sar_noisy.append(noisy_eval['source_0']['SI-SAR'])
    si_sar_generated.append(gen_eval['source_0']['SI-SAR'])
    si_sir_noisy.append(noisy_eval['source_0']['SI-SIR'])
    si_sir_generated.append(gen_eval['source_0']['SI-SIR'])
    snr_noisy.append(noisy_eval['source_0']['SNR'])
    snr_generated.append(gen_eval['source_0']['SNR'])
print('Original SI-SDR', np.mean(si_sdr_noisy))
print('Our SI-SDR', np.mean(si_sdr_generated))
print('\nOriginal SD-SDR', np.mean(sd_sdr_noisy))
print('Our SD-SDR', np.mean(sd_sdr_generated))
print('\nOriginal SI-SAR', np.mean(si_sar_noisy))
print('Our SI-SAR', np.mean(si_sar_generated))
print('\nOriginal SI-SIR', np.mean(si_sir_noisy))
print('Our SI-SIR', np.mean(si_sir_generated))
print('\nOriginal SNR', np.mean(snr_noisy))
print('Our SNR', np.mean(snr_generated))
print('\nDemucs Mean Spectral Cosine Distance', np.mean(noisy_cosine))
print('MSG Mean Spectral Cosine Distance', np.mean(generated_cosine))
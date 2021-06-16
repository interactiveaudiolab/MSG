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
from IPython.display import Audio


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
G1.load_state_dict(torch.load("[Replace with path to generator]"))

def _add_zero_padding(signal, window_length, hop_length):
    """
    Args:
        signal:
        window_length:
        hop_length:
    Returns:
    """
    original_signal_length = len(signal)
    overlap = window_length - hop_length
    num_blocks = np.ceil(len(signal) / hop_length)

    if overlap >= hop_length:  # Hop is less than 50% of window length
        overlap_hop_ratio = np.ceil(overlap / hop_length)

        before = int(overlap_hop_ratio * hop_length)
        after = int((num_blocks * hop_length + overlap) - original_signal_length)

        signal = np.pad(signal, (before, after), 'constant', constant_values=(0, 0))
        extra = overlap

    else:
        after = int((num_blocks * hop_length + overlap) - original_signal_length)
        signal = np.pad(signal, (hop_length, after), 'constant', constant_values=(0, 0))
        extra = window_length

    num_blocks = int(np.ceil((len(signal) - extra) / hop_length))
    num_blocks += 1 if overlap == 0 else 0  # if no overlap, then we need to get another hop at the end

    return signal, num_blocks



clean_path = '[replace with path to clean test set]'
dirty_path ='[replace with path to dirty test set]'

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

clean1 = np.array([])
noisy1 = np.array([])
aud1 = np.array([])

sdr_noisy = []
sdr_generated = []

with torch.no_grad():
  for start in range(50):
    for i in range(7*start, 7*start+7):
      n,c = ds_test[i]
      clean1 = np.concatenate((clean1,c))
      noisy1 = np.concatenate((noisy1,n))

      n_pad = _add_zero_padding(n, 1024,256)[0]

      s_t = fft(torch.from_numpy(n_pad).float().unsqueeze(0).unsqueeze(0).cuda()).detach()
      x_pred_t = G1(s_t.cuda())

      a = x_pred_t.squeeze().squeeze().detach().cpu().numpy()
      aud1 = np.concatenate((aud1,a[0:44100]))
    c = nussl.AudioSignal(audio_data_array=clean1)
    n = nussl.AudioSignal(audio_data_array=noisy1)
    g = nussl.AudioSignal(audio_data_array=aud1)
    bss_eval = nussl.evaluation.BSSEvalScale(
    true_sources_list=[c],
    estimated_sources_list=[n]
    )

    noisy_eval = bss_eval.evaluate()

    bss_eval = nussl.evaluation.BSSEvalScale(
        true_sources_list=[c],
        estimated_sources_list=[g]
    )
    gen_eval = bss_eval.evaluate()

    sdr_noisy.append(noisy_eval['source_0']['SI-SDR'])
    sdr_generated.append(gen_eval['source_0']['SI-SDR'])
print('Original SDR', np.mean(sdr_noisy))
print('Our SDR', np.mean(sdr_generated))
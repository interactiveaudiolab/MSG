import scipy.io.wavfile as wav
import scipy.signal as sig
import matplotlib.pyplot as plt
import numpy as np

import IPython.display as ipd

import numpy as np
from numpy.linalg import norm
from spectrum import arburg
from spectrum.burg import _arburg2
from scipy.signal import lfilter, lfiltic
from tqdm import tqdm

from mlp import audio
from mlp import normalization
from mlp import utils as mlp
from mlp.dataset import WAVAudioDS, TimePostprocessing, WAVTimeAudioDS, PolarPreprocessing
from multiprocessing import Pool
import multiprocessing
import pickle
import torch


def LPC(previous_sig, next_sig, gap_start, gap_end, lpc_order):

    target_length = gap_end - gap_start
    
    ab, _, _ = _arburg2(previous_sig, lpc_order)
    Zb = lfiltic(b=[1], a=ab, y=previous_sig[:-lpc_order-1:-1])
    forw_pred, _ = lfilter(b=[1], a=ab, x=np.zeros((target_length)), zi=Zb)

    next_sig = np.flipud(next_sig)
    af, _, _ = _arburg2(next_sig, lpc_order)
    Zf = lfiltic([1], af, next_sig[:-lpc_order-1:-1])
    backw_pred, _ = lfilter([1], af, np.zeros((target_length)), zi=Zf)
    backw_pred = np.flipud(backw_pred)

    t = np.linspace(0, np.pi/2, target_length)
    sqCos = np.cos(t)**2
    sigout = sqCos*forw_pred + np.flipud(sqCos)*backw_pred
    return sigout


class LPCPipeline:
    def __init__(self, context_size, fs, ms, lpc_order):
        self.context_size = context_size
        self.fs = fs
        self.gap_size = int(np.floor(ms * fs / 1000))
        self.lpc_order = 1000
        
    def __call__(self, src_and_target):
        _, target = src_and_target
        sample_length = len(target)
        gap_start = sample_length // 2 - self.gap_size // 2
        gap_end = gap_start + self.gap_size

        previous_sig = target[gap_start-self.context_size:gap_start]
        gap_sig = target[gap_start:gap_end]
        next_sig = target[gap_end:gap_end+self.context_size]
        
        if len(previous_sig) == 0 or len(next_sig) == 0:
            return None
        
        lpc_result = LPC(previous_sig, next_sig, gap_start, gap_end, self.lpc_order)
        PBAR.update(1)
        
        return lpc_result
    
if __name__=="__main__": 
    
    fs = 48000
    bs = 1
    stroke_width_ms = 32
    patch_width = 64
    patch_height = 64
    
    preprocess = PolarPreprocessing(
        normalization.norm_mag, 
        normalization.norm_phase, 
        patch_width, 
        patch_height
    )

    time_mask = mlp.build_time_purge_mask(patch_width, stroke_width_ms, fs)

    val_files = pickle.load(open("valid.pk", "rb"))

    torch.multiprocessing.set_sharing_strategy('file_system') # I was getting memory errors without this line.
    with Pool(8) as p:
        ds_valid = WAVTimeAudioDS(files=val_files, mk_source=lambda x: x * time_mask, preprocess=preprocess, 
                              patch_width=patch_width, proc_pool=p, fs=fs, random_patches=False)
    
    PBAR = tqdm(total=77928)

    context_size = int(69 * fs / 1000)
    lpc_order = 1000
    chunksize = 250
    
    with Pool(6) as pool:    
        rec_signals = pool.map(LPCPipeline(context_size, fs, ms=32, lpc_order=lpc_order), ds_valid, chunksize=chunksize)
import os
import math
import operator as op

import torch

from itertools import accumulate
from functools import reduce
import numpy as np

def get_all_files(root):
    if os.path.isfile(root):
        return [root]
    else:
        return reduce(op.add, map(lambda c: get_all_files(root + "/" + c), os.listdir(root)), [])

    
def build_stroke_purge_mask(patch_width, patch_height, ms, fs, nperseg=256, channels=2):
    pixels = math.floor(ms * (1 / 500) * (fs / nperseg))
    left_offset = patch_width // 2 - pixels // 2
    
    mask = torch.ones((channels, patch_height, patch_width), dtype=torch.uint8)
    mask[:,:, left_offset:left_offset + pixels] = 0
    
    return mask

def build_time_purge_mask(patch_width, ms, fs):
    px_to_ms = lambda px: round(px * 2.667)
    
    sample_ms = math.floor(ms * fs / 1000)
    sample_length = math.floor(px_to_ms(patch_width) * fs / 1000)
    gap_start = sample_length // 2 - sample_ms // 2
    
    mask = np.ones(sample_length)
    mask[gap_start:gap_start + sample_ms] = 0
    
    return mask

def acc_to_idx(acc, value):
    for i, v in enumerate(acc):
        if value < v:
            return i
        
    return len(acc) - 1
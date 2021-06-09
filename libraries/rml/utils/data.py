import torch

import numpy as np

from typing import List
from torch.utils.data import DataLoader
from itertools import accumulate

from . import utils


class PreLoader:
    def __init__(self, dl: DataLoader, device=utils.device):
        xs, ys = [], []

        for i, data in enumerate(dl):
            xs.append(data[0].to(device))
            ys.append(data[1].to(device))

        self.input = xs
        self.target = ys

    def __iter__(self):
        order = np.random.permutation(len(self))

        for i in range(len(self)):
            yield self.input[order[i]], self.target[order[i]]

    def __len__(self):
        return len(self.input)


class SimpleDS:
    def __init__(self, x, y, transform = lambda x:x):
        self.x = x
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx: int):
        return self.transform(self.x[idx]), self.y[idx]
    

def random_splits(sizes: List[int], *seqs):
    data_size = len(seqs[0])
    total_splits = sum(sizes)
    
    idxs = np.random.permutation(data_size)

    slices = list(map(lambda i: int((i / total_splits) * data_size), sizes)) 
    slices = list(accumulate(slices))
    slices = list(zip([0] + slices[:-1], slices[:-1] + [-1]))    
    slices = list(map(lambda i: slice(i[0], i[1]), slices))
                
    return list(map(lambda s: [seq[idxs[s]] for seq in seqs], slices))

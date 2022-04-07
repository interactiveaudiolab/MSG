import nussl
from torch.utils.data import Dataset
import numpy as np

class DatasetWrapper(Dataset):
    def __init__(self, path_to_ds, target,mono, gt_percent, silent_percent, mix_percent, **kwargs):
        # put these in kwargs: mix_folder='mixture',
        # sample_rate=44_100, segment_dur=1.0
        self.target = target
        self.mono = mono
        self.ds = nussl.datasets.SalientExcerptMixSourceFolder(path_to_ds,
                                                               target,
                                                               **kwargs)
        self.gt_percent = gt_percent if gt_percent < 1 else gt_percent/100
        self.silent_percent = silent_percent+self.gt_percent if silent_percent < 1 else silent_percent/100+self.gt_percent
        self.mix_percent = mix_percent+self.silent_percent if mix_percent < 1 else mix_percent/100+self.silent_percent

    def __getitem__(self, item):
        current_item = self.ds[item]
        #index_of_clean = current_item['metadata']['labels'].index(self.target)
        #index_of_dirty = current_item['metadata']['labels'].index('dirty_'+self.target)
        
        if self.mono:
            clean_sample = current_item['sources'][self.target].to_mono().\
            audio_data.squeeze()
            dirty_sample = current_item['sources']['dirty_'+self.target].to_mono().\
            audio_data.squeeze()
            mix = current_item['mix'].to_mono().audio_data.squeeze()
        else:
            clean_sample = current_item['sources'][self.target].audio_data.squeeze()
            dirty_sample = current_item['sources']['dirty_'+self.target].audio_data.squeeze()
            mix = current_item['mix'].audio_data.squeeze()
        rand_transform = np.random.uniform()
        if rand_transform < self.gt_percent:
            dirty_sample = clean_sample.copy()
        elif rand_transform < self.silent_percent:
            dirty_sample = np.zeros_like(dirty_sample)
        elif rand_transform < self.mix_percent:
            dirty_sample = mix.copy()
        return dirty_sample, clean_sample, mix

    def __len__(self):
        return len(self.ds)

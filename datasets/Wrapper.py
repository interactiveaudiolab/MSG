import nussl
from torch.utils.data import Dataset


class DatasetWrapper(Dataset):
    def __init__(self, path_to_ds, target, **kwargs):
        # put these in kwargs: mix_folder='mixture',
        # sample_rate=44_100, segment_dur=1.0
        self.target = target
        self.ds = nussl.datasets.SalientExcerptMixSourceFolder(path_to_ds,
                                                               target,
                                                               **kwargs)

    def __getitem__(self, item):
        current_item = self.ds[item]
        #index_of_clean = current_item['metadata']['labels'].index(self.target)
        #index_of_dirty = current_item['metadata']['labels'].index('dirty_'+self.target)

        clean_sample = current_item['sources'][self.target].to_mono().\
            audio_data.squeeze()
        dirty_sample = current_item['sources']['dirty_'+self.target].to_mono().\
            audio_data.squeeze()

        return dirty_sample, clean_sample

    def __len__(self):
        return len(self.ds)

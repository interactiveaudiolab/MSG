from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import os
import pickle


class MusicDataset(Dataset):
    def __init__(self,  filenames_dirty, filenames_clean, sr, sample_length):
        """
        A dataset that takes the filenames as input and loads a sample from the list
        :param filenames:
        :param spec_type:
        """
        self.filenames_clean = filenames_clean
        self.filenames_dirty = filenames_dirty
        self.left_overs = 0
        self.sr = sr
        self.sample_length = sample_length
        # We need to know exactly how many items per spectrogram and calc which spec would be retrieved by which idx
        # Here is my current thought process, iterate over each file in the set, load it, convert it to spectral
        # representation, map the number of possible datapoints to a idx to the total number of datapoints and its file
        # position (or something like that), store the map. Then at retrieval of length, give the max idx of the map.
        # At item retrieval access map at idx to get file and position, then return the mel spec.

        # 1. figure out how many stft we have per audio sample
        # 2. map input index i -> stft grouping j, stft k


        # given index maps to filename index
        # self._index_mapping = [(i // self.STFT_PER_SAMPLE, i % self.STFT_PER_SAMPLE)
        #                        for i in range(self.STFT_PER_SAMPLE * len(filenames_clean))]
        self._file_sizes = self._fileSize()
        num_chunks = int(np.ceil(self._file_sizes/self.sample_length))
        self._index_mapping = [(i // num_chunks,
                                i % num_chunks)
                               for i in range((num_chunks) * len(filenames_clean))]
        self._validate()

    def _validate(self):
        assert len(self.filenames_dirty) == len(self.filenames_clean), "Clean and dirty directories must have the same" \
                                                                       "number of items."
        for i in range(len(self.filenames_dirty)):
            assert self.filenames_dirty[i].split('/')[-1] == self.filenames_clean[i].split('/')[-1],\
                "Clean and dirty files must be associated by name."
        print("Dataset Validation Successful")

    def _fileSize(self):
        infile = open(self.filenames_clean[0],'rb')
        data = pickle.load(infile)
        infile.close()
        return data.shape[0]

    def __len__(self):
        # return the max of the keys in the mapping
        return len(self._index_mapping)

    def __getitem__(self, idx):
        file_idx, wave_idx = self._index_mapping[idx]
        clean_filename = self.filenames_clean[file_idx]
        dirty_filename = self.filenames_dirty[file_idx]

        infile = open(dirty_filename,'rb')
        data = pickle.load(infile)
        infile.close()

        dirty_sample = data[wave_idx * self.sample_length:(wave_idx + 1) * self.sample_length]

        infile = open(clean_filename,'rb')
        data = pickle.load(infile)
        infile.close()
        
        clean_sample = data[wave_idx * self.sample_length:(wave_idx + 1) * self.sample_length]

        if clean_sample.shape[0] < self.sample_length:
            clean_sample = np.concatenate((clean_sample, np.zeros(self.sample_length-clean_sample.shape[0])))
            dirty_sample = np.concatenate((dirty_sample, np.zeros(self.sample_length-dirty_sample.shape[0])))

        # dirty_sample = np.abs(librosa.stft(dirty_sample)) ** 2
        # dirty_sample = librosa.feature.melspectrogram(S=dirty_sample, sr=sr)
        return dirty_sample, clean_sample


if __name__ == '__main__':
    import os

    myset = MusicDataset(["Dataset/test/labels/" + elem for elem in os.listdir("Dataset/test/labels")],
                         ["Dataset/test/data/" + elem for elem in os.listdir("Dataset/test/data")], 44100, 44100)
    my_loader = DataLoader(myset)
    for item in my_loader:
        print(item)




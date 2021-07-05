from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np


class MusicDataset(Dataset):
    def __init__(self, filenames_clean, filenames_dirty, spec_type, spec_width):
        """
        A dataset that takes the filenames as input and loads a sample from the list
        :param filenames:
        :param spec_type:
        """
        self.filenames_clean = filenames_clean
        self.filenames_dirty = filenames_dirty
        self.left_overs = 0
        self.spec_type = spec_type
        self.spec_width = spec_width
        # We need to know exactly how many items per spectrogram and calc which spec would be retrieved by which idx
        # Here is my current thought process, iterate over each file in the set, load it, convert it to spectral
        # representation, map the number of possible datapoints to a idx to the total number of datapoints and its file
        # position (or something like that), store the map. Then at retrieval of length, give the max idx of the map.
        # At item retrieval access map at idx to get file and position, then return the mel spec.

        # 1. figure out how many stft we have per audio sample
        # 2. map input index i -> stft grouping j, stft k

        # Brute force approach, assume fixed length:
        self.STFT_PER_SAMPLE = 587//spec_width+1
        # given index maps to filename index
        self._index_mapping = [(i // self.STFT_PER_SAMPLE, i % self.STFT_PER_SAMPLE)
                               for i in range(self.STFT_PER_SAMPLE * len(filenames_clean))]
        self._validate()

    def _validate(self):
        assert len(self.filenames_dirty) == len(self.filenames_clean), "Clean and dirty directories must have the same" \
                                                                       "number of items."
        for i in range(len(self.filenames_dirty)):
            assert self.filenames_dirty[i].split('/')[-1] == self.filenames_clean[i].split('/')[-1],\
                "Clean and dirty files must be associated by name."
        print("Dataset Validation Successful")


    def __len__(self):
        # return the max of the keys in the mapping
        return len(self._index_mapping)

    def __getitem__(self, idx):
        file_idx, stft_idx = self._index_mapping[idx]
        clean_filename = self.filenames_clean[file_idx]
        dirty_filename = self.filenames_dirty[file_idx]

        data, sr = librosa.load(clean_filename, sr=44100)
        if self.spec_type == "mel":
            # data = librosa.feature.melspectrogram(data, sr=sr, n_mels=1024)
            data = np.abs(librosa.stft(data))**2
            data = librosa.feature.melspectrogram(S=data, sr=sr)
        else:
            data = librosa.stft(data)
        if stft_idx == self.STFT_PER_SAMPLE-1:
            left_over = self.spec_width - data.shape[1] % self.spec_width
            clean_sample = data[:, stft_idx * self.spec_width - left_over:(stft_idx + 1) * self.spec_width]
        else:
            clean_sample = data[:, stft_idx * self.spec_width:(stft_idx + 1) * self.spec_width]

        data, sr = librosa.load(dirty_filename, sr=44100)
        if self.spec_type == "mel":
            # data = librosa.feature.melspectrogram(data, sr=sr, n_mels=1024)
            data = np.abs(librosa.stft(data))**2
            data = librosa.feature.melspectrogram(S=data, sr=sr)
        else:
            data = librosa.stft(data)
        if stft_idx == self.STFT_PER_SAMPLE-1:
            left_over = self.spec_width - data.shape[1] % self.spec_width
            dirty_sample = data[:, stft_idx * self.spec_width - left_over:(stft_idx + 1) * self.spec_width]
        else:
            dirty_sample = data[:, stft_idx * self.spec_width:(stft_idx + 1) * self.spec_width]

        return dirty_sample, clean_sample

    def getFullSongSTFT(self, idx):
        #file_idx, stft_idx = self._index_mapping[idx]
        clean_filename = self.filenames_clean[idx]
        dirty_filename = self.filenames_dirty[idx]
        clean_samples = []
        dirty_samples = []

        # load the clean song
        clean_song_data, sr = librosa.load(clean_filename, sr=44100)
        if self.spec_type == "mel":
            # data = librosa.feature.melspectrogram(data, sr=sr, n_mels=1024)
            stft_data = librosa.stft(clean_song_data)
            clean_phase = np.angle(stft_data)
            clean_song_data = np.abs(stft_data) ** 2
            clean_song_data = librosa.feature.melspectrogram(S=clean_song_data, sr=sr)
        else:
            clean_song_data = librosa.stft(clean_song_data)
            clean_phase = np.angle(clean_song_data)

        # load the dirty dirty cong
        dirty_song_data, sr = librosa.load(dirty_filename, sr=44100)
        if self.spec_type == "mel":
            # data = librosa.feature.melspectrogram(data, sr=sr, n_mels=1024)
            stft_data = librosa.stft(dirty_song_data)
            dirty_phase = np.angle(stft_data)
            dirty_song_data = np.abs(stft_data) ** 2
            dirty_song_data = librosa.feature.melspectrogram(S=dirty_song_data, sr=sr)
        else:
            dirty_song_data = librosa.stft(dirty_song_data)
            dirty_phase = np.angle(dirty_song_data)
        self.left_overs = self.spec_width - clean_song_data.shape[1] % self.spec_width
        # for each stft in the song compute extract a snippet of the spectrogram and store it in the return values
        for i in range(self.STFT_PER_SAMPLE):
            if i == self.STFT_PER_SAMPLE - 1:
                clean_sample = clean_song_data[:, i * self.spec_width - self.left_overs:(i + 1) * self.spec_width]
                dirty_sample = dirty_song_data[:, i * self.spec_width - self.left_overs:(i + 1) * self.spec_width]
            else:
                clean_sample = clean_song_data[:, i * self.spec_width:(i + 1) * self.spec_width]
                dirty_sample = dirty_song_data[:, i * self.spec_width:(i + 1) * self.spec_width]

            clean_samples.append(clean_sample)
            dirty_samples.append(dirty_sample)
        return dirty_samples, clean_samples, dirty_phase, clean_phase

    def reconstruct_stft(self, stft_array):
        reconstruction = np.array(stft_array[0])
        for i in range(1, len(stft_array)-1):
            reconstruction = np.concatenate((reconstruction, stft_array[i]), axis=1)
        reconstruction = np.concatenate((reconstruction, stft_array[-1][:, -(self.spec_width-self.left_overs):]), axis=1)
        return reconstruction


if __name__ == '__main__':
    import os

    myset = MusicDataset(["Dataset/test/labels/" + elem for elem in os.listdir("Dataset/test/labels")],
                         ["Dataset/test/data/" + elem for elem in os.listdir("Dataset/test/data")], 'mel', 600)
    my_loader = DataLoader(myset)
    for item in my_loader:
        print(item)

    data = myset.getFullSongSTFT(0)
    data = myset.reconstruct_stft(data[1])
    data = librosa.feature.inverse.mel_to_audio(data, sr=44100)
    print()


from torch.utils.data import Dataset
import librosa
import os

class EvaluationSet(Dataset):
    def __init__(self, songs_dir,sample_rate,source):
        self.noisy_path = songs_dir + f'dirty_{source}'
        self.clean_path = songs_dir + source
        self.sample_rate = sample_rate

    def __len__(self):
        return len(os.listdir(self.song_path))

    def __getitem__(self, idx):
        noisy,sr = librosa.load(self.noisy_path + '/' + os.listdir(self.noisy_path)[idx],sr=self.sample_rate)
        clean,sr = librosa.load(self.clean_path + '/' +os.listdir(self.noisy_path)[idx],sr=self.sample_rate)
        return noisy, clean, os.listdir(self.noisy_path)[idx]
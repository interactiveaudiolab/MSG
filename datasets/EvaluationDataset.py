from torch.utils.data import Dataset
import os
import librosa
from typing import List, Tuple, Union


class EvalSet(Dataset):
    def __init__(self, dataset_path: str, item_length: float = 1,
                 sample_rate: int = 44_100, sources: tuple = ('drums',),
                 as_dict=False):
        """
        Note: when returning as a list the order will be in the same order as
        the sources with the mixture at the end.

        Load a dataset for validation or testing. Audio will be returned as
        dictionaries and will include the mixture and dirty source as well as
        the clean source.
        Assumes the following set directory structure:
        Path/
            mixture/
            source_1/
            source_2/
            source_3/
            source_4/
            etc ...
        Sources must be of the same length as the mixture.
        :param dataset_path: Path to the set. e.g:
            /media/CHONK/data/imputation/DemucsDataset/SetA/Valid
        :param item_length: duration measured in seconds
        :param sample_rate: sample rate of the audio
        :param sources: sources to include in the return value.
            e.g: (drums, vocals, bass, other)
        """
        self.dataset_path = dataset_path if dataset_path.endswith(os.sep) \
            else dataset_path + os.sep
        self.metadata, self.song_starts = self.create_metadata()
        self.item_length = item_length
        self.sample_rate = sample_rate
        self.sources = list(sources) + ["mixture"]
        self.as_dict = as_dict

    def create_metadata(self) -> Tuple[List[dict], List[tuple]]:
        """
        Populate the song metadata so that the getitem function can index songs
        using integer values
        :return:
        """
        metadata = []
        song_starts = []
        items = os.listdir(self.dataset_path)
        target_duration = self.item_length
        for clip in items:
            array, sr = librosa.load(self.dataset_path + "mixture/" + clip,
                                     sr=self.sample_rate)
            starts = [
                {
                    "start": target_duration * i,
                    "filename": clip,
                    "pad": target_duration * (i + 1) >
                           (array.shape[0] / self.sample_rate)
                }
                for i in range(array.shape[0] // target_duration)
            ]
            metadata += starts
            song_starts.append((len(metadata)-len(starts), len(metadata)-1))
        return metadata, song_starts

    def get_song_indices(self) -> List[tuple]:
        """
        returns a list of tuples containing each songs starting and ending
        indices. For example 3 songs of length 10s, 14s, and 5s with 1s clips
        would be stored as:
        [(0,9), (10,23), (24,28)]
        This list can be used to easily index the object for specific songs
        during manual evaluation.
        :return: list of song start and end indices.
        """
        return self.song_starts

    def __getitem__(self, item: int) -> Union[dict, list]:
        """
        Takes a integer and returns a dictionary of the following structure if
        the as_dict flag is passed:
        {
            mixture: np_array,
            source_1: np_array,
            source_2: np_array,
            ...
            last_clip
        }
        Otherwise it returns the list:
        [
        source_1: np_array
        source_2: np_array
        ...
        mixture: np_array
        ]
        :param item: integer value for the item
        :return: list or dict of desired data
        """
        current_item: dict = self.metadata[item]
        return_value = {} if self.as_dict else []
        for source in self.sources:
            if current_item["pad"]:
                array, sr = librosa.load(
                    path=self.dataset_path + source + os.sep +
                         current_item["filename"],
                    sr=self.sample_rate,
                    offset=current_item["start"])
                array = array.resize(self.sample_rate * self.item_length)
            else:
                array, sr = librosa.load(
                    path=self.dataset_path + source + os.sep +
                         current_item["filename"],
                    sr=self.sample_rate,
                    offset=current_item["start"],
                    duration=self.item_length)
            if self.as_dict:
                return_value[source] = array
            else:
                return_value.append(array)
        return return_value


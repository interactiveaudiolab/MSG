"""
Run MSG Inference for a single audio example.
E.g: python Run_Inference.py --audio_file=/Users/boazcogan/Northwestern/misc/model_saves/bass.wav --generator=/Users/boazcogan/Northwestern/misc/model_saves/29netG.pt
"""
import argparse
import librosa
import numpy as np
import nussl
from models.Demucs import *
import os
import torch
import soundfile as sf

from data_generation.generate_test_set import overlap_add_process

def parseConfig(config):
    exp_dict = yaml.load(open(os.path.join(config), 'r'),
                         Loader=yaml.FullLoader)
    return Struct(**exp_dict).parameters

def run_inference(audio_example, generator_checkpoint, config=None):
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    sample_rate = 16_000
    data, sr = librosa.load(audio_example, sr=sample_rate)
    netG = Demucs([""],audio_channels=1, samplerate=sample_rate ,segment_length=16_000, skip_cxn = True,lstm_layers=0, normalize=True).to(device)
    netG.load_state_dict(torch.load(generator_checkpoint))
    netG.eval()
    
    with torch.no_grad():
        estimation = overlap_add_process(data, sample_rate, netG)
    generated_signal = nussl.AudioSignal(audio_data_array=estimation)
    
    # perform inference
    if not os.path.exists("msg_output"):
        os.mkdir("msg_output")

    # write audio
    song_name = audio_example.split('/')[-1]
    sf.write(f'msg_output/{song_name}', generated_signal.peak_normalize().audio_data.T, sample_rate)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_file", "-a", type=str,
                        help="absolute path to the audio file", required=True)
    parser.add_argument("--generator", "-g", type=str, required=True,
                        help="absolute path to the generator checkpoint")
    exp, exp_args = parser.parse_known_args()
    run_inference(exp.audio_file, exp.generator)

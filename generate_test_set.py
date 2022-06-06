from utils.RunEvaluation import convert_to_audio, Struct, parseConfig
from models.Demucs import *
import torch
import torch.nn.functional as F
import os
from model_factory import ModelFactory
import datasets.EvaluationSet as EV
import numpy as np
import soundfile as sf
import argparse
from scipy.signal import get_window

device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str,
                            help='Test Generation Config File',
                            required=True)
    exp, exp_args = parser.parse_known_args()
    config = parseConfig(exp.config)
    os.mkdir(f'/home/noah/TestSet{config.source}')
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    netG = Demucs([config.source],audio_channels=config.audio_channels, samplerate=config.sample_rate ,segment_length=int(config.segment_duration * config.sample_rate), skip_cxn = config.skip_cxn,lstm_layers=config.lstm_layers, normalize=True).to(device)
    netG.load_state_dict(torch.load(config.checkpoint))
    netG.eval()
    with torch.no_grad():
        for i, s in enumerate(config.separators):
            os.mkdir(f'/home/noah/TestSet{config.source}/{s}')
            os.mkdir(f'/home/noah/TestSet{config.source}/{s}/{s}')
            os.mkdir(f'/home/noah/TestSet{config.source}/{s}/MSG')
            eval_set = EV.EvaluationSet(songs_dir=f'/home/boaz/datasets/DemucsDataset/{s}Data7Second/',sample_rate=config.sample_rate,source=config.source)
            for (noisy,ground_truth,song_name) in eval_set:
                generated = overlap_add_process(noisy,16000,netG)
                ground_truth_signal, noisy_signal, generated_signal = convert_to_audio(noisy, ground_truth, generated)
                sf.write(f'/home/noah/TestSet{config.source}/{s}/{s}/{song_name}',noisy_signal.peak_normalize().audio_data.T,config.sample_rate)
                sf.write(f'/home/noah/TestSet{config.source}/{s}/MSG/{song_name}',generated_signal.peak_normalize().audio_data.T,config.sample_rate)

def overlap_add_process(signal, win_size,model):
        sig = torch.tensor(signal,device=device)[None, :, None]
        hop_size = win_size//2
        sig = sig.permute(0, 2, 1)
        batch, channels, n_frames = sig.size()
        sig_unfolded = F.unfold(
            sig.unsqueeze(-1),
            kernel_size=(win_size, 1),
            padding=(win_size, 0),
            stride=(hop_size, 1),
        )
        n_chunks = sig_unfolded.shape[-1]
        window = get_window('hamming',win_size)
        out = []
        for idx in range(n_chunks):
            sig_chunk = sig_unfolded[..., idx][..., None]
            #print(f'Sig Chunk = {sig_chunk.shape}')
            #call MSG
            #est_chunk = model()
            inp = F.pad(sig_chunk.permute(0,2,1), (4000, 4000), "constant", 0)
            est_chunk = model(inp,sig_chunk.unsqueeze(1) ).squeeze(1).permute(1,2,0).cpu().numpy()
            #print(f'Est Chunk = {est_chunk.shape}')
            #est_chunk = sig_chunk.cpu().numpy()
            est_chunk = est_chunk.reshape(1, -1)
            est_chunk = est_chunk * window
            out.append(torch.from_numpy(est_chunk))
        out = torch.stack(out).reshape(n_chunks, 1, win_size)
        out = out.permute(1, 2, 0)

        out = F.fold(
            out,
            (n_frames, 1),
            kernel_size=(win_size, 1),
            padding=(win_size, 0),
            stride=(hop_size, 1),
        )
        est_src = out.squeeze(-1).reshape(batch, 1, -1)
        return est_src.numpy().flatten()

if __name__ == '__main__':
    main()


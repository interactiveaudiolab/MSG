import os
import argparse
import re
import yaml
import numpy as np
import nussl
import librosa
from scipy.spatial.distance import cosine

import torch

from models.MelGAN import Audio2Mel, GeneratorMel
from datasets.WaveDataset import MusicDataset


pattern = re.compile('[\W_]+')

np.random.seed(0)
torch.manual_seed(0)

def parse_args(args_list):
    arg_names = [pattern.sub('', s) for s in args_list[::2]]
    result = {}
    for i, k in enumerate(arg_names):
        result[k] = _sanitize_value(args_list[2*i+1])
    return result

def _sanitize_value(v):
    try:
        return int(v)
    except ValueError:
        pass

    try:
        return float(v)
    except ValueError:
        pass

    if v.lower() in ['null', 'none']:
        return None

    if isinstance(v, str):
        if v.lower() == 'false':
            return False
        if v.lower() == 'true':
            return True

    return v

def update_parameters(exp_dict, params_dict):
    sanitized_exp_keys = {pattern.sub('', key): key for key in exp_dict.keys()}
    for param_key, param_val in params_dict.items():
        if param_key in sanitized_exp_keys.keys():
            exp_dict[sanitized_exp_keys[param_key]] = param_val
    return exp_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', '-e', type=str, help='Experiment yaml file')
    exp, exp_args = parser.parse_known_args()
    exp_file_path = exp.exp

    if not exp_file_path:
        raise ParameterError('Must have an experiment definition `--exp-def`!')

    exp_dict = yaml.load(open(os.path.join(exp_file_path), 'r'),
                         Loader=yaml.FullLoader)

    if len(exp_args) > 0:
        try:
            new_args = parse_args(exp_args)
        except IndexError:
            raise ParameterError('Cannot parse input parameters!')
        new_dict = update_parameters(exp_dict['parameters'], new_args)
        exp_dict['parameters'] = new_dict

    params = exp_dict['parameters']

    G1 = GeneratorMel(params['n_mel_channels'], params['ngf'], params['n_residual_layers'])
    G1.load_state_dict(torch.load( params['model_load_dir']+ str(params['load_epoch'] + 'netG.pt')))


  #change this


    test_dirty = []
    test_clean = []

    for s in os.listdir(params['original_sources_path:']):
        if params['source'] in s:
            test_dirty.append(params['separated_sources_path']  + s)
            test_clean.append(params['original_sources_path:']  + s)

    ds_test = MusicDataset(test_dirty,test_clean,44100,44100)
    G1.to('cuda')
    G1.eval()
    fft = Audio2Mel(n_mel_channels=params['n_mel_channels']).cuda()

    si_sdr_noisy = []
    si_sdr_generated = []

    sd_sdr_noisy = []
    sd_sdr_generated = []

    si_sar_noisy = []
    si_sar_generated = []

    si_sir_noisy = []
    si_sir_generated = []

    snr_noisy = []
    snr_generated = []

    noisy_cosine = []
    generated_cosine = []


    with torch.no_grad():
        for start in range(50):
            clean1 = np.array([])
            noisy1 = np.array([])
            aud1 = np.array([])
        for i in range(7*start, 7*start+7):
            n,c = ds_test[i]
            clean1 = np.concatenate((clean1,c))
            noisy1 = np.concatenate((noisy1,n))

            s_t = fft(torch.from_numpy(n).float().unsqueeze(0).unsqueeze(0).cuda()).detach()
            x_pred_t = G1(s_t.cuda(),torch.from_numpy(n).cuda())

            a = x_pred_t.squeeze().squeeze().detach().cpu().numpy()
            aud1 = np.concatenate((aud1,a))

        c = nussl.AudioSignal(audio_data_array=clean1)
        n = nussl.AudioSignal(audio_data_array=noisy1)
        g = nussl.AudioSignal(audio_data_array=aud1)
        bss_eval = nussl.evaluation.BSSEvalScale(
        true_sources_list=[c],
        estimated_sources_list=[n]
        )

        noisy = librosa.feature.melspectrogram(y=noisy1, sr=44100, n_mels=128,
                                        fmax=8000)
        clean = librosa.feature.melspectrogram(y=clean1, sr=44100, n_mels=128,
                                        fmax=8000)
        generated = librosa.feature.melspectrogram(y=aud1, sr=44100, n_mels=128,
                                        fmax=8000)

        noisy_cosine.append(cosine(clean.flatten(),noisy.flatten()))
        generated_cosine.append(cosine(clean.flatten(),generated.flatten()))

        noisy_eval = bss_eval.evaluate()

        bss_eval = nussl.evaluation.BSSEvalScale(
            true_sources_list=[c],
            estimated_sources_list=[g]
        )
        gen_eval = bss_eval.evaluate()

        si_sdr_noisy.append(noisy_eval['source_0']['SI-SDR'])
        si_sdr_generated.append(gen_eval['source_0']['SI-SDR'])
        sd_sdr_noisy.append(noisy_eval['source_0']['SD-SDR'])
        sd_sdr_generated.append(gen_eval['source_0']['SD-SDR'])
        si_sar_noisy.append(noisy_eval['source_0']['SI-SAR'])
        si_sar_generated.append(gen_eval['source_0']['SI-SAR'])
        si_sir_noisy.append(noisy_eval['source_0']['SI-SIR'])
        si_sir_generated.append(gen_eval['source_0']['SI-SIR'])
        snr_noisy.append(noisy_eval['source_0']['SNR'])
        snr_generated.append(gen_eval['source_0']['SNR'])
    lines = []
    lines.append('Original SI-SDR'+ str(np.mean(si_sdr_noisy)))
    lines.append('Our SI-SDR' + str(np.mean(si_sdr_generated)))
    lines.append('\nOriginal SD-SDR'+ str(np.mean(sd_sdr_noisy)))
    lines.append('Our SD-SDR'+ str(np.mean(sd_sdr_generated)))
    lines.append('\nOriginal SI-SAR', np.mean(si_sar_noisy))
    lines.append('Our SI-SAR'+ str(np.mean(si_sar_generated)))
    lines.append('\nOriginal SI-SIR' + str(np.mean(si_sir_noisy)))
    lines.append('Our SI-SIR'+ str(np.mean(si_sir_generated)))
    lines.append('\nOriginal SNR'+ str(np.mean(snr_noisy)))
    lines.append('Our SNR'+ str(np.mean(snr_generated)))
    lines.append('\nDemucs Mean Spectral Cosine Distance'+ str(np.mean(noisy_cosine)))
    lines.append('MSG Mean Spectral Cosine Distance'+ str(np.mean(generated_cosine)))

    with open(params['log_dir'] + params['log_id'] + 'logs.txt', 'w') as f:
      f.writelines(lines)
    

class ParameterError(Exception):
    pass

if __name__ == '__main__':
    main()

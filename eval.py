import os
import argparse
import re
import yaml
import numpy as np
import nussl
import librosa
import wandb

import torch
import torch.nn.functional as F

from models.MelGAN import Audio2Mel, GeneratorMel
from datasets.EvaluationDataset import EvalSet
from model_factory import ModelFactory

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
    wandb.init(config=params)
    config = wandb.config


    global device
    device = torch.device(f"cuda:{config.gpus[0]}" if torch.cuda.is_available() else "cpu")
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)


    ds_test = EvalSet(config.dataset_path, 1,44100,sources=('dirty_bass', 'bass'),as_dict=False, hop_length= config.hop_len)
    song_indices = ds_test.get_song_indices()
    
    ModelSelector = ModelFactory(config, torch.optim.Adam)
    netG = ModelSelector.generator().to(device)
    netG.load_state_dict(torch.load(config.model_load_path))


    sdr_noisy = []
    sdr_generated = []

    sir_noisy = []
    sir_generated = []

    sar_noisy = []
    sar_generated = []

 


    with torch.no_grad():
        for i in range(50):

            shift = int(1/config.hop_len)
            reduction_factor = int(44100 * config.hop_len)
            song = i
            song_length = (song_indices[song][1]) - song_indices[song][0]

            aud1 = np.zeros((song_length+shift)*reduction_factor)
            clean1 = np.zeros((song_length+shift)*reduction_factor)
            noisy1 = np.zeros((song_length+shift)*reduction_factor)
            mix1 = np.zeros((song_length+shift)*reduction_factor)

            for i in range(song_indices[song][0],song_indices[song][1]+1):
                x_t_0 = torch.from_numpy(ds_test[i][1]).unsqueeze(0).unsqueeze(0).to(device)
                x_t_1 = ds_test[i][2]
                x_t_mix = ds_test[i][3]


                inp  = F.pad(x_t_0, (2900,2900), "constant", 0)
                x_pred_t = netG(inp,x_t_0.unsqueeze(1)).squeeze(1)
                a = x_pred_t.squeeze().squeeze().detach().cpu().numpy()

                offset = song_indices[song][0]
                ind = i - offset

                aud1[ind*reduction_factor: (ind+shift)*reduction_factor] += a

                noisy1[ind*reduction_factor: (ind+shift)*reduction_factor] += x_t_0.cpu().squeeze(0).squeeze(0).numpy()
                clean1[ind*reduction_factor: (ind+shift)*reduction_factor] += x_t_1
                mix1[ind*reduction_factor: (ind+shift)*reduction_factor] += x_t_mix
            for i in range(0,shift-1):
                aud1[i*reduction_factor:(i+1)*reduction_factor] /= (i+1)
                aud1[-(i+1)*reduction_factor:-i*reduction_factor] /= (i+1)

                clean1[i*reduction_factor:(i+1)*reduction_factor] /= (i+1)
                clean1[-(i+1)*reduction_factor:-i*reduction_factor] /= (i+1)

                noisy1[i*reduction_factor:(i+1)*reduction_factor] /= (i+1)
                noisy1[-(i+1)*reduction_factor:-i*reduction_factor] /= (i+1)

                mix1[i*reduction_factor:(i+1)*reduction_factor] /= (i+1)
                mix1[-(i+1)*reduction_factor:-i*reduction_factor] /= (i+1)

            aud1[(i+1)*reduction_factor:-(i+1)*reduction_factor] /= (i+2)
            noisy1[(i+1)*reduction_factor:-(i+1)*reduction_factor] /= (i+2)
            clean1[(i+1)*reduction_factor:-(i+1)*reduction_factor] /= (i+2)
            mix1[(i+1)*reduction_factor:-(i+1)*reduction_factor] /= (i+2)

            noisy1 = noisy1[0:(-i+2)*reduction_factor]
            aud1 = aud1[0:(-i+2)*reduction_factor]
            clean1 = clean1[0:(-i+2)*reduction_factor]
            mix1 = mix1[0:(-i+2)*reduction_factor]
            
            clean1[librosa.amplitude_to_db(clean1)<-60] = 0
            aud1[librosa.amplitude_to_db(aud1)<-60] = 0
            noisy1[librosa.amplitude_to_db(noisy1)<-60] = 0

            c = nussl.AudioSignal(audio_data_array=clean1)
            n = nussl.AudioSignal(audio_data_array=noisy1)
            g = nussl.AudioSignal(audio_data_array=aud1)

            c1 = nussl.AudioSignal(audio_data_array=mix1-clean1)
            n1 = nussl.AudioSignal(audio_data_array=mix1-noisy1)
            g1 = nussl.AudioSignal(audio_data_array=mix1-aud1)

            bss_eval = nussl.evaluation.BSSEvalV4(
            true_sources_list=[c,c1],
            estimated_sources_list=[n,n1],
            )
            noisy_eval = bss_eval.evaluate()

            bss_eval = nussl.evaluation.BSSEvalV4(
                true_sources_list=[c,c1],
                estimated_sources_list=[g,g1]
            )
            gen_eval = bss_eval.evaluate()
            
            sdr_noisy.append(np.nanmedian(noisy_eval['source_0']['SDR']))
            sdr_generated.append(np.nanmedian(gen_eval['source_0']['SDR']))
            sar_noisy.append(np.nanmedian(noisy_eval['source_0']['SAR']))
            sar_generated.append(np.nanmedian(gen_eval['source_0']['SAR']))
            sir_noisy.append(np.nanmedian(noisy_eval['source_0']['SIR']))
            sir_generated.append(np.nanmedian(gen_eval['source_0']['SIR']))
            

    lines = []
    lines.append('\nOriginal SD-SDR: '+ str(np.nanmedian(sdr_noisy)))
    lines.append('Our SD-SDR: '+ str(np.nanmedian(sdr_generated)))
    lines.append('\nOriginal SAR: '+ str(np.nanmedian(sar_noisy)))
    lines.append('Our SAR: '+ str(np.nanmedian(sar_generated)))
    lines.append('\nOriginal SIR: ' + str(np.nanmedian(sir_noisy)))
    lines.append('Our SIR: '+ str(np.nanmedian(sir_generated)))


    with open(params['log_dir'] + params['run_id'] + 'logs1.txt', 'w') as f:
      f.write('\n'.join(lines))
    

class ParameterError(Exception):
    pass

if __name__ == '__main__':
    main()

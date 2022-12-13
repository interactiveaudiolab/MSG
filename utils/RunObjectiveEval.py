import torch
import torch.nn.functional as F

import nussl
import librosa
import yaml
import os
from collections import namedtuple
from model_factory import ModelFactory
import datasets.EvaluationSet as EV
import numpy as np
import soundfile as sf
import math
import json
from sklearn.metrics import precision_score,recall_score

class Struct:
     def __init__(self, **entries):
         for key, value in entries.items():
             if isinstance(value, dict):
                 self.__dict__.update({key:Struct(**value)})
             else:
                self.__dict__.update({key:value})

def parseConfig(config):
    exp_dict = yaml.load(open(os.path.join(config), 'r'),
                         Loader=yaml.FullLoader)
    return Struct(**exp_dict).parameters

def run_inference(netG, ds, start, end, shift, reduction_factor, device):
    song_length = end - start
    generated = np.zeros((song_length + shift) * reduction_factor)
    ground_truth = np.zeros((song_length + shift) * reduction_factor)
    noisy = np.zeros((song_length + shift) * reduction_factor)
    mix = np.zeros((song_length + shift) * reduction_factor)
    for i in range(start, end + 1, 1):
        # second is to perform SDR, SIR, SAR evaluation on each song
        source_class, _ = ds[i]
        noisy_source = torch.from_numpy(source_class[1]).unsqueeze(0).unsqueeze(0).to(device)
        ground_truth_source = source_class[2]
        mixture_segment = source_class[3]

        # perform the inference
        inp = F.pad(noisy_source, (4000, 4000), "constant", 0)
        x_pred_t = netG(inp, noisy_source.unsqueeze(1)).squeeze(1)
        a = x_pred_t.squeeze().squeeze().detach().cpu().numpy()

        # get the offsets and current segment index
        offset = start
        ind = i - offset

        # Add overlapping signals together
        generated[ind * reduction_factor: (ind + shift) * reduction_factor] += a
        noisy[ind * reduction_factor: (ind + shift) * reduction_factor] += \
            noisy_source.cpu().squeeze(0).squeeze(0).numpy()
        ground_truth[ind * reduction_factor: (ind + shift) * reduction_factor] += \
            ground_truth_source
        mix[ind * reduction_factor: (ind + shift) * reduction_factor] += \
            mixture_segment

    # handle edge cases of the overlap add
    for i in range(0, shift - 1):
        generated[i * reduction_factor:(i + 1) * reduction_factor] /= (i + 1)
        generated[-(i + 1) * reduction_factor:-i * reduction_factor] /= (i + 1)

        ground_truth[i * reduction_factor:(i + 1) * reduction_factor] /= (i + 1)
        ground_truth[-(i + 1) * reduction_factor:-i * reduction_factor] /= (i + 1)

        noisy[i * reduction_factor:(i + 1) * reduction_factor] /= (i + 1)
        noisy[-(i + 1) * reduction_factor:-i * reduction_factor] /= (i + 1)

        mix[i * reduction_factor:(i + 1) * reduction_factor] /= (i + 1)
        mix[-(i + 1) * reduction_factor:-i * reduction_factor] /= (i + 1)

    # handle average case of the overlap add
    generated[(i + 1) * reduction_factor:-(i + 1) * reduction_factor] /= (i + 2)
    noisy[(i + 1) * reduction_factor:-(i + 1) * reduction_factor] /= (i + 2)
    ground_truth[(i + 1) * reduction_factor:-(i + 1) * reduction_factor] /= (i + 2)
    mix[(i + 1) * reduction_factor:-(i + 1) * reduction_factor] /= (i + 2)

    # remove padding introduced during inference
    noisy = noisy[0:(-i + 2) * reduction_factor]
    generated = generated[0:(-i + 2) * reduction_factor]
    ground_truth = ground_truth[0:(-i + 2) * reduction_factor]
    mix = mix[0:(-i + 2) * reduction_factor]

    # db threshold for sounds that are too quiet to be perceptible
    #ground_truth[librosa.amplitude_to_db(ground_truth) < -60] = 0
    #generated[librosa.amplitude_to_db(generated) < -60] = 0
    #noisy[librosa.amplitude_to_db(noisy) < -60] = 0

    return noisy, ground_truth, mix, generated


def calculate_spec_rolloff(target, estimated, sr,log=False,threshold=-40):
    def _get_stuff(sig,sr): 
        S_rms = librosa.feature.rms(y=sig)
        db_filter = librosa.amplitude_to_db(S_rms)
        db_filter[db_filter>=threshold] = 1
        db_filter[db_filter<threshold] = math.nan
        rolloff = librosa.feature.spectral_rolloff(y=sig,roll_percent=.98,sr=sr) * db_filter
        return rolloff
    def _get_hpss_rolloffs(sig):
        h,p = librosa.effects.hpss(sig)
        return _get_stuff(h),_get_stuff(p)
 
    gt_cd = _get_stuff(target,sr)[0]
    est_cd = _get_stuff(estimated,sr)[0]
    if log:
        return np.log2(est_cd) - np.log2(gt_cd)
    return est_cd - gt_cd

def get_num_onsets(target,estimated,sr):
    def get_num_onsets(sig,sr):
        o_env = librosa.onset.onset_strength(y=sig, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
        return(len(onset_frames))
    gt_onsets = get_num_onsets(target,sr)
    est_onsets = get_num_onsets(estimated, sr)
    return [est_onsets - gt_onsets]
    
def calculate_onset_strengths(target, estimated,sr):
    def get_onsets(sig,sr):
        o_env = librosa.onset.onset_strength(y=sig, sr=sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=sr)
        return o_env[onset_frames]
    gt_onsets = get_onsets(target,sr)
    est_onsets = get_onsets(estimated,sr)
    return gt_onsets, est_onsets

def get_strength_vals(y,hops,thresh,sr,**kwargs):
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr,hop_length=hops)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_strength, hop_length=hops,sr=sr)
    return onset_strength,len(onset_frames)

def threshold_strength(y,hops,thresh,sr,**kwargs):
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr,hop_length=hops)
    onset_strength[onset_strength>=thresh] = 1
    onset_strength[onset_strength<thresh] = 0
    return onset_strength

def precision(true_pos,false_pos):
    return true_pos/(true_pos+false_pos)

def recall(true_pos,false_neg):
    return true_pos/(true_pos+false_neg)

def calculate_scores(gt_signal,sep_signal,sr,hop):
    wait = .06 * 16000 / hop
    gt = librosa.onset.onset_detect(gt_signal,sr=sr,hop_length=hop,backtrack=False,wait=wait)
    sep = librosa.onset.onset_detect(sep_signal,sr=sr,hop_length=hop,backtrack=False,wait=wait)
    true_pos = 0
    false_pos = 0
    false_neg = 0 

    for onset in sep:
        if onset in gt:
            true_pos +=1
        else:
            false_pos +=1
    for onset in gt:
        if onset not in sep:
            false_neg+=1
    return true_pos,false_pos,false_neg

def Evaluate(config,names):
    config = parseConfig(config)
    results = {}
    best_g = config.best_g
    for i in range(len(config.test_sources_paths)):
        #sep_precision,sep_recall,msg_precision,msg_recall = EvaluateLoop(config,best_g,names,config.test_sources_paths[i],config.evaluation_models[i])
        #results[f'{config.evaluation_models[i]} Precision:'] = sep_precision
        #results[f'{config.evaluation_models[i]} + MSG Precision:'] = msg_precision
        #results[f'{config.evaluation_models[i]} Recall:'] = sep_recall
        #results[f'{config.evaluation_models[i]} + MSG Recall:'] = msg_recall
        sep_rolloff, msg_rolloff = EvaluateLoop(config,best_g,names,config.test_sources_paths[i],config.evaluation_models[i])
        #sep_precision, sep_recall, msg_precision, msg_recall = EvaluateLoop(config,best_g,names,config.test_sources_paths[i],config.evaluation_models[i])
        # np.save(f'/home/noah/plotting_data/{config.source}/{config.evaluation_models[i]}_precision.npy',sep_precision)
        # np.save(f'/home/noah/plotting_data/{config.source}/{config.evaluation_models[i]}+msg_precision.npy',msg_precision)
        # np.save(f'/home/noah/plotting_data/{config.source}/{config.evaluation_models[i]}_recall.npy',sep_recall)
        # np.save(f'/home/noah/plotting_data/{config.source}/{config.evaluation_models[i]}+msg_recall.npy',msg_recall)
        # np.save(f'/home/noah/plotting_data/{config.source}/gt_onset_strength.npy',gt_strengths)
        # np.save(f'/home/noah/plotting_data/{config.source}/{config.evaluation_models[i]}_onset_strength.npy',sep_strengths)
        # np.save(f'/home/noah/plotting_data/{config.source}/{config.evaluation_models[i]}+msg_onset_strength.npy',msg_strengths)
        # np.save(f'/home/noah/plotting_data/{config.source}/gt_detected_onsets_pp.npy',gt_onsets)
        # np.save(f'/home/noah/plotting_data/{config.source}/{config.evaluation_models[i]}_detected_onsets_pp.npy',sep_onsets)
        # np.save(f'/home/noah/plotting_data/{config.source}/{config.evaluation_models[i]}+msg_detected_onsets_pp.npy',msg_onsets)
        #if i ==0:
            #np.save(f'/home/noah/plotting_data/{config.source}/gt_onset_strength.npy ', gt_onset_strengths)
        np.save(f'/home/noah/plotting_data/{config.source}/{config.evaluation_models[i]}_log_rolloff.npy',sep_rolloff)
        np.save(f'/home/noah/plotting_data/{config.source}/{config.evaluation_models[i]}+msg_log_rolloff.npy', msg_rolloff)
       #np.save(f'/home/noah/plotting_data/{config.source}/{config.evaluation_models[i]}_onset_strength.npy',sep_onset_strengths)
        #np.save(f'/home/noah/plotting_data/{config.source}/{config.evaluation_models[i]}+msg_onset_strength.npy', msg_onset_strengths)
        #np.save(f'/home/noah/plotting_data/{config.source}/{config.evaluation_models[i]}_num_onsets.npy',sep_num_onsets)
        #np.save(f'/home/noah/plotting_data/{config.source}/{config.evaluation_models[i]}+msg_num_onsets.npy',msg_num_onsets)
    #with open(f'/home/noah/{config.source}_results.json', 'w') as f:
     #   json.dump(results, f)

def EvaluateLoop(config, best_g, names,dataset_path,dataset_name) -> tuple:
    
    # get device
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

    

    # load the generator, load the checkpoint, send to device
    sep_rolloff = np.array([])
    msg_rolloff = np.array([])

    hop_length=256


    eval_set = EV.EvaluationSet(songs_dir=f'/home/boaz/datasets/DemucsDataset/all_separator_set16/test/{dataset_name}Data16/',sample_rate=config.sample_rate,source=config.source)
    # sep_true_pos = 0
    # sep_false_neg = 0
    # sep_false_pos = 0

    # msg_true_pos = 0
    # msg_false_neg = 0
    # msg_false_pos = 0


    # list of start indices and end indices
    # run evaluation on each song:
    # gt_onsets = np.array([])
    # sep_onsets = np.array([])
    # msg_onsets = np.array([])

    # gt_strength_vals = np.array([])
    # sep_strength_vals = np.array([])
    # msg_strength_vals = np.array([])

    # sep_precision = np.array([])
    # sep_recall = np.array([])
    # msg_precision = np.array([])
    # msg_recall = np.array([])

    t = .75
    for (noisy,ground_truth,song_name) in eval_set:
        msg, _ = librosa.load(f'{config.msg_path}/{dataset_name}/MSG/{song_name}',config.sample_rate)
        ground_truth_signal = nussl.AudioSignal(audio_data_array=ground_truth).peak_normalize().audio_data.flatten()
        noisy_signal = nussl.AudioSignal(audio_data_array=noisy).peak_normalize().audio_data.flatten()
        # gt_strength_vals = np.append(gt_strength_vals, threshold_strength(ground_truth_signal,256,t,config.sample_rate))
        # sep_strength_vals = np.append(sep_strength_vals, threshold_strength(noisy_signal,256,t,config.sample_rate))
        # msg_strength_vals = np.append(msg_strength_vals,threshold_strength(msg,256,t,config.sample_rate))
        msg_rolloff = np.append(msg_rolloff,calculate_spec_rolloff(ground_truth_signal,msg,config.sample_rate,log=True))
        #msg_num_onsets = np.concatenate((msg_num_onsets,get_num_onsets(ground_truth,generated,config.sample_rate)))
        #msg_onset_strengths = np.concatenate((msg_onset_strengths,calculate_onset_strengths(ground_truth,generated,config.sample_rate)[1]))
        # ground_truth_signal = nussl.AudioSignal(audio_data_array=ground_truth).peak_normalize().audio_data.flatten()
        # noisy_signal = nussl.AudioSignal(audio_data_array=noisy).peak_normalize().audio_data.flatten()
        # gt_onsets = np.append(gt_onsets,get_strength_vals(ground_truth_signal,hop_length,t,config.sample_rate)[1])
        # sep_onsets = np.append(sep_onsets,get_strength_vals(noisy_signal,hop_length,t,config.sample_rate)[1])
        # msg_onsets = np.append(msg_onsets,get_strength_vals(msg,hop_length,t,config.sample_rate)[1])
        # gt_strengths = np.append(gt_strengths,get_strength_vals(ground_truth_signal,hop_length,t,config.sample_rate)[0])
        # sep_strengths = np.append(sep_strengths,get_strength_vals(noisy_signal,hop_length,t,config.sample_rate)[0])
        # msg_strengths = np.append(msg_strengths,get_strength_vals(msg,hop_length,t,config.sample_rate)[0])
        sep_rolloff = np.append(sep_rolloff,calculate_spec_rolloff(ground_truth_signal,noisy_signal,config.sample_rate,log=True))
        #sep_num_onsets = np.concatenate((sep_num_onsets,get_num_onsets(ground_truth,noisy,config.sample_rate)))
        #sep_onset_strengths = np.concatenate((sep_onset_strengths,calculate_onset_strengths(ground_truth,noisy,config.sample_rate)[1]))
        #gt_onset_strengths = np.concatenate((sep_onset_strengths,calculate_onset_strengths(ground_truth,noisy,config.sample_rate)[0]))

    # sep_precision = np.append(sep_precision,precision_score(gt_strength_vals,sep_strength_vals))
    # msg_precision = np.append(msg_precision,precision_score(gt_strength_vals,msg_strength_vals))
    # sep_recall = np.append(sep_recall,recall_score(gt_strength_vals,sep_strength_vals))
    # msg_recall = np.append(msg_recall,recall_score(gt_strength_vals,msg_strength_vals))
    #return msg_rolloff, msg_num_onsets, msg_onset_strengths, sep_rolloff, sep_num_onsets, sep_onset_strengths, gt_onset_strengths
    
    # sep_precision = precision(sep_true_pos,sep_false_pos)
    # msg_precision = precision(sep_true_pos,msg_false_pos)
    # sep_recall = recall(sep_true_pos,sep_false_neg)
    # msg_recall = recall(sep_true_pos,msg_false_neg)
    # return sep_precision, sep_recall, msg_precision, msg_recall
    #return  gt_onsets, sep_onsets,msg_onsets, gt_strengths, sep_strengths, msg_strengths
    return sep_rolloff,msg_rolloff
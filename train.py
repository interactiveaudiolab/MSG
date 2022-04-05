import os
import sys
import argparse
import re
import yaml
from model_factory import ModelFactory
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import wandb
import librosa, librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import imageio

import torch.nn as nn

from models.MelGAN import Audio2Mel, GeneratorMelMix, DiscriminatorMel
from models.Demucs import *
from datasets.WaveDataset import MusicDataset
from datasets.Wrapper import DatasetWrapper
from datasets.EvaluationDataset import EvalSetWrapper
from utils.losses import *
import utils.save_and_log as sal
import utils.RunEpoch as rp


pattern = re.compile('[\W_]+')
#default GPU is 1
device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")

def create_saves_directory(directory_path, development_flag=False):
    if development_flag:
        return
    if os.path.exists(directory_path):
        raise Exception(f"The saves directory for {directory_path} already exists")
    os.mkdir(directory_path)


def parse_args(args_list):
    arg_names = [pattern.sub('', s) for s in args_list[::2]]
    result = {}
    for i, k in enumerate(arg_names):
        result[k] = _sanitize_value(args_list[2*i+1])
    return result

def _sanitize_value(value):
    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    if isinstance(value, str):
        if value.lower() in ['null', 'none']:
            return None
        if value.lower() == 'false':
            return False
        if value.lower() == 'true':
            return True

    return value

def update_parameters(exp_dict, params_dict):
    sanitized_exp_keys = {pattern.sub('', key): key for key in exp_dict.keys()}
    for param_key, param_val in params_dict.items():
        if param_key in sanitized_exp_keys.keys():
            exp_dict[sanitized_exp_keys[param_key]] = param_val
    return exp_dict

def initialize_nussl_dataloader(train_path, valid_path, source, batch_size, n_cpu, **kwargs):
    train_set = DatasetWrapper(train_path, source, **kwargs)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                               num_workers=n_cpu, shuffle=True)
    valid_set = EvalSetWrapper(valid_path, source, **kwargs)
    valid_loader = DataLoader(valid_set, batch_size=1,
                               num_workers=n_cpu, shuffle=False)
    return train_loader, valid_loader


def train(yaml_file=None):

    if yaml_file:
        exp_file_path = yaml_file
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--exp', '-e', type=str, help='Experiment yaml file')
        exp, exp_args = parser.parse_known_args()
        exp_file_path = exp.exp

    if not exp_file_path:
        raise ParameterError('Must have an experiment definition `--exp-def`!')

    exp_dict = yaml.load(open(os.path.join(exp_file_path), 'r'),
        Loader=yaml.FullLoader)

    params = exp_dict['parameters']

    wandb.init(config=params)
    config = wandb.config
    if config.start_epoch ==0: 
        create_saves_directory(config.model_save_dir, config.debug)
    #global device
    #device = torch.device(f"cuda:{config.gpus[0]}" if torch.cuda.is_available() else "cpu")
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    ModelSelector = ModelFactory(config, torch.optim.Adam)
    
    netG = ModelSelector.generator().to(device)
    if config.multi_disc:
        netD, netD_spec = ModelSelector.discriminator()
        netD.to(device)
        netD_spec.to(device)
        optD_spec = torch.optim.Adam(netD_spec.parameters(), lr=config.lr, betas=(config.b1,config.b2))
    else:
        netD = ModelSelector.discriminator().to(device)


    optG = torch.optim.Adam(netG.parameters(), lr=config.lr, betas=(config.b1,config.b2))
    optD = torch.optim.Adam(netD.parameters(), lr=config.lr, betas=(config.b1,config.b2))

    train_loader, valid_loader = initialize_nussl_dataloader(
                                    config.train_sources_path,
                                    config.valid_sources_path,
                                    config.source,
                                    config.batch_size,
                                    config.n_cpu,
                                    mix_folder = config.mix_folder,
                                    sample_rate = config.sample_rate,
                                    segment_dur = config.segment_duration,
                                    verbose = config.verbose,
                                    mono = config.mono
                                    )


    start_epoch=config.start_epoch

    if start_epoch > 0:
        netG.load_state_dict(torch.load(config.model_load_dir +  str(start_epoch-1) + "netG.pt"))
        netD.load_state_dict(torch.load(config.model_load_dir +  str(start_epoch-1) + "netD.pt"))
        optG.load_state_dict(torch.load(config.model_load_dir +  str(start_epoch-1) +
                                                                    "optG.pt").state_dict())
        optD.load_state_dict(torch.load(config.model_load_dir +  str(start_epoch-1) +
                                                                    "optD.pt").state_dict())
    ###################
    ##### TRAINING ####
    ###### LOOP #######
    ###################
    writer = SummaryWriter()
    costs = []
    netG.train()
    netD.train()
    best_g = [None,None,None]
    if config.multi_disc:
        netD_spec.train()
    steps = 0
    sdr = SISDRLoss()
    best_SDR = 0
    best_reconstruct = 100
    best_l1 = 100
    for epoch in range(start_epoch, config.n_epochs):
        if (epoch+1) % config.checkpoint_interval == 0 and epoch != start_epoch and not config.disable_save:
            sal.save_model(config.model_save_dir, netG.state_dict(), netD.state_dict(), optG,optD,epoch, spec=False, config=config)
        steps, _, _ = rp.runEpoch(train_loader, config, netG, netD, optG, optD, device, epoch, steps, writer)

        if epoch % config.validation_epoch==0:
            with torch.no_grad():
                _, costs, aud = rp.runEpoch(valid_loader, config, netG, netD, optG, optD, device, epoch, steps, writer, validation=True)
            best_g, best_l1, best_reconstruct, best_SDR = sal.iteration_logs(netD, netG, optG, optD, steps, epoch, config, best_l1,best_SDR, best_reconstruct, aud, costs,best_g)
    return wandb.run.get_url(), best_g
class ParameterError(Exception):
    pass

if __name__ == '__main__':
    train()


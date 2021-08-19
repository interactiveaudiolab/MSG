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

from models.MelGAN import Audio2Mel, GeneratorMelMix, DiscriminatorMel, SISDRLoss
from models.Demucs import *
from datasets.WaveDataset import MusicDataset
from datasets.Wrapper import DatasetWrapper


pattern = re.compile('[\W_]+')
device = ""

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

def initialize_nussl_dataloader(train_path, valid_path, source, batch_size, n_cpu, **kwargs):
    train_set = DatasetWrapper(train_path, source, **kwargs)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                               num_workers=n_cpu, shuffle=True)
    valid_set = DatasetWrapper(valid_path, source, **kwargs)
    valid_loader = DataLoader(valid_set, batch_size=1,
                               num_workers=n_cpu, shuffle=False)
    return train_loader, valid_loader

def save_model(save_path, netG, netD, optG, optD ,epoch, **kwargs):
    '''
    Input: save_path (string), netG (state_dict), netD (state_dict), 
    optG (torch.optim), optD(torch.optim)
    '''
    torch.save(netG, save_path +  str(epoch) + "netG.pt")
    torch.save(netD, save_path  +  str(epoch) + "netD.pt")
    torch.save(optG, save_path   +  str(epoch) + "optG.pt")
    torch.save(optD, save_path   +  str(epoch) + "optD.pt")
    if spec:
        torch.save(netD_spec, save_path  +  str(epoch) + "netD_spec.pt")
        torch.save(optD_spec, save_path   +  str(epoch) + "optD_spec.pt")


def run_validate(valid_loader, netG, netD, config):
    disc_losses = []
    gen_losses = []
    feature_losses = []
    reconstruction_losses = []
    sdrs = [] 
    output_aud = None

    fft = Audio2Mel(n_mel_channels=config.n_mel_channels).to(device)
    with torch.no_grad():
        for iterno, x_t in enumerate(valid_loader):
            x_t_0 = x_t[0].unsqueeze(1).float().to(device)
            x_t_1 = x_t[1].unsqueeze(1).float().to(device)
            x_t_2 = x_t[2].unsqueeze(1).float().to(device)
            inp  = torch.cat( (F.pad(x_t_0,(2900,2900), "constant", 0),F.pad(x_t_2, (2900,2900), "constant", 0)), dim=1)
            # s_t = fft(x_t_0)
            x_pred_t = netG(inp,x_t_0.unsqueeze(1)).squeeze(1)
            s_pred_t = fft(x_pred_t)
            s_test = fft(x_t_1)
            s_error = F.l1_loss(s_test, s_pred_t)
            if iterno == int(config.random_sample):
                output_aud = (x_t_0.squeeze(0).squeeze(0).cpu().numpy(), 
                                x_t_1.squeeze(0).squeeze(0).cpu().numpy(), 
                                x_pred_t.squeeze(0).squeeze(0).cpu().numpy())

            # Calculate valid reconstruction loss

            s_test = fft(x_t_1)
            s_error = F.l1_loss(s_test, s_pred_t)

            # Calculate valid reconstruction loss
            sdr = SISDRLoss()
            sdr_loss = sdr(x_pred_t.squeeze(1).unsqueeze(2), x_t_1.squeeze(1).unsqueeze(2))

            D_fake_det = netD(x_pred_t.to(device).detach())
            D_real = netD(x_t_1.to(device))

            # Calculate valid discriminator loss

            loss_D = 0
            for scale in D_fake_det:
                loss_D += F.relu(1 + scale[-1]).mean()
            for scale in D_real:
                loss_D += F.relu(1 - scale[-1]).mean()

            # Calculate valid generator and feature loss

            D_fake = netD(x_pred_t.to(device))
            loss_G = 0
            for scale in D_fake:
                loss_G += -scale[-1].mean()
            loss_feat = 0
            feat_weights = 4.0 / (config.n_layers_D + 1)
            D_weights = 1.0 / config.num_D
            wt = D_weights * feat_weights
            for i in range(config.num_D):
                for j in range(len(D_fake[i]) - 1):
                    loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())
            
            disc_losses.append(loss_D.item())
            gen_losses.append(loss_G.item())
            feature_losses.append(loss_feat.item())
            reconstruction_losses.append(s_error.item())
            sdrs.append(sdr_loss.cpu())
        return np.mean(disc_losses), np.mean(gen_losses), np.mean(feature_losses), np.mean(reconstruction_losses), -1*np.mean(sdrs), output_aud


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
   
    create_saves_directory(config.model_save_dir)
    
    global device
    device = torch.device(f"cuda:{config.gpus[0]}" if torch.cuda.is_available() else "cpu")
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    ModelSelector = ModelFactory(config)
    
    netG = ModelSelector.generator().to(device)
    if config.multi_disc:
        netD, netD_spec = ModelSelector.discriminator()
        netD.to(device)
        netD_spec.to(device)
        optD_spec = torch.optim.Adam(netD_spec.parameters(), lr=config.lr, betas=(config.b1,config.b2))
    else:
        netD = ModelSelector.discriminator().to(device)
    #netG = nn.DataParallel(netG, device_ids=config.gpus)
    #netD = nn.DataParallel(netD.to(device), device_ids=config.gpus)
    #netD_spec = nn.DataParallel(netD_spec.to(device), device_ids=config.gpus)
    fft = Audio2Mel(n_mel_channels=config.n_mel_channels).to(device)
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
                                    verbose = config.verbose
                                    )


    start_epoch=config.start_epoch

    if start_epoch > 0:
        netG.load_state_dict(torch.load(config.model_load_dir +  str(start_epoch-1) + "netG.pt"))
        #netD.load_state_dict(torch.load(config.model_load_dir +  str(start_epoch-1) + "netD.pt"))
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
    if config.multi_spec:
        netD_spec.train()
    steps = 0
    sdr = SISDRLoss()
    best_SDR = 0
    best_reconstruct = 100
    for epoch in range(start_epoch, config.n_epochs):
        if (epoch+1) % config.checkpoint_interval == 0 and epoch != start_epoch:
            save_model(config.model_save_dir, netG.state_dict(), netD.state_dict(), optG,optD,epoch, spec=true, netD_spec= netD_spec.state_dict(), optD_spec = optD_spec )
        for iterno, x_t in enumerate(train_loader):
            x_t_0 = x_t[0].unsqueeze(1).float().to(device)
            x_t_1 = x_t[1].unsqueeze(1).float().to(device)
            x_t_2 = x_t[2].unsqueeze(1).float().to(device)
            #s_t = fft(x_t_0)
            #print(x_t_0.shape)
            inp  = torch.cat( (F.pad(x_t_0,(2900,2900), "constant", 0),F.pad(x_t_2, (2900,2900), "constant", 0)), dim=1)
            x_pred_t = netG(inp,x_t_0.unsqueeze(1)).squeeze(1)
            s_pred_t = fft(x_pred_t)
            s_test = fft(x_t_1)
            s_error = F.l1_loss(s_test, s_pred_t)

            #######################
            # Train Discriminator #
            #######################
            sdr = SISDRLoss()
            sdr_loss = sdr(x_pred_t.squeeze(1).unsqueeze(2), x_t_1.squeeze(1).unsqueeze(2))

            D_fake_det = netD(x_pred_t.to(device).detach())
            D_real = netD(x_t_1.to(device))

            D_fake_det_spec = netD_spec(x_pred_t.squeeze(1).to(device).detach())
            D_real_spec = netD_spec(x_t_1.squeeze(1).to(device))

            loss_D = 0
            loss_D_spec = 0

            for scale in D_fake_det:
                loss_D += F.relu(1 + scale[-1]).mean()
            for scale in D_real:
                loss_D += F.relu(1 - scale[-1]).mean()
            
             
            loss_D_spec += F.relu(1 + D_fake_det_spec[-1]).mean()
             
            loss_D_spec += F.relu(1 - D_real_spec[-1]).mean()

            if epoch >= config.pretrain_epoch:
                netD.zero_grad()
                loss_D.backward()
                optD.step()
                netD_spec.zero_grad()
                loss_D_spec.backward()
                optD_spec.step()

            ###################
            # Train Generator #
            ###################
            D_fake = netD(x_pred_t.to(device))
            D_fake_spec = netD_spec(x_pred_t.squeeze(1).to(device))

            loss_G = 0
            for scale in D_fake:
                loss_G += -scale[-1].mean()
            
            loss_G += -D_fake_spec[-1].mean()*3
            
            loss_feat = 0
            feat_weights = 4.0 / (config.n_layers_D + 1)
            D_weights = 1.0 / config.num_D
            wt = D_weights * feat_weights
            for i in range(config.num_D):
                for j in range(len(D_fake[i]) - 1):
                    loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())
            
            wt = 4.0 / (config.n_layers_D_spec + 1)
            loss_feat_spec = 0
            for k in range(len(D_fake_spec)-1):
                loss_feat_spec += wt * F.l1_loss(D_fake_spec[k], D_real_spec[k].detach())

            netG.zero_grad()
            if epoch >= config.pretrain_epoch:
                (loss_G + config.lambda_feat * loss_feat + config.lambda_feat_spec* loss_feat_spec).backward()
                optG.step()
            else:
                wav_loss = F.l1_loss(x_t_0,x_pred_t)

               # true_spec = torch.stft(input=x_t_0.squeeze(0).squeeze(1),n_fft=1024)
               # real_part, imag_part = true_spec.unbind(-1)
               # true_mag_spec = torch.log10(torch.sqrt(real_part ** 2 + imag_part ** 2) + 1e-5)
               # fake_spec = torch.stft(input=x_pred_t.squeeze(0).squeeze(1),n_fft=1024)
               # real_part, imag_part = fake_spec.unbind(-1)
               # fake_mag_spec = torch.log10(torch.sqrt(real_part ** 2 + imag_part ** 2) + 1e-5)

                #spec_loss = F.l1_loss(true_mag_spec,fake_mag_spec)

                (wav_loss).backward()
                optG.step()
            ######################
            # Update tensorboard #
            ######################
            costs = [[loss_D.item(), loss_G.item(), loss_feat.item(), s_error.item(),-1 *sdr_loss,loss_D_spec.item(),loss_feat_spec.item()]]
            writer.add_scalar("loss/discriminator", costs[-1][0], steps)
            writer.add_scalar("loss/generator", costs[-1][1], steps)
            writer.add_scalar("loss/feature_matching", costs[-1][2], steps)
            writer.add_scalar("loss/mel_reconstruction", costs[-1][3], steps)
            writer.add_scalar("loss/sdr", costs[-1][4], steps)

            sys.stdout.write(f'\r[Epoch {epoch}, Batch {iterno}]:\
                                [Generator Loss: {costs[-1][1]:.4f}]\
                                [Discriminator Loss: {costs[-1][0]:.4f}]\
                                [Feature Loss: {costs[-1][2]:.4f}]\
                                [Reconstruction Loss: {costs[-1][3]:.4f}]\
                                [SDR: {costs[-1][4]:.4f}]')
            wandb.log({
                'Generator Loss':costs[-1][1],
                'Wav Discriminator Loss': costs[-1][0],
                'Spec Discriminator Loss': costs[-1][5],
                'Wav Feature Loss': costs[-1][2],
                'Spec Feature Loss': costs[-1][6],
                'Reconstruction Loss': costs[-1][3],
                'SDR': costs[-1][4],
                'epoch' : epoch
            })

            if steps % config.validation_steps==0: 
                valid_d, valid_g, valid_feat, valid_s, valid_sdr, aud = run_validate(valid_loader, netG, netD, config)
                if steps==0:
                    sf.write('validation_original.wav', aud[1] ,config.sample_rate)
                    sf.write('validation_demucs.wav', aud[0] ,config.sample_rate)
                    wandb.log({"Validation Audio": 
                        [wandb.Audio(
                            'validation_original.wav', 
                            caption= "Original Sample", 
                            sample_rate=config.sample_rate
                        ), 
                        wandb.Audio(
                            'validation_demucs.wav',
                            caption = "Demucs Sample",
                            sample_rate=config.sample_rate
                        )]})
                if valid_sdr > best_SDR:
                    save_model(config.model_save_dir, netG.state_dict(), netD.state_dict(), optG,optD,epoch, spec=true, netD_spec= netD_spec.state_dict(), optD_spec = optD_spec)
                    best_SDR = valid_sdr
                if valid_s < best_reconstruct:
                    save_model(config.model_save_dir, netG.state_dict(), netD.state_dict(), optG,optD,epoch, spec=true, netD_spec= netD_spec.state_dict(), optD_spec = optD_spec)
                    best_reconstruct = valid_s
                wandb.log({
                    'Valid Generator Loss': valid_g,
                    'Valid Discriminator Loss': valid_d,
                    'Valid Feature Loss': valid_feat,
                    'Valid Reconstruction Loss': valid_s,
                    'Valid SDR': valid_sdr
                })
                sf.write(f'generated_{steps}.wav', aud[2] ,config.sample_rate)
                wandb.log({f'{steps} Steps': 
                [wandb.Audio(
                    f'generated_{steps}.wav', 
                    caption= f'Generated Audio, {steps} Steps', 
                    sample_rate=44100
                )]
                })

                fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(10,7))

                titles = ['Demucs','Original','MSG']
                axes = [ax1,ax2,ax3]

                for i in range(3):
                
                    spec_data = librosa.feature.melspectrogram(y=aud[i], sr=44100, n_mels=128,
                                                        fmax=8000)
                    S_dB = librosa.power_to_db(spec_data, ref=np.max)
                    _fig_ax = librosa.display.specshow(S_dB, x_axis='time',
                                                    y_axis='mel', sr=config.sample_rate,
                                                    fmax=8000, ax=axes[i])
                    axes[i].set(title=titles[i])
                plt.savefig(f'spectrogram_{steps}.png')
                wandb.log({f'Spectrograms, {steps} Steps': 
                    [wandb.Image(imageio.imread(f'spectrogram_{steps}.png'))]})
            steps+=1

class ParameterError(Exception):
    pass

if __name__ == '__main__':
    main()

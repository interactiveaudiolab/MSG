import os
import sys
import argparse
import re
import yaml
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

from libraries.models.MelGAN import Audio2Mel, GeneratorMel, DiscriminatorMel, SISDRLoss
from libraries.mlp.WaveDataset import MusicDataset



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
    netG = GeneratorMel(
        params['n_mel_channels'], params['ngf'], params['n_residual_layers'],params['skip_cxn']
        ).cuda()
    netD = DiscriminatorMel(
            params['num_D'], params['ndf'], params['n_layers_D'], params['downsamp_factor']
        ).cuda()
    fft = Audio2Mel(n_mel_channels=params['n_mel_channels']).cuda()

    optG = torch.optim.Adam(netG.parameters(), lr=params['lr'], betas=(params['b1'],params['b2']))
    optD = torch.optim.Adam(netD.parameters(), lr=params['lr'], betas=(params['b1'],params['b2']))

    ###############
    ### DATASET ###
    ## CREATION ###
    ###############

    train_dirty = []
    train_clean = []
    val_dirty = []
    val_clean = []
    dirty_data = [elem for elem in os.listdir(params['separated_sources_path']) if "bass" in elem]

    for s in dirty_data:
        if np.random.rand() < .9:
            train_dirty.append(params['separated_sources_path'] + '/' + s)
            train_clean.append(params['original_sources_path'] +'/' + s)
        else:
            val_dirty.append(params['separated_sources_path']+'/' + s)
            val_clean.append(params['original_sources_path'] + '/' + s)

    ds_valid = MusicDataset(val_dirty, val_clean, 44100,44100)
    ds_train = MusicDataset(train_dirty, train_clean, 44100, 44100)

    valid_loader = DataLoader(ds_valid, batch_size=params['batch_size'], 
                                num_workers=params['n_cpu'], shuffle=False)
    train_loader = DataLoader(ds_train, batch_size=params['batch_size'], 
                                num_workers=params['n_cpu'], shuffle=True)

    start_epoch=params['start_epoch']

    if start_epoch > 0:
        netG.load_state_dict(torch.load(params['model_load_dir'] +  str(start_epoch-1) + "netG.pt"))
        netD.load_state_dict(torch.load(params['model_load_dir'] +  str(start_epoch-1) + "netD.pt"))
        optG.load_state_dict(torch.load(params['model_load_dir'] +  str(start_epoch-1) +
                                                                    "optG.pt").state_dict())
        optD.load_state_dict(torch.load(params['model_load_dir'] +  str(start_epoch-1) +
                                                                    "optD.pt").state_dict())



    ###################
    ##### TRAINING ####
    ###### LOOP #######
    ###################
    writer = SummaryWriter()
    costs = []
    netG.train()
    netD.train()
    steps = 0
    sdr = SISDRLoss()

    for epoch in range(start_epoch, params['n_epochs']):
        if (epoch+1) % 100 == 0 and epoch != start_epoch:
            torch.save(netG.state_dict(), params['model_save_dir'] +  str(epoch) + "netG.pt")
            torch.save(netD.state_dict(), params['model_save_dir']  +  str(epoch) + "netD.pt")
            torch.save(optG, params['model_save_dir']  +  str(epoch) + "optG.pt")
            torch.save(optD, params['model_save_dir']  +  str(epoch) + "optD.pt")
        for iterno, x_t in enumerate(train_loader):
            x_t_0 = x_t[0].unsqueeze(1).float().cuda()
            x_t_1 = x_t[1].unsqueeze(1).float().cuda()
            s_t = fft(x_t_0)
            x_pred_t = netG(s_t,x_t_0)
            s_pred_t = fft(x_pred_t)
            s_test = fft(x_t_1)
            s_error = F.l1_loss(s_test, s_pred_t)

            #######################
            # Train Discriminator #
            #######################
            sdr = SISDRLoss()


            sdr_loss = sdr(x_pred_t.squeeze(1).unsqueeze(2), x_t_1.squeeze(1).unsqueeze(2))

            D_fake_det = netD(x_pred_t.cuda().detach())
            D_real = netD(x_t_1.cuda())

            loss_D = 0
            for scale in D_fake_det:
                loss_D += F.relu(1 + scale[-1]).mean()
            for scale in D_real:
                loss_D += F.relu(1 - scale[-1]).mean()
            netD.zero_grad()
            loss_D.backward()
            optD.step()
            ###################
            # Train Generator #
            ###################
            D_fake = netD(x_pred_t.cuda())
            loss_G = 0
            for scale in D_fake:
                loss_G += -scale[-1].mean()
            loss_feat = 0
            feat_weights = 4.0 / (params['n_layers_D'] + 1)
            D_weights = 1.0 / params['num_D']
            wt = D_weights * feat_weights
            for i in range(params['num_D']):
                for j in range(len(D_fake[i]) - 1):
                    loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())
            netG.zero_grad()
            (loss_G + params['lambda_feat'] * loss_feat).backward()
            optG.step()
            ######################
            # Update tensorboard #
            ######################
            costs = [[loss_D.item(), loss_G.item(), loss_feat.item(), s_error.item(),-1 *sdr_loss]]
            writer.add_scalar("loss/discriminator", costs[-1][0], steps)
            writer.add_scalar("loss/generator", costs[-1][1], steps)
            writer.add_scalar("loss/feature_matching", costs[-1][2], steps)
            writer.add_scalar("loss/mel_reconstruction", costs[-1][3], steps)
            writer.add_scalar("loss/sdr_loss", costs[-1][4], steps)
            steps += 1

            sys.stdout.write(f'\r[Epoch {epoch}, Batch {iterno}]:\
                                [Generator Loss: {costs[-1][1]:.4f}]\
                                [Discriminator Loss: {costs[-1][0]:.4f}]\
                                [Feature Loss: {costs[-1][2]:.4f}]\
                                [Reconstruction Loss: {costs[-1][3]:.4f}]\
                                [SDR Loss: {costs[-1][4]:.4f}]')

class ParameterError(Exception):
    pass

if __name__ == '__main__':
    main()

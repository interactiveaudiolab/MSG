import os
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch


def train(model_dict, train_loader, model_parameters, local_path, experiment_dir):
    for epoch in range(model_parameters['start_epoch'], model_parameters['n_epochs']):
        if (epoch + 1) % 100 == 0 and epoch != model_parameters['start_epoch']:
            torch.save(model_dict['netG'].state_dict(), local_path + experiment_dir + str(epoch) + "netG.pt")
            torch.save(model_dict['netD'].state_dict(), local_path + experiment_dir + str(epoch) + "netD.pt")
            torch.save(model_dict['optG'], local_path + experiment_dir + str(epoch) + "optG.pt")
            torch.save(model_dict['optD'], local_path + experiment_dir + str(epoch) + "optD.pt")
        for iterno, x_t in enumerate(train_loader):
            x_t_0 = x_t[0].unsqueeze(1).float().to('device')
            x_t_1 = x_t[1].unsqueeze(1).float().to('device')
            s_t = model_dict['fft'](x_t_0)
            x_pred_t = model_dict['netG'](s_t, x_t_0)
            s_pred_t = model_dict['fft'](x_pred_t)
            s_test = model_dict['fft'](x_t_1)
            s_error = F.l1_loss(s_test, s_pred_t)

            #######################
            # Train Discriminator #
            #######################
            sdr = model_parameters['SISDRLoss']()

            sdr_loss = sdr(x_pred_t.squeeze(1).unsqueeze(2), x_t_1.squeeze(1).unsqueeze(2))

            D_fake_det = model_dict['netD'](x_pred_t.to('device').detach())
            D_real = model_dict['netD'](x_t_1.to('device'))

            loss_D = 0
            for scale in D_fake_det:
                loss_D += F.relu(1 + scale[-1]).mean()
            for scale in D_real:
                loss_D += F.relu(1 - scale[-1]).mean()
            model_dict['netD'].zero_grad()
            loss_D.backward()
            model_dict['optD'].step()
            ###################
            # Train Generator #
            ###################
            D_fake = model_dict['netD'](x_pred_t.to('device'))
            loss_G = 0
            for scale in D_fake:
                loss_G += -scale[-1].mean()
            loss_feat = 0
            feat_weights = 4.0 / (model_parameters['n_layers_D'] + 1)
            D_weights = 1.0 / model_parameters['num_D']
            wt = D_weights * feat_weights
            for i in range(model_parameters['num_D']):
                for j in range(len(D_fake[i]) - 1):
                    loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())
            model_dict['netG'].zero_grad()
            (loss_G + model_parameters['lambda_feat'] * loss_feat).backward()
            model_dict['optG'].step()
            ######################
            # Update tensorboard #
            ######################
            costs = [loss_D.item(), loss_G.item(), loss_feat.item(), s_error.item(), -1 * sdr_loss]
            model_parameters['writer'].add_scalar("loss/discriminator", costs[0], model_parameters['steps'])
            model_parameters['writer'].add_scalar("loss/generator", costs[1], model_parameters['steps'])
            model_parameters['writer'].add_scalar("loss/feature_matching", costs[2], model_parameters['steps'])
            model_parameters['writer'].add_scalar("loss/mel_reconstruction", costs[3], model_parameters['steps'])
            model_parameters['writer'].add_scalar("loss/sdr_loss", costs[4], model_parameters['steps'])
            model_parameters['steps'] += 1

            sys.stdout.write(f'\r[Epoch {epoch}, Batch {iterno}]:\
                                [Generator Loss: {costs[1]:.4f}]\
                                [Discriminator Loss: {costs[0]:.4f}]\
                                [Feature Loss: {costs[2]:.4f}]\
                                [Reconstruction Loss: {costs[3]:.4f}]\
                                [SDR Loss: {costs[4]:.4f}]')

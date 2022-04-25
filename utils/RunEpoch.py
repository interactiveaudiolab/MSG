import torch
import torch.nn.functional as F
from .losses import *
from .save_and_log import  *

import numpy as np


def runEpoch(loader, config, netG, netD, optG, optD, device, epoch,
                steps, writer, gen_autoclip,disc_autoclip,adv_autobalancer ,optD_spec=None, netD_spec=None,
                validation=False):
    costs = [[0,0,0,0,0,0,0]]
    gan_loss_calculator = GANLoss(netD)
    output_aud = [np.array([]),np.array([]),np.array([])]
    validation_song_seconds = 0
    for iterno, x_t in enumerate(loader):
        if config.mono:
            x_t_0 = x_t[0].unsqueeze(1).float().to(device)
            x_t_1 = x_t[1].unsqueeze(1).float().to(device)
            x_t_2 = x_t[2].unsqueeze(1).float().to(device)
        else:
            x_t_0 = x_t[0].float().to(device)
            x_t_1 = x_t[1].float().to(device)
            x_t_2 = x_t[2].float().to(device)
            x_t_1_mono = (x_t_1[:, 0, :] + x_t_1[:, 1, :])
            x_t_1_mono /= torch.max(torch.abs(x_t_1_mono))

        inp = F.pad(x_t_0, (4000, 4000), "constant", 0)

        x_pred_t = netG(inp, x_t_0.unsqueeze(1)).squeeze(1)
        wav_loss = F.l1_loss(x_t_1, x_pred_t)

        if not config.mono:
            x_pred_t_mono = (x_pred_t[:, 0, :] + x_pred_t[:, 1, :])
            x_pred_t_mono /= torch.max(torch.abs(x_pred_t_mono))
            mel_reconstruction_loss = mel_spec_loss(x_pred_t_mono.squeeze(1),x_t_1_mono.squeeze(1))
        else:
            mel_reconstruction_loss = mel_spec_loss(x_pred_t.squeeze(1),x_t_1.squeeze(1))

        #######################
        # L1, SDR Loss        #
        #######################
        
        sdr = SISDRLoss()
        if config.mono:
            sdr_loss = sdr(x_pred_t.squeeze(1).unsqueeze(2),
                           x_t_1.squeeze(1).unsqueeze(2))
        else:
            sdr_loss = sdr(x_pred_t_mono.unsqueeze(2), x_t_1_mono.unsqueeze(2))

        #######################
        # Train Discriminator #
        #######################

        fake = x_pred_t.to(device)
        real = x_t_1.to(device)
        loss_D = gan_loss_calculator.discriminator_loss(fake, real)


        if not validation and epoch>config.pretrain_epoch:
            netD.zero_grad()
            loss_D.backward()
            optD.step()

        ###################
        # Train Generator #
        ###################
        loss_G, loss_feat = gan_loss_calculator.generator_loss(fake,real)

        if not validation:
            netG.zero_grad()
            if epoch >= config.pretrain_epoch:
                if config.adv_only:
                    total_generator_loss = sum(adv_autobalancer(loss_G,loss_feat))
                else:   
                    total_generator_loss = sum(adv_autobalancer(loss_G,loss_feat,mel_reconstruction_loss))
                total_generator_loss.backward()
            else:
                mel_reconstruction_loss.backward()
            _, gen_grad_norm = gen_autoclip(netG)
            optG.step()
            
            costs = [
                [loss_D.item(), loss_G.item(), loss_feat.item(),
                mel_reconstruction_loss.item(),
                -1 * sdr_loss, wav_loss.item(),gen_grad_norm]]
        else:
            curr_costs = [loss_D.item(), loss_G.item(), loss_feat.item(),
                mel_reconstruction_loss.item(),
                -1 * sdr_loss, wav_loss.item(),0]
            for i in range(len(costs[0])):
                costs[0][i] += curr_costs[i]
        # Call basic log info
        if not validation:
            basic_logs(costs, writer, steps, epoch, iterno)
        else:
            validation_writer(epoch,steps)
        steps += 1
        if validation and x_t[3][0] == config.validation_song:
            if config.mono and validation_song_seconds >= config.validation_song_start and validation_song_seconds <= config.validation_song_end:
                output_aud[0] = np.concatenate((output_aud[0], x_t_0.squeeze(0).squeeze(0).cpu().numpy()))
                output_aud[1] = np.concatenate((output_aud[1], x_t_1.squeeze(0).squeeze(0).cpu().numpy()))
                output_aud[2] = np.concatenate((output_aud[2], x_pred_t.squeeze(0).squeeze(0).cpu().numpy()))
            validation_song_seconds += 1
                #output_aud = (x_t_0.squeeze(0).cpu().numpy(),
                              #x_t_1.squeeze(0).cpu().numpy(),
                              #x_pred_t.squeeze(0).cpu().numpy())
    if validation:
        for i in range(len(costs[0])):
            costs[0][i] /= (iterno+1)
    return steps, costs, output_aud

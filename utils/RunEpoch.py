import torch
import torch.nn.functional as F
from .losses import *
from .save_and_log import  *

import numpy as np


def runEpoch(loader, config, netG, netD, optG, optD, fft, device, epoch,
                steps, writer, optD_spec=None, netD_spec=None,
                validation=False):
    costs = [[0,0,0,0,0,0,0]]
    output_aud = None
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

        inp = F.pad(x_t_0, (2900, 2900), "constant", 0)

        x_pred_t = netG(inp, x_t_0.unsqueeze(1)).squeeze(1)
        wav_loss = F.l1_loss(x_t_1, x_pred_t)

        if not config.mono:
            x_pred_t_mono = (x_pred_t[:, 0, :] + x_pred_t[:, 1, :])
            x_pred_t_mono /= torch.max(torch.abs(x_pred_t_mono))
            s_pred_t = fft(x_pred_t_mono.unsqueeze(1))
            s_test = fft(x_t_1_mono.unsqueeze(1))
        else:
            s_pred_t = fft(x_pred_t)
            s_test = fft(x_t_1)

        #######################
        # L1, SDR Loss        #
        #######################
        s_error = F.l1_loss(s_test, s_pred_t)
        sdr = SISDRLoss()
        if config.mono:
            sdr_loss = sdr(x_pred_t.squeeze(1).unsqueeze(2),
                           x_t_1.squeeze(1).unsqueeze(2))
        else:
            sdr_loss = sdr(x_pred_t_mono.unsqueeze(2), x_t_1_mono.unsqueeze(2))

        #######################
        # Train Discriminator #
        #######################
        if config.mono:
            D_fake_det_spec, D_real_spec = disc_outputs(
                config, x_pred_t.squeeze(1), x_t_1.squeeze(1), device, netD_spec)
        else:
            D_fake_det_spec, D_real_spec = disc_outputs(
                config, x_pred_t_mono, x_t_1_mono, device, netD_spec)

        D_fake_det = netD(x_pred_t.to(device).detach())
        D_real = netD(x_t_1.to(device))


        loss_D = 0
        loss_D_spec = 0
        loss_D += waveform_discriminator_loss(D_fake_det, D_real)
        loss_D_spec += spectral_discriminator_loss(D_fake_det_spec, D_real_spec)

        if epoch >= config.pretrain_epoch and (
                epoch <= 100 or epoch % config.l1interval != 0) and not validation:
            netD.zero_grad()
            loss_D.backward()
            optD.step()
            netD_spec.zero_grad()
            loss_D_spec.backward()
            if config.multi_disc:
                optD_spec.step()

        ###################
        # Train Generator #
        ###################
        D_fake = netD(x_pred_t.to(device))
        if config.mono:
            D_fake_spec = netD_spec(x_pred_t.squeeze(1).to(device))
        else:
            D_fake_spec = netD_spec(x_pred_t_mono.to(device))

        loss_G = Gen_loss(D_fake, D_fake_spec)
        loss_feat, loss_feat_spec = feature_loss(config, D_fake, D_real, D_fake_spec, D_real_spec)

        netG.zero_grad()
        if not validation:
            if epoch >= config.pretrain_epoch and (
                    epoch <= 100 or epoch % config.l1interval != 0):
                torch.autograd.set_detect_anomaly(True)
                (
                            loss_G + config.lambda_feat * loss_feat + config.lambda_feat_spec * loss_feat_spec).backward()
                optG.step()
            else:
                (wav_loss).backward()
                optG.step()
        if not validation:
            costs = [
                [loss_D.item(), loss_G.item(), loss_feat.item(),
                s_error.item(),
                -1 * sdr_loss, loss_D_spec.item(), loss_feat_spec.item()]]
        else:
            curr_costs = [loss_D.item(), loss_G.item(), loss_feat.item(),
                s_error.item(),
                -1 * sdr_loss, loss_D_spec.item(), loss_feat_spec.item()]
            for i in range(len(costs[0])):
                costs[0][i] += curr_costs[i]
        # Call basic log info
        if not validation:
            basic_logs(costs, writer, steps, epoch, iterno)
        else:
            validation_writer(epoch,steps)
        steps += 1
        if validation and iterno == int(config.random_sample):
            if config.mono:
                output_aud = (x_t_0.squeeze(0).squeeze(0).cpu().numpy(),
                              x_t_1.squeeze(0).squeeze(0).cpu().numpy(),
                              x_pred_t.squeeze(0).squeeze(0).cpu().numpy())
            else:
                output_aud = (x_t_0.squeeze(0).cpu().numpy(),
                              x_t_1.squeeze(0).cpu().numpy(),
                              x_pred_t.squeeze(0).cpu().numpy())
    if validation:
        for i in range(len(costs[0])):
            costs[0][i] /= (iterno+1)
    return steps, costs, output_aud

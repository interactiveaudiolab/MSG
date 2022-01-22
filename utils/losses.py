import torch.nn.functional as F

def disc_outputs(config, x_pred_t, x_t_1, device, netD_spec):
    D_fake_det_spec = netD_spec(
        x_pred_t.to(device).detach())
    D_real_spec = netD_spec(x_t_1.to(device))
    return D_fake_det_spec, D_real_spec

def Gen_loss(D_fake, D_fake_spec):
    loss_G = 0
    for scale in D_fake:
        loss_G += -scale[-1].mean()

    loss_G += -D_fake_spec[-1].mean()
    return loss_G

def waveform_discriminator_loss(D_fake, D_real):
    loss_D = 0
    for scale in D_fake:
        loss_D += F.relu(1 + scale[-1]).mean()
    for scale in D_real:
        loss_D += F.relu(1 - scale[-1]).mean()
    return loss_D


def spectral_discriminator_loss(fake, real):
    loss_D_spec = 0
    loss_D_spec += F.relu(1 + fake[-1]).mean()

    loss_D_spec += F.relu(1 - real[-1]).mean()
    return loss_D_spec

def feature_loss(config, D_fake, D_real, D_fake_spec, D_real_spec):
    loss_feat = 0
    feat_weights = 4.0 / (config.n_layers_D + 1)
    D_weights = 1.0 / config.num_D
    wt = D_weights * feat_weights
    for i in range(config.num_D):
        for j in range(len(D_fake[i]) - 1):
            loss_feat += wt * F.l1_loss(D_fake[i][j],
                                        D_real[i][j].detach())

    wt = 4.0 / (config.n_layers_D_spec + 1)
    loss_feat_spec = 0
    for k in range(len(D_fake_spec) - 1):
        loss_feat_spec += wt * F.l1_loss(D_fake_spec[k],
                                         D_real_spec[k].detach())
    return loss_feat, loss_feat_spec

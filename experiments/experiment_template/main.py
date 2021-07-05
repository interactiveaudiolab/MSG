import os
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

from models.MelGAN import Audio2Mel, GeneratorMel, DiscriminatorMel, SISDRLoss
from datasets.WaveDataset import MusicDataset


np.random.seed(0)
torch.manual_seed(0)

##############################
# Make These hyperparameters #
#############################

local_path = '/drive/MelGan-Imputation/'
start_epoch = 0 # epoch to start training from
n_epochs = 3000 # number of epochs of training
pretrain_epoch = 100
dataset_name = 'MUSDB-18' # name of the dataset
batch_size = 16 # size of the batches
lr = 0.0001 # adam: learning rate
b1 = 0.5 # adam: decay of first order momentum of gradient
b2 = 0.9 # adam: decay of first order momentum of gradient
decay_epoch = 100 # epoch from which to start lr decay
n_cpu = 1 # number of cpu threads to use during batch generation
img_height = 128 # size of image height
img_width = 128 # size of image width
channels = 1 # number of image channels
sample_interval = 100 # interval between sampling of images from generators
checkpoint_interval = 100 # interval between model checkpoints

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#resource.setrlimit(resource.RLIMIT_NOFILE, (4096, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))

n_mel_channels = 80
ngf = 32
n_residual_layers = 3

num_D = 3
ndf = 16
n_layers_D = 4
downsamp_factor = 4
lambda_feat = 10
save_interval = 20
log_interval = 100
experiment_dir = 'saves_pretrain/'

netG = GeneratorMel(n_mel_channels, ngf, n_residual_layers,skip_cxn=True).cuda()
netD = DiscriminatorMel(
        num_D, ndf, n_layers_D, downsamp_factor
    ).cuda()
fft = Audio2Mel(n_mel_channels=n_mel_channels).cuda()

optG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
optD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))

###############
### DATASET ###
## CREATION ###
###############
dirty_path ='/drive/MelGan-Imputation/datasets/demucs_train_flattened'
clean_path ='/drive/MelGan-Imputation/datasets/original_train_sources'

train_dirty = []
train_clean = []
val_dirty = []
val_clean = []
dirty_data = [elem for elem in os.listdir(dirty_path) if "bass" in elem]

for s in dirty_data:
    if np.random.rand() < .9:
        train_dirty.append(dirty_path + '/' + s)
        train_clean.append(clean_path +'/' + s)
    else:
        val_dirty.append(dirty_path +'/' + s)
        val_clean.append(clean_path + '/' + s)

ds_valid = MusicDataset(val_dirty, val_clean, 44100,44100)
ds_train = MusicDataset(train_dirty, train_clean, 44100, 44100)

valid_loader = DataLoader(ds_valid, batch_size=batch_size, num_workers=n_cpu, shuffle=False)
train_loader = DataLoader(ds_train, batch_size=batch_size, num_workers=n_cpu, shuffle=True)

if start_epoch > 0:
    netG.load_state_dict(torch.load(local_path + experiment_dir +  str(start_epoch-1) + "netG.pt"))
    netD.load_state_dict(torch.load(local_path + experiment_dir +  str(start_epoch-1) + "netD.pt"))
    optG.load_state_dict(torch.load(local_path + experiment_dir +  str(start_epoch-1) +
                                                                "optG.pt").state_dict())
    optD.load_state_dict(torch.load(local_path + experiment_dir +  str(start_epoch-1) +
                                                                "optD.pt").state_dict())



###################
##### TRAINING ####
###### LOOP #######
###################
writer = SummaryWriter()
costs = []
results = []
netG.train()
netD.train()
steps = 0
sdr = SISDRLoss()

for epoch in range(start_epoch, n_epochs):
    if (epoch+1) % 100 == 0 and epoch != start_epoch:
        torch.save(netG.state_dict(), local_path + experiment_dir +  str(epoch) + "netG.pt")
        torch.save(netD.state_dict(), local_path + experiment_dir +  str(epoch) + "netD.pt")
        torch.save(optG, local_path + experiment_dir +  str(epoch) + "optG.pt")
        torch.save(optD, local_path + experiment_dir +  str(epoch) + "optD.pt")
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
        feat_weights = 4.0 / (n_layers_D + 1)
        D_weights = 1.0 / num_D
        wt = D_weights * feat_weights
        for i in range(num_D):
            for j in range(len(D_fake[i]) - 1):
                loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())
        netG.zero_grad()
        (loss_G + lambda_feat * loss_feat).backward()
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

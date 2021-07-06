import os
import sys
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch

from models.MelGAN import Audio2Mel, GeneratorMel, DiscriminatorMel, SISDRLoss
from datasets.WaveDataset import MusicDataset
from experiments.experiment_template.train import train

np.random.seed(0)
torch.manual_seed(0)

##############################
# Make These hyperparameters #
#############################

local_path = '/drive/MelGan-Imputation/'
start_epoch = 0  # epoch to start training from
n_epochs = 3000  # number of epochs of training
pretrain_epoch = 100
dataset_name = 'MUSDB-18'  # name of the dataset
batch_size = 16  # size of the batches
lr = 0.0001  # adam: learning rate
b1 = 0.5  # adam: decay of first order momentum of gradient
b2 = 0.9  # adam: decay of first order momentum of gradient
decay_epoch = 100  # epoch from which to start lr decay
n_cpu = 1  # number of cpu threads to use during batch generation
img_height = 128  # size of image height
img_width = 128  # size of image width
channels = 1  # number of image channels
sample_interval = 100  # interval between sampling of images from generators
checkpoint_interval = 100  # interval between model checkpoints

cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# resource.setrlimit(resource.RLIMIT_NOFILE, (4096, resource.getrlimit(resource.RLIMIT_NOFILE)[1]))

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

netG = GeneratorMel(n_mel_channels, ngf, n_residual_layers, skip_cxn=True).to(device)
netD = DiscriminatorMel(
    num_D, ndf, n_layers_D, downsamp_factor
).to(device)
fft = Audio2Mel(n_mel_channels=n_mel_channels).to(device)

optG = torch.optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.9))
optD = torch.optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.9))

###############
### DATASET ###
## CREATION ###
###############
dirty_path = '/drive/MelGan-Imputation/datasets/demucs_train_flattened'
clean_path = '/drive/MelGan-Imputation/datasets/original_train_sources'

train_dirty = []
train_clean = []
val_dirty = []
val_clean = []
dirty_data = [elem for elem in os.listdir(dirty_path) if "bass" in elem]

for s in dirty_data:
    if np.random.rand() < .9:
        train_dirty.append(dirty_path + '/' + s)
        train_clean.append(clean_path + '/' + s)
    else:
        val_dirty.append(dirty_path + '/' + s)
        val_clean.append(clean_path + '/' + s)

ds_valid = MusicDataset(val_dirty, val_clean, 44100, 44100)
ds_train = MusicDataset(train_dirty, train_clean, 44100, 44100)

valid_loader = DataLoader(ds_valid, batch_size=batch_size, num_workers=n_cpu, shuffle=False)
train_loader = DataLoader(ds_train, batch_size=batch_size, num_workers=n_cpu, shuffle=True)

if start_epoch > 0:
    netG.load_state_dict(torch.load(local_path + experiment_dir + str(start_epoch - 1) + "netG.pt"))
    netD.load_state_dict(torch.load(local_path + experiment_dir + str(start_epoch - 1) + "netD.pt"))
    optG.load_state_dict(torch.load(local_path + experiment_dir + str(start_epoch - 1) +
                                    "optG.pt").state_dict())
    optD.load_state_dict(torch.load(local_path + experiment_dir + str(start_epoch - 1) +
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
model_dict = {
    'netG': netG,
    'netD': netD,
    'optG': optG,
    'optD': optD,
    'fft': fft
}
model_parameters = {
    'start_epoch': start_epoch,
    'n_epochs': n_epochs,
    'SISDRLoss': SISDRLoss,
    'writer': writer,
    'steps': steps,
    'n_layers_D': n_layers_D,
    'num_D': num_D,
    'lambda_feat': lambda_feat,
    'device': device
}

train(model_dict, train_loader, model_parameters, local_path, experiment_dir)

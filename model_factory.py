import torch
from models.MelGAN import Audio2Mel, GeneratorMel, DiscriminatorMel, SISDRLoss

class ModelFactory():
    def __init__(self, model, netG_params, netD_params ,optimizer, optG_params, optD_params):
        self.model = model
        self.netG_params = netG_params
        self.netD_params = netD_params
        self.optimizer = optimizer
        self.optG_params = optG_params
        self.optD_params = optD_params
    def getModel(self):
        if self.model == 'MelGAN':
            netG = GeneratorMel(self.netG_params)
            netD = DiscriminatorMel(self.netD_params)
        if self.optimizer == 'Adam':
            optG = torch.optim.Adam(netG.parameters(), lr=self.optG_params.lr, betas=(self.optG_params.b1,self.optG_params.b2))
            optD = torch.optim.Adam(netD.parameters(), lr=self.optD_params.lr, betas=(self.optD_params.b1,self.optD_params.b2))
        if self.optimizer == 'SGD':
            optG = torch.optim.SGD(netG.parameters(), lr=self.optG_params.lr)
            optD = torch.optim.SGD(netD.parameters(), lr=self.optD_params.lr)
        return netG, netD, optG, optD